import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import ConvexHull
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import ConvexHull
from pathlib import Path
from typing import Dict
import mediapipe as mp


# ======================================================================
#  INTEGRATED FEATURE EXTRACTION + SEVERITY PREDICTION
# ======================================================================
class MediaPipeFeatureExtractor:
    """Extract facial & body features and predict malnutrition severity"""

    def __init__(self, model_path="model.json"):
        # Initialize mediapipe models
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

        # Load trained XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # These must match your model.json feature order
        self.features = [
            "left_eye_x_norm", "left_eye_y_norm",
            "right_eye_x_norm", "right_eye_y_norm",
            "nose_x_norm", "nose_y_norm",
            "face_height", "face_aspect_ratio",
            "mouth_aspect_ratio", "jaw_angle",
            "shoulder_width", "neck_shoulder_ratio"
        ]

        print("✅ Model and MediaPipe initialized successfully.")

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION FROM SINGLE IMAGE
    # ------------------------------------------------------------------
    def extract_from_image(self, image_path: str) -> Dict:
        """Extract features and predict malnutrition severity from a single image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {"image_path": image_path, "error": "Failed to load image"}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            features = {"image_path": str(image_path), "filename": Path(image_path).name}

            # Face features
            face_results = self.face_mesh.process(image_rgb)
            if not face_results.multi_face_landmarks:
                features["face_detected"] = False
                features["severity_level"] = "No face detected"
                return features

            landmarks = face_results.multi_face_landmarks[0].landmark
            features.update(self._extract_face_features(landmarks, w, h))

            # Pose features
            pose_results = self.pose.process(image_rgb)
            if pose_results.pose_landmarks:
                features.update(self._extract_pose_features(pose_results.pose_landmarks.landmark))
            else:
                # Default values if pose not detected
                features["shoulder_width"] = 0.0
                features["neck_shoulder_ratio"] = 0.0

            features["face_detected"] = True

            # Once features are ready — predict severity
            features.update(self._predict_severity(features))

            return features

        except Exception as e:
            return {"image_path": image_path, "error": str(e)}

    # ------------------------------------------------------------------
    # FACE FEATURE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_face_features(self, landmarks, w: int, h: int) -> Dict:
        """Extract geometric features from face landmarks"""
        features = {}
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # Key landmark indices
        left_eye, right_eye, nose_tip = 33, 263, 1
        mouth_upper, mouth_lower, chin, forehead = 13, 14, 152, 10
        left_cheek, right_cheek = 234, 454

        iod = np.linalg.norm(points[left_eye] - points[right_eye])
        iod = max(iod, 1)

        # Normalized coordinates
        features["left_eye_x_norm"] = points[left_eye][0] / w
        features["left_eye_y_norm"] = points[left_eye][1] / h
        features["right_eye_x_norm"] = points[right_eye][0] / w
        features["right_eye_y_norm"] = points[right_eye][1] / h
        features["nose_x_norm"] = points[nose_tip][0] / w
        features["nose_y_norm"] = points[nose_tip][1] / h

        # Face dimensions
        features["face_height"] = abs(points[forehead][1] - points[chin][1]) / iod
        features["face_aspect_ratio"] = features["face_height"] / (
            abs(points[right_eye][0] - points[left_eye][0]) / iod
        )

        # Mouth features
        mouth_width = np.linalg.norm(points[61] - points[291])
        mouth_height = np.linalg.norm(points[mouth_upper] - points[mouth_lower])
        features["mouth_aspect_ratio"] = mouth_width / max(mouth_height, 0.01)

        # Jaw angle
        left_jaw, chin_pt, right_jaw = points[234], points[chin], points[454]
        vec1, vec2 = left_jaw - chin_pt, right_jaw - chin_pt
        cos_angle = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6
        )
        features["jaw_angle"] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return features

    # ------------------------------------------------------------------
    # POSE FEATURE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_pose_features(self, landmarks) -> Dict:
        """Extract shoulder/neck-based body metrics"""
        features = {}
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]

        shoulder_width = np.linalg.norm(
            np.array([left_shoulder.x, left_shoulder.y]) -
            np.array([right_shoulder.x, right_shoulder.y])
        )
        neck_length = abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2)

        features["shoulder_width"] = shoulder_width
        features["neck_shoulder_ratio"] = neck_length / max(shoulder_width, 0.01)

        return features

    # ------------------------------------------------------------------
    # SEVERITY PREDICTION
    # ------------------------------------------------------------------
    def _predict_severity(self, features: Dict) -> Dict:
        """Run XGBoost model to predict malnutrition severity"""
        try:
            df = pd.DataFrame([{k: features.get(k, 0.0) for k in self.features}])
            dtest = xgb.DMatrix(df[self.features])
            pred = self.model.predict(dtest)

            severity_score = float(pred.flatten()[0])*1000

            # Convert numeric score → category
            if 0.24 < severity_score < 0.25:
                severity = "Normal"
            elif severity_score < 0.24:
                severity = "Mild"
            elif severity_score < 0.75:
                severity = "Moderate"
            else:
                severity = "Severe"

            return {
                "severity_score": severity_score,
                "severity_level": severity
            }

        except Exception as e:
            return {"severity_error": str(e)}

    # ------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------
    def extract_from_folder(self, folder_path: Path, split_name: str = "test") -> pd.DataFrame:
        """Extract features + severity predictions for all images in a folder"""
        folder_path = Path(folder_path)
        image_paths = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        print(f"Processing {len(image_paths)} images from {split_name}...")

        results = []
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images...")
            result = self.extract_from_image(img_path)
            result["split"] = split_name
            results.append(result)

        df = pd.DataFrame(results)
        detected = df["face_detected"].sum() if "face_detected" in df else 0
        print(f"✅ Done. Faces detected: {detected}/{len(df)}")
        return df

    def __del__(self):
        self.face_mesh.close()
        self.pose.close()