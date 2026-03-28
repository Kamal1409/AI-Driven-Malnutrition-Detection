import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import pickle
import warnings


class MalnutritionPredictor:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.model = None

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

    def extract_face_features(self, image):
        """Extract facial landmarks using MediaPipe Face Mesh"""
        features = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Extract key facial points
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)

            # Calculate face shape features
            # Face width (distance between cheekbones)
            face_width = np.linalg.norm(landmarks[234] - landmarks[454])

            # Face length (forehead to chin)
            face_length = np.linalg.norm(landmarks[10] - landmarks[152])

            # Cheek hollowness (distance variations)
            left_cheek_depth = np.mean([landmarks[i][2] for i in [123, 116, 117]])
            right_cheek_depth = np.mean([landmarks[i][2] for i in [352, 345, 346]])

            # Eye socket depth
            left_eye_depth = np.mean([landmarks[i][2] for i in [159, 145, 133]])
            right_eye_depth = np.mean([landmarks[i][2] for i in [386, 374, 362]])

            # Jaw definition
            jaw_width_top = np.linalg.norm(landmarks[127] - landmarks[356])
            jaw_width_bottom = np.linalg.norm(landmarks[172] - landmarks[397])

            # Temporal region (temples)
            left_temple = landmarks[21][2]
            right_temple = landmarks[251][2]

            # Face ratio
            face_ratio = face_width / (face_length + 1e-6)

            # Statistical features
            landmarks_flat = landmarks.flatten()
            mean_coord = np.mean(landmarks_flat)
            std_coord = np.std(landmarks_flat)

            features = [
                face_width, face_length, face_ratio,
                left_cheek_depth, right_cheek_depth,
                left_eye_depth, right_eye_depth,
                jaw_width_top, jaw_width_bottom,
                left_temple, right_temple,
                mean_coord, std_coord
            ]
        else:
            features = [0] * 13

        return features

    def extract_body_features(self, image):
        """Extract body pose landmarks using MediaPipe Pose"""
        features = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            landmarks = np.array(landmarks)

            # Body proportions
            # Shoulder width
            shoulder_width = np.linalg.norm(landmarks[11][:2] - landmarks[12][:2])

            # Torso length
            torso_length = np.linalg.norm(landmarks[11][:2] - landmarks[23][:2])

            # Hip width
            hip_width = np.linalg.norm(landmarks[23][:2] - landmarks[24][:2])

            # Arm thickness (shoulder to elbow to wrist)
            left_arm_upper = np.linalg.norm(landmarks[11][:2] - landmarks[13][:2])
            left_arm_lower = np.linalg.norm(landmarks[13][:2] - landmarks[15][:2])
            right_arm_upper = np.linalg.norm(landmarks[12][:2] - landmarks[14][:2])
            right_arm_lower = np.linalg.norm(landmarks[14][:2] - landmarks[16][:2])

            # Leg thickness
            left_leg_upper = np.linalg.norm(landmarks[23][:2] - landmarks[25][:2])
            left_leg_lower = np.linalg.norm(landmarks[25][:2] - landmarks[27][:2])
            right_leg_upper = np.linalg.norm(landmarks[24][:2] - landmarks[26][:2])
            right_leg_lower = np.linalg.norm(landmarks[26][:2] - landmarks[28][:2])

            # Body ratios
            shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
            torso_shoulder_ratio = torso_length / (shoulder_width + 1e-6)

            # Limb ratios (indicator of muscle mass)
            left_arm_ratio = left_arm_upper / (left_arm_lower + 1e-6)
            right_arm_ratio = right_arm_upper / (right_arm_lower + 1e-6)
            left_leg_ratio = left_leg_upper / (left_leg_lower + 1e-6)
            right_leg_ratio = right_leg_upper / (right_leg_lower + 1e-6)

            # Visibility scores (lower visibility might indicate thinner limbs)
            avg_visibility = np.mean(landmarks[:, 3])
            limb_visibility = np.mean([landmarks[i][3] for i in [13, 14, 15, 16, 25, 26, 27, 28]])

            # Statistical features
            coords_flat = landmarks[:, :3].flatten()
            mean_coord = np.mean(coords_flat)
            std_coord = np.std(coords_flat)

            features = [
                shoulder_width, torso_length, hip_width,
                left_arm_upper, left_arm_lower, right_arm_upper, right_arm_lower,
                left_leg_upper, left_leg_lower, right_leg_upper, right_leg_lower,
                shoulder_hip_ratio, torso_shoulder_ratio,
                left_arm_ratio, right_arm_ratio, left_leg_ratio, right_leg_ratio,
                avg_visibility, limb_visibility,
                mean_coord, std_coord
            ]
        else:
            features = [0] * 21

        return features

    def extract_features_from_image(self, image_path):
        """Extract all features from a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None

        face_features = self.extract_face_features(image)
        body_features = self.extract_body_features(image)

        return face_features + body_features

    def load_dataset(self):
        """Load and process entire dataset"""
        data = []
        labels = []

        for split in ['train', 'valid', 'test']:
            for label_name in ['healthy', 'malnurished']:
                folder_path = os.path.join(self.dataset_path, split, label_name)
                label = 0 if label_name == 'healthy' else 1

                if not os.path.exists(folder_path):
                    print(f"Warning: {folder_path} does not exist")
                    continue

                image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

                for img_file in image_files:
                    img_path = os.path.join(folder_path, img_file)
                    features = self.extract_features_from_image(img_path)

                    if features is not None and not all(f == 0 for f in features):
                        data.append(features)
                        labels.append(label)
                        print(f"Processed: {split}/{label_name}/{img_file}")

        return np.array(data), np.array(labels)

    def create_feature_names(self):
        """Create feature names for better interpretability"""
        face_features = [
            'face_width', 'face_length', 'face_ratio',
            'left_cheek_depth', 'right_cheek_depth',
            'left_eye_depth', 'right_eye_depth',
            'jaw_width_top', 'jaw_width_bottom',
            'left_temple', 'right_temple',
            'face_mean_coord', 'face_std_coord'
        ]

        body_features = [
            'shoulder_width', 'torso_length', 'hip_width',
            'left_arm_upper', 'left_arm_lower', 'right_arm_upper', 'right_arm_lower',
            'left_leg_upper', 'left_leg_lower', 'right_leg_upper', 'right_leg_lower',
            'shoulder_hip_ratio', 'torso_shoulder_ratio',
            'left_arm_ratio', 'right_arm_ratio', 'left_leg_ratio', 'right_leg_ratio',
            'avg_visibility', 'limb_visibility',
            'body_mean_coord', 'body_std_coord'
        ]

        return face_features + body_features

    def preprocess_data(self, X_train, X_test, X_val):
        """Preprocess and normalize features"""
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        return X_train_scaled, X_test_scaled, X_val_scaled

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        self.model = xgb.XGBClassifier(
            n_estimators=25,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        )

        eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )

        return self.model

    def evaluate_model(self, X_test, y_test, feature_names):
        """Evaluate model and generate comprehensive metrics"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print("MODEL EVALUATION METRICS")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Malnourished']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy', 'Malnourished'],
                    yticklabels=['Healthy', 'Malnourished'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }

    def plot_correlation_heatmap(self, X, y, feature_names):
        """Plot correlation heatmap of features"""
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y

        # Calculate correlation matrix
        correlation_matrix = df.corr()

        # Plot full correlation heatmap
        plt.figure(figsize=(20, 18))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot correlation with target
        target_corr = correlation_matrix['label'].drop('label').sort_values(ascending=False)

        plt.figure(figsize=(10, 12))
        colors = ['green' if x > 0 else 'red' for x in target_corr.values]
        plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        plt.yticks(range(len(target_corr)), target_corr.index)
        plt.xlabel('Correlation with Malnutrition', fontsize=12)
        plt.title('Feature Correlation with Target Label', fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('target_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot top correlated features heatmap
        top_features = target_corr.abs().nlargest(15).index.tolist()
        top_features.append('label')
        top_corr = df[top_features].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Top 15 Features Correlation with Target', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap_top.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_new_image(self, image_path):
        """Predict malnutrition status for a new image"""
        features = self.extract_features_from_image(image_path)

        if features is None or all(f == 0 for f in features):
            return None, None

        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        result = "Malnourished" if prediction == 1 else "Healthy"
        confidence = probability[prediction] * 100

        return result, confidence

    def save_model(self, model_path='malnutrition_model.pkl'):
        """Save trained model and scaler"""
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"\nModel saved to {model_path}")

    def load_model(self, model_path='malnutrition_model.pkl'):
        """Load trained model and scaler"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"\nModel loaded from {model_path}")
