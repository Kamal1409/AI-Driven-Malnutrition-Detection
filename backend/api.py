from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from malnutrition_predictor import MalnutritionPredictor
import joblib
import numpy as np

# ---------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Malnutrition Severity Prediction API")

# Allow CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
predictor = MalnutritionPredictor()
predictor.load_model('models/malnutrition_model.pkl')

severity_model = joblib.load('models/malnutrition_severity_model.pkl')

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Malnutrition Severity Prediction API is running."}

# ---------------------------------------------------------
# 1️⃣ Image upload & prediction endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, saves it temporarily,
    runs malnutrition detection, and returns results.
    """
    try:
        # Save uploaded image temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        result, confidence = predictor.predict_new_image(str(temp_path))

        # Clean up
        os.remove(temp_path)

        # Build response
        if result is None:
            return {
                "filename": file.filename,
                "severity_score": None,
                "severity_level": None,
                "face_detected": False
            }

        return {
            "filename": file.filename,
            "severity_score": confidence,
            "severity_level": result,
            "face_detected": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 2️⃣ Severity calculation endpoint (numerical form)
# ---------------------------------------------------------
class SeverityInput(BaseModel):
    age: int
    height: float
    weight: float
    sex: int  # 1 = male, 0 = female

@app.post("/severity")
async def calculate_severity(data: SeverityInput):
    """
    Accepts age, height, weight, and sex to predict malnutrition severity level.
    """
    try:
        X_input = np.array([[data.sex, data.age, data.height, data.weight]])
        print(f"Severity model input: {X_input}")

        severity_label = severity_model.predict(X_input)[0]

        return {"severity": severity_label}

    except Exception as e:
        print(f"Severity model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# # ---------------------------------------------------------
# # 3️⃣ Camera photo endpoint (simplified)
# # ---------------------------------------------------------
# @app.post("/image")
# async def predict_image_photo(file: UploadFile = File(...)):
#     """
#     Handles photo capture from the frontend camera (used in 'Take Photo' mode).
#     """
#     try:
#         temp_path = UPLOAD_DIR / file.filename
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         result, confidence = predictor.predict_new_image(str(temp_path))
#         os.remove(temp_path)

#         if result is None:
#             return {
#                 "filename": file.filename,
#                 "severity_score": None,
#                 "severity_level": None,
#                 "face_detected": False
#             }

#         return {
#             "filename": file.filename,
#             "severity_score": confidence,
#             "severity_level": result,
#             "face_detected": True,
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
