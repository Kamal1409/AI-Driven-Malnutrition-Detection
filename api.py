from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from main import MediaPipeFeatureExtractor

# ---------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Malnutrition Severity Prediction API")

# Allow CORS (so your React/Angular/Vue frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model once at startup
model = MediaPipeFeatureExtractor("model.json")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
#  Root endpoint
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Malnutrition Severity Prediction API is running."}

# ---------------------------------------------------------
#  Image upload & prediction endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accepts an image file, saves it temporarily, runs prediction, and returns results."""
    try:
        # Save uploaded image temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        result = model.extract_from_image(str(temp_path))

        # Clean up (optional)
        os.remove(temp_path)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "filename": result.get("filename"),
            "severity_score": result.get("severity_score"),
            "severity_level": result.get("severity_level"),
            "face_detected": result.get("face_detected", True),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
