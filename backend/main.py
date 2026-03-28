import cv2
from malnutrition_predictor import MalnutritionPredictor
import joblib
import numpy as np
# Initialize the predictor
predictor = MalnutritionPredictor()

# Load trained model and scaler
predictor.load_model(r"backend\models\malnutrition_model.pkl")

# Path to new image
image_path = r"backend\uploads\tokyo.jpg" # change this to your test image path

# Run prediction
result, confidence = predictor.predict_new_image(image_path)

# Display result
if result is None:
    print("Could not extract landmarks — ensure face and body are clearly visible in the image.")
elif result == "Healthy":
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
elif result == "Malnourished":

    # Load the trained model
    severity_model = joblib.load('backend\models\malnutrition_severity_model.pkl')
    print("The person is detected as Malnourished. Let's predict severity.")
    age = int(input("Enter age: "))
    height = float(input("Enter height (in cm): "))
    weight = float(input("Enter weight (in kg): "))
    sex = int(input("Enter sex (1 for male, 0 for female): "))

    # Prepare input for model
    X_input = np.array([[sex, age, height, weight]])

    # Predict severity (directly gives class name)
    severity_label = severity_model.predict(X_input)[0]

    print(f"Predicted malnutrition type: {severity_label}")