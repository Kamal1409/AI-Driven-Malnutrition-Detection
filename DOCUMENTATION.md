# Project Documentation

This document provides a detailed overview of the functions and components within the Malnutrition Detection System project.

## Backend

The backend is built using FastAPI and Python, utilizing machine learning models for malnutrition detection.

### `backend/api.py`

This file defines the FastAPI application and its endpoints.

#### `read_root()`
- **Description**: The root endpoint of the API. Returns a simple message to indicate the API is running.
- **Parameters**: None
- **Return Value**: `dict` - `{"message": "Malnutrition Severity Prediction API is running."}`

#### `predict_image(file: UploadFile)`
- **Description**: Endpoint to upload an image and get a malnutrition prediction. It saves the image temporarily, runs the prediction model, and returns the result.
- **Parameters**:
    - `file` (`UploadFile`): The image file to be analyzed.
- **Return Value**: `dict` - Contains `filename`, `severity_score`, `severity_level`, and `face_detected` status.

#### `calculate_severity(data: SeverityInput)`
- **Description**: Endpoint to calculate malnutrition severity based on numerical inputs (age, height, weight, sex).
- **Parameters**:
    - `data` (`SeverityInput`): A Pydantic model containing `age` (int), `height` (float), `weight` (float), and `sex` (int).
- **Return Value**: `dict` - Contains the predicted `severity` label.

### `backend/malnutrition_predictor.py`

This file contains the `MalnutritionPredictor` class, which handles feature extraction, model training, and prediction.

#### `MalnutritionPredictor` Class

##### `__init__(self, dataset_path='dataset')`
- **Description**: Initializes the `MalnutritionPredictor` instance, sets up MediaPipe Face Mesh and Pose solutions, and initializes the scaler.
- **Parameters**:
    - `dataset_path` (`str`): Path to the dataset directory. Default is `'dataset'`.

##### `extract_face_features(self, image)`
- **Description**: Extracts facial landmarks and calculates features like face width, length, cheek depth, etc., using MediaPipe Face Mesh.
- **Parameters**:
    - `image` (`numpy.ndarray`): The image to process.
- **Return Value**: `list` - A list of extracted facial features.

##### `extract_body_features(self, image)`
- **Description**: Extracts body pose landmarks and calculates features like shoulder width, limb ratios, etc., using MediaPipe Pose.
- **Parameters**:
    - `image` (`numpy.ndarray`): The image to process.
- **Return Value**: `list` - A list of extracted body features.

##### `extract_features_from_image(self, image_path)`
- **Description**: Loads an image from the given path and extracts both face and body features.
- **Parameters**:
    - `image_path` (`str`): Path to the image file.
- **Return Value**: `list` - Combined list of face and body features, or `None` if image load fails.

##### `load_dataset(self)`
- **Description**: Iterates through the dataset directory structure, loads images, and extracts features for training/testing.
- **Parameters**: None
- **Return Value**: `tuple` - `(data, labels)` where `data` is a numpy array of features and `labels` is a numpy array of target labels.

##### `create_feature_names(self)`
- **Description**: Generates a list of human-readable names for the extracted features.
- **Parameters**: None
- **Return Value**: `list` - List of feature names strings.

##### `preprocess_data(self, X_train, X_test, X_val)`
- **Description**: Scales the feature data using `StandardScaler`. Fits on training data and transforms test/validation data.
- **Parameters**:
    - `X_train`, `X_test`, `X_val` (`numpy.ndarray`): Feature sets.
- **Return Value**: `tuple` - Scaled versions of the input arrays.

##### `train_model(self, X_train, y_train, X_val, y_val)`
- **Description**: Configures and trains an XGBoost classifier.
- **Parameters**:
    - `X_train`, `y_train`: Training data and labels.
    - `X_val`, `y_val`: Validation data and labels.
- **Return Value**: `xgb.XGBClassifier` - The trained model.

##### `evaluate_model(self, X_test, y_test, feature_names)`
- **Description**: Evaluates the trained model on test data, calculates metrics (Accuracy, Precision, Recall, F1, AUC), and generates plots (Confusion Matrix, ROC Curve, Feature Importance).
- **Parameters**:
    - `X_test`, `y_test`: Test data and labels.
    - `feature_names` (`list`): List of feature names for importance plotting.
- **Return Value**: `dict` - Dictionary of evaluation metrics.

##### `plot_correlation_heatmap(self, X, y, feature_names)`
- **Description**: Generates and saves correlation heatmaps for the features and target variable.
- **Parameters**:
    - `X`, `y`: Data and labels.
    - `feature_names` (`list`): Names of the features.
- **Return Value**: None

##### `predict_new_image(self, image_path)`
- **Description**: Extracts features from a new image, scales them, and predicts the malnutrition status using the trained model.
- **Parameters**:
    - `image_path` (`str`): Path to the new image.
- **Return Value**: `tuple` - `(result, confidence)` where `result` is "Healthy" or "Malnourished" and `confidence` is the probability percentage.

##### `save_model(self, model_path='malnutrition_model.pkl')`
- **Description**: Saves the trained model and scaler to a pickle file.
- **Parameters**:
    - `model_path` (`str`): Destination path.
- **Return Value**: None

##### `load_model(self, model_path='malnutrition_model.pkl')`
- **Description**: Loads a trained model and scaler from a pickle file.
- **Parameters**:
    - `model_path` (`str`): Source path.
- **Return Value**: None

### `backend/main.py`
This script is used for local testing of the `MalnutritionPredictor`. It initializes the predictor, loads the model, and runs a prediction on a test image.

### `backend/utils/inference.py`
Currently empty.

---

## Frontend

The frontend is a React application using Vite.

### `frontend/src/App.jsx`

The main component of the application.

#### `App()`
- **Description**: The root functional component that manages the application state and UI.
- **State Variables**:
    - `file`: Stores the selected image file.
    - `preview`: URL for the image preview.
    - `loading`: Boolean for analysis loading state.
    - `result`: Stores the analysis result from the backend.
    - `isCameraCapture`: Boolean indicating if the image came from the camera.
    - `severityForm`: Object storing inputs for severity calculation (age, height, weight, sex).
    - `severityResult`: Stores the result of the severity calculation.
    - `loadingSeverity`: Boolean for severity calculation loading state.

#### `handleFileInputChange(e)`
- **Description**: Event handler for the file input element.
- **Parameters**:
    - `e`: The event object.

#### `handleSelectedFile(f, fromCamera = false)`
- **Description**: Validates and sets the selected file and its preview.
- **Parameters**:
    - `f`: The file object.
    - `fromCamera` (`bool`): Flag indicating if the file is from the camera.

#### `handleDrop(e)`
- **Description**: Event handler for drag-and-drop functionality.
- **Parameters**:
    - `e`: The event object.

#### `takePhoto()`
- **Description**: Opens the webcam, captures a photo, and sets it as the selected file. Uses a dynamically created modal for the video stream.

#### `analyze()`
- **Description**: Sends the selected image to the backend `/predict` endpoint and updates the `result` state.

#### `calculateSeverity()`
- **Description**: Sends the form data (age, height, weight, sex) to the backend `/severity` endpoint and updates the `severityResult` state.

#### `severityStyle(sev)`
- **Description**: Helper function to determine the CSS styles (color, background, border) for the result card based on the severity level string.
- **Parameters**:
    - `sev` (`str`): The severity level string.
- **Return Value**: `object` - A style object.

### `frontend/src/main.jsx`
The entry point of the React application. It renders the `App` component into the DOM.
