# AI-Driven Malnutrition Detection  
### Facial & Body Landmark Based Nutritional Assessment System

An AI-powered, non-invasive malnutrition screening system that uses **facial geometry, body pose landmarks, and engineered anthropometric features** to detect and classify malnutrition severity with high accuracy.

> Achieves **95.52% accuracy** using a lightweight, interpretable XGBoost-based framework.

---

##  Overview

Early detection of malnutrition is critical for preventing long-term health complications, especially in children and low-resource environments.

Traditional screening methods such as:
- BMI
- MUAC
- Height-for-age
- Weight-for-age  

require physical contact, trained personnel, and specialized equipment.

This project proposes a **computer vision-based alternative** using:

- MediaPipe Face Mesh (468 landmarks)
- MediaPipe Pose (33 body landmarks)
- Engineered anthropometric & texture features
- XGBoost & Random Forest classifiers
- Pseudo-labeling for low-data training

---

##  System Architecture

The system follows a **three-layer modular architecture**:

### 1️ Presentation Layer
- Web-based interface
- Upload image or capture via camera
- Displays health status & severity level

### 2️ Application Layer
- Built with **FastAPI**
- REST API endpoints
- Image validation & routing
- Returns structured JSON responses

### 3️ Intelligence Layer
- MediaPipe landmark extraction
- Feature engineering (34 features)
- StandardScaler normalization
- Binary XGBoost classifier
- Severity classification model

---

## Feature Engineering

Instead of deep CNN black-box models, this system uses **clinically motivated engineered features**.

### Geometric Features
- Facial width-to-height ratio
- Jaw width & definition
- Cheek depth & hollowness
- Inter-ocular distance
- Shoulder-to-hip ratio
- Limb-to-torso ratio

### Texture Features
- HSV color statistics
- Laplacian variance (image sharpness)
- Image entropy
- Local Binary Pattern (LBP) histograms
- Gray Level Co-occurrence Matrix (GLCM) statistics

All features are normalized using **inter-ocular distance** to ensure scale invariance.

---

## Learning Pipeline

### Stage 1: Binary Classification
**Model:** XGBoost  
**Output:** Healthy vs Malnourished  

### Stage 2: Severity Classification
**Model:** Random Forest / Multi-class XGBoost  
**Classes:**
- Mild
- Moderate
- Severe

---

## Pseudo-Labeling Strategy

Due to limited annotated datasets:

1. UMAP dimensionality reduction  
2. K-Means / Gaussian Mixture clustering  
3. Cluster evaluation using:
   - Silhouette Score
   - Calinski–Harabasz Index
   - Davies–Bouldin Index  
4. Semantic mapping to nutritional categories  

---

## Performance

### Overall Accuracy: **95.52%**  
**Macro F1-Score:** 0.9576  

### Model Comparison

| Model                          | Accuracy |
|--------------------------------|----------|
| Custom XGBoost                 | **0.9552** |
| LightGBM                       | 0.9279 |
| Random Forest + MobileNetV2    | 0.87 |
| EfficientNet-B0                | 0.527 |

### Key Insight
Feature-driven ensemble models outperform deep CNNs in **low-data regimes** while maintaining interpretability.

---

## Hyperparameters (XGBoost)

- Estimators: 200–300  
- Max Depth: 6  
- Learning Rate: 0.05–0.1  
- Subsample: 0.8  
- Colsample By Tree: 0.8  
- Early Stopping Enabled  

---

## Tech Stack

- Python
- FastAPI
- MediaPipe
- XGBoost
- Random Forest
- Scikit-learn
- UMAP
- OpenCV
- NumPy
- Pandas

---

## Installation

```bash
git clone https://github.com/yourusername/ai-malnutrition-detection.git
cd ai-malnutrition-detection

pip install -r requirements.txt
```
## Run Backend

```bash
uvicorn app:app --reload
```

---

## Project Structure

```bash
├── app.py
├── models/
│   ├── xgboost_model.pkl
│   ├── severity_model.pkl
│   └── scaler.pkl
├── feature_extraction/
│   ├── landmarks.py
│   ├── geometric_features.py
│   └── texture_features.py
├── utils/
├── frontend/
├── uploads/
└── README.md
```

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- AUC  

---

## Applications

- Rural healthcare screening  
- Low-resource medical environments  
- Pediatric nutritional monitoring  
- NGO-based community screening  
- Edge device deployment  

---

## Future Work

- Validation using clinically annotated datasets  
- Micronutrient deficiency detection  
- Edge device optimization  
- Longitudinal nutritional recovery tracking  
- Bias and fairness auditing  

---

## Advantages Over CNN-Based Systems

- Lightweight  
- Interpretable  
- Works with limited labeled data  
- Lower computational cost  
- Suitable for real-time deployment  

---

## Citation

If you use this work, please cite:

**AI-Driven Malnutrition Detection Using Facial and Body Landmark Analysis**

---

## Authors

- Herbert George  
- Rachuri Kamal Adithya  
- Vineeta Pareek  
- Gowtam B   
