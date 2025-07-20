# Heart Disease Prediction using Ensemble Learning

This project implements an end-to-end machine learning pipeline to predict heart disease risk using clinical data. It includes data preprocessing, ensemble model training, an inference API built with FastAPI, and a minimal web interface for user interaction.

---

## 🔹 Objective

Develop a robust and interpretable heart disease prediction system using machine learning. The system should support real-time predictions via a REST API and be accessible through a simple frontend UI.

---

## 📊 Machine Learning Pipeline

### ➤ Dataset

A structured dataset containing clinical and demographic features:

* **Target**: Presence of heart disease (0: No, 1: Yes)

### ➤ Features Used

* `Age`
* `Sex` (1: Male, 0: Female)
* `ChestPainType_ASY`
* `RestingBP`
* `Cholesterol`
* `FastingBS`
* `MaxHR`
* `ExerciseAngina`
* `Oldpeak`
* `ST_Slope_Up`

### ➤ Preprocessing

* Categorical encoding (LabelEncoding and One-Hot for dummy variables)
* Scaling with `StandardScaler` (for: `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`)
* Train/test split (80/20), 5-fold cross-validation

### ➤ Model

* **Model Type**: `VotingClassifier`
* **Base Learners**:

  * `RandomForestClassifier`
  * `KNeighborsClassifier`
* **Voting Strategy**: Hard voting (majority class)

### ➤ Exported Artifacts

* `hard_vote_model.pkl` – Trained ensemble model

---

## 🧪 Evaluation

* Accuracy: \~90%
* F1-Score, Precision, Recall computed
* 5-fold Cross-Validation
* Confusion matrix and ROC AUC used for performance validation

---

## 🧩 Backend: FastAPI

The model is served using a FastAPI application with the following endpoint:

### ➤ `/predict` (POST)

**Request Body:**

```json
{
  "Age": 52,
  "Sex": 1,
  "ChestPainType_ASY": 1,
  "RestingBP": 140,
  "Cholesterol": 260,
  "FastingBS": 1,
  "MaxHR": 160,
  "ExerciseAngina": 1,
  "Oldpeak": 2.3,
  "ST_Slope_Up": 1
}
```

**Response:**

```json
{
  "prediction": "High Risk"  // or "Low Risk"
}
```

**CORS** is enabled for frontend interaction.

---

## 🌐 Frontend: Minimal UI (HTML + JS)

A single-page UI is built to collect user inputs and fetch predictions via the FastAPI backend.

### ➤ Features

* Input form for all model-required features
* `Fetch API` for POST request to `http://127.0.0.1:8000/predict`
* Displays prediction result on-screen

**Note**: Must be served on a CORS-allowed domain (e.g., `localhost`, `127.0.0.1`).

---

## 🗂 Project Structure

```
📦 heart-disease-prediction/
 ┣ 📁 backend/
 ┃ ┣ 📜 main.py                 # FastAPI app with /predict route
 ┃ ┣ 📜 hard_vote_model.pkl     # Trained ensemble model
 ┃ ┗ 📜 scaler.pkl              # Trained StandardScaler
 ┣ 📁 frontend/
 ┃ ┗ 📜 index.html              # Simple HTML UI for input and prediction
 ┣ 📁 training/
 ┃ ┣ 📜 train_model.py          # Data processing and model training
 ┃ ┗ 📜 utils.py                # Helper functions for encoding/scaling
 ┗ 📜 README.md
```

---

## 🔧 Installation & Run Instructions

### ➤ Install dependencies

```bash
poetry install
# or
pip install -r requirements.txt
```

### ➤ Run API Server

```bash
uvicorn main:app --reload
```

### ➤ Open Frontend

Open `frontend/index.html` in a browser (ensure FastAPI server is running).

---

