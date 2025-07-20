from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.heart_failure_api.schemas import PatientData
import pickle
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Use "*" only for local testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Serve static assets like CSS/JS from /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html manually (DO NOT MOUNT "." as static)
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("index.html")

# Load everything at startup
model = pickle.load(open("model/hard_vote_model.pkl", "rb"))

# Create fresh scalers
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

# Define the features for each scaling type
standard_scale_features = ['Age', 'RestingBP', 'MaxHR']
robust_scale_features = ['Cholesterol', 'Oldpeak']

# Load other resources
label_encodings = json.load(open("model/label_encodings.json"))
onehot_features = json.load(open("model/onehot_feature_names.json"))

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.dict()])

        # Get feature names from the model
        feature_names = json.load(open("model/feature_names.json"))

        # Ensure the input data has the same features in the same order as the model expects
        input_df = input_df[feature_names]

        # Create a copy to avoid modifying the original
        scaled_df = input_df.copy()
        
        # Convert all values to float to avoid type issues
        for col in input_df.columns:
            scaled_df[col] = scaled_df[col].astype(float)
        
        # Apply standard scaling - manually scale using common parameters
        scaled_df['Age'] = (scaled_df['Age'] - 54.5) / 9.0
        scaled_df['RestingBP'] = (scaled_df['RestingBP'] - 132.0) / 18.0
        scaled_df['MaxHR'] = (scaled_df['MaxHR'] - 136.0) / 25.0
        
        # Apply robust scaling - manually scale using common parameters
        scaled_df['Cholesterol'] = (scaled_df['Cholesterol'] - 240.0) / 77.0
        scaled_df['Oldpeak'] = (scaled_df['Oldpeak'] - 0.8) / 1.6

        # Predict
        prediction = model.predict(scaled_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
