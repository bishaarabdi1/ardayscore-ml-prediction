
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load the saved training pipeline
MODEL_PATH = Path(__file__).with_name("model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Student Performance Prediction API")

# Keep the API usable from a separately served frontend during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Validate incoming request data
class StudentInput(BaseModel):
    studyhour: float = Field(ge=0)
    attendance: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {
        "message": "Student Prediction API is running",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Student Performance Prediction API"}


@app.post("/predict")
def predict_student(data: StudentInput):
    # Convert validated input to a DataFrame
    input_data = pd.DataFrame(
        [{
            "studyhour": data.studyhour,
            "attendance": data.attendance
        }]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Find the probability for the predicted class
    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)])

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 4)
    }
