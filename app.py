import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model artifact not found: {os.path.basename(path)}. "
            "Place the trained model and scaler in the project root."
        )
    with open(path, "rb") as file:
        return pickle.load(file)


model = load_artifact(MODEL_PATH)
scaler = load_artifact(SCALER_PATH)


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
):
    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]
    input_df = pd.DataFrame([values], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    result = {
        "Prediction": (
            "Diabetes predicted by the model"
            if prediction == 1
            else "No diabetes predicted by the model"
        )
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            diabetes_index = classes.index(1)
            result["Model probability for diabetes"] = round(
                float(probabilities[diabetes_index]), 4
            )

    return result


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=1, minimum=0),
        gr.Number(label="Glucose", value=120, minimum=0),
        gr.Number(label="Blood Pressure", value=70, minimum=0),
        gr.Number(label="Skin Thickness", value=20, minimum=0),
        gr.Number(label="Insulin", value=79, minimum=0),
        gr.Number(label="BMI", value=25.0, minimum=0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5, minimum=0),
        gr.Number(label="Age", value=30, minimum=1),
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="Diabetes Prediction System",
    description=(
        "Enter eight clinical features to get a prediction from the trained "
        "XGBoost model. This educational demo is not a medical diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch()
