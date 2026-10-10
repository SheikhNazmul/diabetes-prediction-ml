import spaces
import os
import pickle
import gradio as gr
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "final_xgb_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
@spaces.GPU(duration=1)
def predict_diabetes(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age
):
    values = [
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]

    input_df = pd.DataFrame([values], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    result = (
        "Diabetes predicted"
        if prediction == 1
        else "No diabetes predicted"
    )

    response = {"Prediction": result}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        response["Model probability for diabetes"] = round(
            float(probabilities[1]), 4
        )

    return response

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
        "Enter the eight clinical features to get a prediction "
        "from the trained XGBoost model. This demo is not a medical diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch()