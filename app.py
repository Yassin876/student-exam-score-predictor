"""Local Gradio app that predicts exam scores with the saved Random Forest model."""

import warnings
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_random_forest_model.joblib")
encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
scaler = joblib.load(MODEL_DIR / "min_max_scaler.joblib")

FEATURE_ORDER = [
    "Gender",
    "Parental Education Level",
    "Lunch Type",
    "Test Preparation Course",
    "Study Time",
    "Absences",
]
CATEGORICAL = [
    "Gender",
    "Parental Education Level",
    "Lunch Type",
    "Test Preparation Course",
]
NUMERIC = ["Study Time", "Absences"]


def predict_exam_score(
    gender,
    parental_education,
    lunch_type,
    test_preparation,
    study_time,
    absences,
):
    """Encode inputs the same way as training, then return subject scores and their average."""
    raw = {
        "Gender": gender,
        "Parental Education Level": parental_education,
        "Lunch Type": lunch_type,
        "Test Preparation Course": test_preparation,
        "Study Time": float(study_time),
        "Absences": float(absences),
    }

    encoded = {
        column: int(encoders[column].transform([raw[column]])[0]) for column in CATEGORICAL
    }
    scaled = scaler.transform(pd.DataFrame([[raw["Study Time"], raw["Absences"]]], columns=NUMERIC))
    encoded["Study Time"] = float(scaled[0, 0])
    encoded["Absences"] = float(scaled[0, 1])

    features = pd.DataFrame([encoded], columns=FEATURE_ORDER)
    math_score, reading_score, writing_score = model.predict(features)[0]
    scores = [float(np.clip(value, 0, 100)) for value in (math_score, reading_score, writing_score)]
    average = sum(scores) / 3

    summary = (
        f"Predicted exam score: {average:.1f} / 100\n"
        f"Math {scores[0]:.1f}  |  Reading {scores[1]:.1f}  |  Writing {scores[2]:.1f}\n\n"
        f"Inputs used: {gender}, parental education {parental_education}, "
        f"{lunch_type} lunch, test preparation {test_preparation.lower()}, "
        f"study time {float(study_time):.1f} hours, absences {float(absences):.0f}."
    )
    return round(scores[0], 1), round(scores[1], 1), round(scores[2], 1), round(average, 1), summary


demo = gr.Interface(
    fn=predict_exam_score,
    inputs=[
        gr.Dropdown(choices=["Female", "Male"], label="Gender", value="Female"),
        gr.Dropdown(
            choices=["High School", "College", "Bachelor", "Master"],
            label="Parental Education Level",
            value="Bachelor",
        ),
        gr.Dropdown(choices=["Standard", "Free/Reduced"], label="Lunch Type", value="Standard"),
        gr.Dropdown(
            choices=["Completed", "Not completed"],
            label="Test Preparation Course",
            value="Completed",
        ),
        gr.Slider(minimum=2, maximum=18, step=0.1, value=10, label="Study Time (hours)"),
        gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Absences"),
    ],
    outputs=[
        gr.Number(label="Predicted Math Score"),
        gr.Number(label="Predicted Reading Score"),
        gr.Number(label="Predicted Writing Score"),
        gr.Number(label="Predicted Exam Score (average)"),
        gr.Textbox(label="Result summary", lines=4),
    ],
    title="Student Exam Score Predictor",
    description=(
        "Enter student features. The saved Random Forest model predicts Math, Reading, "
        "and Writing scores. The exam score is their average."
    ),
    examples=[
        ["Female", "Master", "Standard", "Completed", 16, 1],
        ["Male", "Bachelor", "Standard", "Completed", 12, 3],
        ["Male", "High School", "Free/Reduced", "Not completed", 4, 9],
        ["Female", "College", "Free/Reduced", "Not completed", 8, 6],
    ],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
