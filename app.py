
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Student Withdrawal Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.joblib")

@st.cache_data
def load_resources():
    with open(BASE_DIR / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    with open(BASE_DIR / "category_options.json", encoding="utf-8") as f:
        options = json.load(f)

    with open(BASE_DIR / "region_mapping.json", encoding="utf-8") as f:
        region_mapping = json.load(f)

    unemployment = pd.read_csv(
        BASE_DIR / "unemployment_lookup.csv"
    )

    return config, options, region_mapping, unemployment

model = load_model()
config, options, region_mapping, unemployment = load_resources()

st.title("Student Academic Performance & Withdrawal Risk Prediction")
st.write(
    "An educational machine learning application that estimates "
    "the probability of course withdrawal after Day 30."
)

st.info(
    "This model predicts course withdrawal risk, not permanent "
    "university dropout. It is intended for educational demonstration "
    "and should not be used for automatic academic decisions."
)

st.subheader("Student Information")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        code_module = st.selectbox(
            "Course Module",
            options["code_module"]
        )

        code_presentation = st.selectbox(
            "Course Presentation",
            options["code_presentation"]
        )

        gender = st.selectbox(
            "Gender",
            options["gender"]
        )

        region = st.selectbox(
            "Region",
            options["region"]
        )

        highest_education = st.selectbox(
            "Highest Education",
            options["highest_education"]
        )

        imd_band = st.selectbox(
            "IMD Band",
            options["imd_band"]
        )

        age_band = st.selectbox(
            "Age Band",
            options["age_band"]
        )

        disability = st.selectbox(
            "Disability",
            options["disability"]
        )

    with col2:
        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

        studied_credits = st.number_input(
            "Studied Credits",
            min_value=0,
            max_value=630,
            value=60,
            step=10
        )

        date_registration = st.number_input(
            "Registration Day (relative to course start)",
            min_value=-400,
            max_value=30,
            value=-30,
            step=1
        )

        st.markdown("**First 30 Days: Assessment Activity**")

        early_assessment_count = st.number_input(
            "Assessments Submitted",
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )

        early_total_weight = st.number_input(
            "Total Assessment Weight",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0
        )

        early_avg_submission_day = st.number_input(
            "Average Submission Day",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Withdrawal Risk",
        use_container_width=True
    )

if submitted:

    ons_region = region_mapping.get(region)

    unemployment_rate = np.nan

    if ons_region is not None:

        matched = unemployment[
            (unemployment["code_presentation"] == code_presentation) &
            (unemployment["ons_region"] == ons_region)
        ]

        if not matched.empty:
            unemployment_rate = float(
                matched.iloc[0]["unemployment_rate"]
            )

    # Match training data: no early assessment means missing
    # assessment-derived features, not numeric zero.
    if early_assessment_count == 0:
        early_assessment_count = np.nan
        early_total_weight = np.nan
        early_avg_submission_day = np.nan

    input_data = pd.DataFrame([{
        "code_module": code_module,
        "code_presentation": code_presentation,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "date_registration": date_registration,
        "early_assessment_count": early_assessment_count,
        "early_total_weight": early_total_weight,
        "early_avg_submission_day": early_avg_submission_day,
        "unemployment_rate": unemployment_rate
    }])

    input_data = input_data[config["selected_features"]]

    risk_probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    threshold = float(config["threshold"])
    at_risk = risk_probability >= threshold

    st.divider()
    st.subheader("Prediction Result")

    left, right = st.columns(2)

    with left:
        st.metric(
            "Estimated Withdrawal Probability",
            f"{risk_probability * 100:.2f}%"
        )

    with right:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    if at_risk:
        st.warning(
            "⚠️ Higher predicted course withdrawal risk "
            "based on the selected threshold."
        )
    else:
        st.success(
            "Lower predicted course withdrawal risk "
            "based on the selected threshold."
        )

    st.subheader("Prediction Probability Chart")

    chart_data = pd.DataFrame({
        "Outcome": ["Not Withdrawn", "Withdrawn"],
        "Probability": [
            1 - risk_probability,
            risk_probability
        ]
    }).set_index("Outcome")

    st.bar_chart(chart_data)

    st.subheader("External Data Information")

    if pd.isna(unemployment_rate):
        st.write(
            "No reliable historical unemployment rate mapping "
            "was available for the selected region. The model "
            "used its training-data imputation procedure."
        )
    else:
        st.write(
            f"Historical regional unemployment rate: "
            f"{unemployment_rate:.2f}%"
        )

    st.caption(
        "Predictions are statistical estimates, not confirmed outcomes. "
        "The model was evaluated on historical OULAD records and may "
        "not generalize to other universities or student populations."
    )

st.divider()
st.caption(
    "Machine Learning CCP Project | "
    "Student Academic Performance & Dropout Risk Prediction"
)
