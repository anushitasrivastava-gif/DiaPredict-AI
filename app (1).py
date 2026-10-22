import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="DiaPredict AI",
    page_icon="🩺",
    layout="centered"
)

saved = joblib.load("/content/diapredict_model.joblib")
model = saved["model"]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff5f8, #f2f5ff);
}
h1 { color: #b34b78; }
div.stButton > button {
    background-color: #b34b78;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 DiaPredict AI")
st.subheader("An AI-Based Diabetes Risk Screening Prototype")

st.write(
    "Enter sample health measurements to demonstrate "
    "how a machine-learning model generates a prediction."
)

st.info(
    "Educational project only. This tool cannot diagnose diabetes "
    "or replace professional medical advice."
)

with st.form("health_form"):
    st.markdown("### Health Measurements")

    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose", min_value=0, max_value=300, value=120
    )
    bp = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=70
    )
    skin = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=20
    )
    insulin = st.number_input(
        "Insulin", min_value=0, max_value=1000, value=80
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=100.0,
        value=25.0, step=0.1
    )
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0,
        value=0.5, step=0.01
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30
    )

    submitted = st.form_submit_button("Check Model Prediction")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    probability = model.predict_proba(row)[0][1]

    st.markdown("### Model Result")
    st.metric("Estimated Model Probability", f"{probability:.1%}")
    st.caption(
        "This is a model-generated estimate from a public dataset, "
        "not a person's confirmed medical risk."
    )
    st.warning(
        "This result is not a diagnosis. Consult a qualified "
        "healthcare professional for medical assessment."
    )

st.divider()
st.caption(
    "DiaPredict AI | Student Project | Educational Demonstration"
)
