
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# DIAPREDICT AI 2.0
# Educational Synthetic ML Demonstration
# ==========================================

st.set_page_config(
    page_title="DiaPredict AI",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD NEW MODEL
# ==========================================

@st.cache_resource
def load_model():
    bundle = joblib.load("diapredict_model_v2.joblib")

    model = bundle["model"]
    features = bundle["features"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]

    encoder = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    categories = dict(zip(categorical_features, encoder.categories_))

    return model, features, numeric_features, categorical_features, categories


try:
    model, features, numeric_features, categorical_features, categories = load_model()
except Exception as e:
    st.error("Unable to load the new model.")
    st.code(str(e))
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ==========================================
# LOGIN PAGE
# ==========================================


if not st.session_state.logged_in:

    st.title("🩺 DiaPredict AI")
    st.subheader("Welcome to Smarter Health Awareness")
    st.header("Login to Continue")

    st.write("Enter your details to explore the educational dashboard.")

    with st.form("login_form"):
        username = st.text_input(
            "Email or Username",
            placeholder="Enter your email or username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter a demo password"
        )

        login_button = st.form_submit_button("Login")

    st.caption(
        "Demo login only. This is not a secure authentication system. "
        "Do not enter real passwords or sensitive information."
    )

    if login_button:
        if username.strip() and password.strip():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.warning("Please enter both username and password.")

    st.stop()



# ==========================================
# MAIN DASHBOARD
# ==========================================

st.sidebar.title("🩺 DiaPredict AI")
st.sidebar.write("Educational Health Awareness Platform")

st.sidebar.success(
    "Welcome, " + st.session_state.username
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Home Dashboard",
        "Health Assessment",
        "Diabetes Awareness",
        "Diabetes Across Age Groups",
        "About the Project"
    ]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ==========================================
# HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    st.title("Welcome to DiaPredict AI 2.0")
    st.subheader("An AI-Based Educational Diabetes Awareness Prototype")

    st.write(
        "DiaPredict AI is a student-developed project that demonstrates "
        "how machine learning can process health-related information "
        "and generate an educational classification output."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Records", "1,000")
    c2.metric("Input Variables", "12")
    c3.metric("Model", "Logistic Regression")

    st.divider()

    st.subheader("Explore the Platform")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Health Assessment**\n\nEnter sample measurements and observations.")
        st.info("**Diabetes Awareness**\n\nUnderstand diabetes and its different types.")

    with col2:
        st.info("**Age Group Information**\n\nExplore diabetes-related health considerations across ages.")
        st.info("**About the Project**\n\nLearn about the technology and methodology.")

    st.warning(
        "This project uses entirely synthetic educational data. "
        "Its predictions are not medically validated and must not "
        "be interpreted as an actual diabetes diagnosis or risk estimate."
    )


# ==========================================
# HEALTH ASSESSMENT
# ==========================================

elif page == "Health Assessment":

    st.title("Health Assessment")
    st.write("Enter sample information for the educational AI demonstration.")

    st.info(
        "You can use fictional sample values. Do not enter another person's "
        "private health information."
    )

    with st.form("health_form"):

        st.subheader("1. Personal and Physical Measurements")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=25
            )

            weight = st.number_input(
                "Weight (kg)",
                min_value=10.0,
                max_value=300.0,
                value=60.0,
                step=0.5
            )

        with col2:
            height = st.number_input(
                "Height (cm)",
                min_value=80.0,
                max_value=230.0,
                value=160.0,
                step=0.5
            )

            bmi = weight / ((height / 100) ** 2)

            st.metric("Calculated BMI", f"{bmi:.2f} kg/m²")

        st.caption(
            "BMI is calculated automatically from weight and height."
        )

        st.divider()

        st.subheader("2. Pregnancy Status")

        pregnancy = st.selectbox(
            "Select pregnancy status",
            categories["Pregnancy_Status"].tolist()
        )

        st.caption(
            "Pregnancy is recorded only as a dataset demonstration input. "
            "It does not establish gestational diabetes."
        )

        st.divider()

        st.subheader("3. Symptoms and Observations")

        symptom_fields = [
            ("Dizziness", "Dizziness"),
            ("Numbness_Tingling_Pain", "Numbness / Tingling / Pain"),
            ("Slow_Healing_Wounds_Sores", "Slow-Healing Wounds and Sores"),
            ("Skin_Tags_Acrochordons", "Skin Tags (Acrochordons)"),
            ("Acanthosis_Nigricans", "Acanthosis Nigricans"),
            ("Hand_Mobility_Changes", "Hand Mobility Changes"),
            ("Vision_Shifts", "Vision Shifts")
        ]

        responses = {}

        for feature, label in symptom_fields:

            options = categories[feature].tolist()

            responses[feature] = st.selectbox(
                label,
                options=options,
                key=feature
            )

        submitted = st.form_submit_button(
            "Generate Educational Result",
            use_container_width=True
        )

    if submitted:

        # Prepare all model inputs
        input_values = {
            "Age": age,
            "Weight_kg": weight,
            "Height_cm": height,
            "BMI": bmi,
            "Pregnancy_Status": pregnancy
        }

        input_values.update(responses)

        # Preserve exact training feature order
        row = pd.DataFrame(
            [[input_values[f] for f in features]],
            columns=features
        )

        try:
            prediction = model.predict(row)[0]

            st.divider()
            st.subheader("Educational Model Output")

            st.success("The model successfully processed your sample inputs.")

            st.metric(
                "Synthetic Demonstration Class",
                str(prediction)
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(row)[0]

                if 1 in model.classes_:
                    class_index = list(model.classes_).index(1)

                    st.metric(
                        "Synthetic Class 1 Score",
                        f"{probabilities[class_index] * 100:.2f}%"
                    )

            st.write("**Your submitted sample information:**")
            st.dataframe(row, use_container_width=True)

            st.warning(
                "This output is generated by a model trained on artificial "
                "labels. It is not a measure of actual diabetes probability, "
                "a medical assessment, or a diagnosis. Symptoms and BMI "
                "alone cannot establish whether someone has diabetes."
            )

        except Exception as e:
            st.error("Prediction could not be generated.")
            st.code(str(e))


# ==========================================
# DIABETES AWARENESS
# ==========================================

elif page == "Diabetes Awareness":

    st.title("Understanding Diabetes")

    st.write(
        "Diabetes is a chronic health condition in which the body "
        "cannot properly regulate blood glucose (blood sugar). "
        "Insulin is a hormone produced by the pancreas that helps "
        "glucose enter cells for energy."
    )

    st.subheader("Types of Diabetes")

    with st.expander("Type 1 Diabetes"):
        st.write(
            "Type 1 diabetes is an autoimmune condition in which "
            "the immune system attacks insulin-producing cells "
            "in the pancreas. People with this condition need "
            "insulin treatment."
        )

    with st.expander("Type 2 Diabetes"):
        st.write(
            "Type 2 diabetes occurs when the body does not use "
            "insulin effectively or does not produce enough "
            "insulin. It involves several genetic, biological "
            "and environmental factors."
        )

    with st.expander("Gestational Diabetes"):
        st.write(
            "Gestational diabetes develops during pregnancy. "
            "It requires appropriate medical monitoring and "
            "can affect the health of both the pregnant person "
            "and the baby."
        )

    st.subheader("Common Signs and Symptoms")

    symptoms = [
        "Increased thirst",
        "Frequent urination",
        "Increased hunger",
        "Unexplained fatigue",
        "Blurred vision",
        "Slow-healing wounds",
        "Numbness or tingling in hands and feet"
    ]

    for symptom in symptoms:
        st.write("• " + symptom)

    st.info(
        "Some people with Type 2 diabetes may have no noticeable "
        "symptoms. A blood glucose test and professional medical "
        "evaluation are needed for diagnosis."
    )


# ==========================================
# DIABETES ACROSS AGE GROUPS
# ==========================================

elif page == "Diabetes Across Age Groups":

    st.title("Diabetes Across Different Age Groups")

    st.write(
        "Diabetes can affect people at different stages of life. "
        "Its presentation, associated concerns and healthcare "
        "needs can vary with age."
    )

    with st.expander("Children and Adolescents"):
        st.write(
            "Type 1 diabetes is commonly diagnosed during childhood "
            "or adolescence, although it can occur at any age. "
            "Type 2 diabetes can also develop in young people. "
            "Symptoms may include thirst, frequent urination and fatigue."
        )

    with st.expander("Young Adults"):
        st.write(
            "Diabetes may occur in young adulthood. Type 1 and "
            "Type 2 diabetes have different causes. Family history, "
            "health conditions and other factors can influence "
            "individual health."
        )

    with st.expander("Middle-Aged Adults"):
        st.write(
            "Type 2 diabetes is frequently diagnosed during adulthood. "
            "Routine health checks can help identify elevated blood "
            "glucose levels and support early medical management."
        )

    with st.expander("Older Adults"):
        st.write(
            "Older adults may experience additional health concerns "
            "alongside diabetes, including cardiovascular conditions, "
            "vision problems and changes in kidney function. "
            "Healthcare plans should be individualised."
        )

    st.warning(
        "Age alone cannot determine whether a person has diabetes."
    )


# ==========================================
# ABOUT THE PROJECT
# ==========================================

elif page == "About the Project":

    st.title("About DiaPredict AI")

    st.subheader("Project Objective")

    st.write(
        "The objective of DiaPredict AI is to demonstrate the "
        "application of Artificial Intelligence and Machine Learning "
        "in processing health-related information through a simple "
        "interactive educational platform."
    )

    st.subheader("Technology Used")

    technologies = [
        "Python Programming Language",
        "Google Colab",
        "Pandas and NumPy",
        "Scikit-learn",
        "Logistic Regression",
        "Joblib",
        "Streamlit"
    ]

    for tech in technologies:
        st.write("• " + tech)

    st.subheader("Dataset Information")

    st.write(
        "The model was trained using 1,000 artificially generated "
        "educational records. Its outcome labels were generated "
        "using a synthetic teaching rule rather than verified "
        "clinical diagnoses."
    )

    st.subheader("Project Limitations")

    st.write(
        "The synthetic dataset does not represent real patients. "
        "The model's evaluation results cannot establish real-world "
        "medical accuracy or reliability. The system is intended "
        "only for demonstrating a machine-learning workflow."
    )

    st.success(
        "DiaPredict AI is a student educational prototype, "
        "not a medical diagnostic system."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "© 2026 DiaPredict AI | Student Educational Project | "
    "Synthetic Data Demonstration | Not for Medical Diagnosis"
)

