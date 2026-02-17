import streamlit as st
import numpy as np
import joblib
import base64

# Load trained model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

#style with css

st.set_page_config(layout="wide")

# Function to convert image
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load image
img = get_base64("diabitics_bg1.png")

# Apply background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Diabetes Prediction App")
st.write("Predict whether a person is diabetic based on medical inputs.")

# User inputs
pregnancies_input = st.radio("Pregnancies",["Yes","No"])
if pregnancies_input=="yes":
    pregnancies = 1
else:
    pregnancies = 0    
#pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Not Diabetic (Probability: {probability:.2f})")

