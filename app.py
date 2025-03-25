import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("model_rfcModel.pkl")
# scaler = joblib.load("scaler.pkl")

# Set Streamlit page layout
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

# Title
st.markdown("<h2 style='text-align: center;'>Machine Failure Prediction</h2>", unsafe_allow_html=True)

# Main input section (centered)
with st.form("prediction_form"):
    st.subheader("Enter Machine Parameters")

    # Type selection (first column)
    

    col1, col2 = st.columns(2)  # Creating two columns for layout

    with col1:
        temperature = st.slider(":rainbow[Temperature]", 35.55, 121.94)
        vibration = st.slider(":rainbow[Vibration]", -17.9, 113.8)
        humidity = st.slider(":rainbow[Humidity]", 30.0, 80.0)
    
    with col2:
        pressure = st.slider(":rainbow[Pressure]", 1,5)
        energy_consumption = st.slider(":rainbow[Energy Consumption]", 0.5, 5.0)
        machine_status = st.radio(":rainbow[Machine Status]", (0,1,2), horizontal= True)

    # Predict button
    predict_button = st.form_submit_button("Predict Failure")

# Prediction logic
if predict_button:
    input_data = np.array([[temperature, vibration, humidity, pressure, energy_consumption, machine_status]])

    # Apply MinMaxScaler transformation
    # input_data_scaled = scaler.transform(input_data)

    # Predict using the ML model
    prediction = model.predict(input_data)

    # Display result
    failure_types = {
        0: "No Failure",
        1: "Heat Dissipation Failure",
        2: "Power Failure",
        3: "Overstrain Failure",
        4: "Tool Wear Failure",
        5: "Random Failure"
    }

    result = failure_types.get(prediction[0], "Unknown Failure")

    if prediction[0] == 0:
        st.success(f"No Failure Predicted ✅")
    else:
        st.error(f"Machine will Fail.⚠️")