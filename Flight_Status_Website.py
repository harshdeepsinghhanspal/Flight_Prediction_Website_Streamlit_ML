import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('flight_status_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('flight_status_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Flight Status Predictor ✈️")

# Sliders for input features
wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 20)
temperature = st.slider("Temperature (°C)", -10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)

# Prepare input data
new_data = np.array([[wind_speed, temperature, humidity]])

# Apply scaling using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# Debugging: Print values
st.write("Raw Input:", new_data)
st.write("Scaled Input:", new_data_scaled)

# Make prediction
prediction = model.predict(new_data_scaled)

# Debugging: Print prediction output
st.write("Prediction Output:", prediction)

# Display result
if prediction[0] == 1:
    st.success("✅ Flight Status: Fly")
else:
    st.error("❌ Flight Status: Don't Fly")