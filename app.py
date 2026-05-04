import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('xgboost_model.pkl')

# 2. Build the User Interface
st.title("Predictive Maintenance Dashboard")
st.write("Enter the machine's real-time sensor data to predict the likelihood of failure.")

# Create two columns for a clean layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Readings")
    air_temp = st.number_input("Air Temperature (K)", min_value=250.0, max_value=350.0, value=300.0)
    process_temp = st.number_input("Process Temperature (K)", min_value=250.0, max_value=350.0, value=310.0)
    rpm = st.number_input("Rotational Speed (RPM)", min_value=0, max_value=3000, value=1500)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0)

with col2:
    st.subheader("Machine Details")
    tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=300, value=100)
    machine_type = st.selectbox("Machine Quality Type", ["L", "M", "H"])
    st.write("*L = Low, M = Medium, H = High Quality*")

# 3. Calculate Engineered Features (Behind the scenes)
temp_diff = process_temp - air_temp
power_stress = torque * rpm

# One-hot encode the machine type
type_l = 1 if machine_type == "L" else 0
type_m = 1 if machine_type == "M" else 0

# 4. Format the data exactly how the model expects it
input_data = pd.DataFrame({
    'Air temperature _K_': [air_temp],
    'Process temperature _K_': [process_temp],
    'Rotational speed _rpm_': [rpm],
    'Torque _Nm_': [torque],
    'Tool wear _min_': [tool_wear],
    'Temp_Difference': [temp_diff],
    'Power_Stress': [power_stress],
    'Type_L': [type_l],
    'Type_M': [type_m]
})

# 5. Make the Prediction
st.markdown("---")
if st.button("Predict Machine Status", type="primary"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Probability of failure (Class 1)

    if prediction[0] == 1:
        st.error(f"⚠️ WARNING: Machine Failure Predicted! (Failure Risk: {probability:.1%})")
        st.write("Recommendation: Schedule immediate maintenance. Check the power stress and temperature differential.")
    else:
        st.success(f"✅ Machine Operating Normally (Failure Risk: {probability:.1%})")