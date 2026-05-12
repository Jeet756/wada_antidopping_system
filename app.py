import pandas as pd
import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("Anti-Doping Detection System")

st.markdown("Enter the athlete's biological parameters:")

# Feature names
feature_names = [
    'Athlete_ID',
    'Age',
    'Gender',
    'Hemoglobin',
    'Hematocrit',
    'Reticulocyte',
    'TE_Ratio',
    'Cortisol',
    'Number_of_Tests',
    'Test_Type',
    'EPO_Detected',
    'Steroids_Detected',
    'Stimulants_Detected',
    'Passport_Anomalies',
    'Prior_Violations',
    'Competition_Level',
    'Altitude_Training',
    'Heart_Rate',
    'VO2_Max',
    'Average_Speed',
    'Training_Hours_Per_Week',
    'Competition_Wins',
    'Personal_Best',
    'Rank_in_Competition',
    'Injuries_History',
    'Recovery_Time',
    'Stress_Level',
    'Sleep_Hours',
    'Protein_Intake',
    'Calcium_Level',
    'Iron_Level',
    'Mental_Coach',
    'Sponsorship_Status',
    'PB_3_Years_Ago',
    'PB_2_Years_Ago',
    'PB_Last_Year',
    'Avg_Performance_3_Years_Ago',
    'Avg_Performance_2_Years_Ago',
    'Avg_Performance_Last_Year',
    'Performance_Improvement_Rate',
    'Sudden_Performance_Spike',
    'Sport_Cycling',
    'Sport_Skiing',
    'Sport_Swimming',
    'Sport_Weightlifting',
    'Country_China',
    'Country_France',
    'Country_Germany',
    'Country_Italy',
    'Country_Russia',
    'Country_UK',
    'Country_USA'
]

# User inputs
features = []

for name in feature_names:
    value = st.number_input(f"{name}", step=0.01)
    features.append(value)

# Prediction button
if st.button("Detect Doping"):

    # Create dataframe
    input_df = pd.DataFrame([features], columns=feature_names)

    # Convert same as training
    input_df = pd.get_dummies(input_df)

    # Match training features
    model_features = model.feature_names_in_

    # Add missing columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep exact same order
    input_df = input_df[model_features]

    # Predict
    prediction = model.predict(input_df)

    # Probability
    try:
        probability = model.predict_proba(input_df)

        st.write("Prediction Probability:")
        st.write(f"No Doping (0): {probability[0][0]:.4f}")
        st.write(f"Doping (1): {probability[0][1]:.4f}")

    except:
        st.write("Probability not supported")

    # Final result
    if prediction[0] == 1:
        st.error("⚠️ Doping Detected!")
    else:
        st.success("✅ No Doping Detected.")