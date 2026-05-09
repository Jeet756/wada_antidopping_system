
import pandas as pd
import streamlit as st
import numpy as np
import joblib
model = joblib.load('model.pkl')

st.title("Anti-Doping Detection System")

st.markdown("Enter the athlete's biological parameters:")
feature_names=['Athlete_ID',
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
 'Country_USA']
features = []
for name in feature_names:
    val = st.number_input(f"{name}", step=0.01)
    features.append(val)

if st.button("Detect Doping"):
    input_data = np.array([features])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Doping Detected!")
    else:
        st.success("✅ No Doping Detected.")