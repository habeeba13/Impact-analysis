# utils/user_prediction.py
import streamlit as st
import numpy as np

def user_prediction_input(model):
    st.sidebar.subheader("📈 Predict Revenue")
    acc_input = st.sidebar.slider("Sorting Accuracy", 0.7, 0.99, 0.85)
    vol_input = st.sidebar.number_input("Volume Processed (kg)", 1000, 5000, 3000)
    price_input = st.sidebar.number_input("Material Price ($/kg)", 50.0, 150.0, 100.0)
    
    prediction = model.predict(np.array([[acc_input, vol_input, price_input]]))
    st.sidebar.write(f"### 💰 Predicted Revenue: ${prediction[0]:,.2f}")
