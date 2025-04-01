# app.py
import streamlit as st
from utils.data_generation import generate_economic_data, generate_environmental_data
from utils.economic_analysis import revenue_prediction, plot_economic_charts
from utils.environmental_analysis import carbon_footprint_reduction, plot_emission_trend
from utils.user_prediction import user_prediction_input

# Page setup
st.set_page_config(page_title="♻️ Robotic Waste Sorting Dashboard", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #00695c;
    }
    .stMetric {
        background-color: #e0f2f1;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("♻️ Robotic Waste Sorting Impact Dashboard")
st.markdown("Analyze how robotic waste sorting affects both **economics** and **environment**.")

# Sidebar Info
st.sidebar.title("ℹ️ About the Metrics")
st.sidebar.markdown("**R-squared:** Indicates how well the model fits the data. A higher value means better accuracy.")
st.sidebar.markdown("**MSE (Mean Squared Error):** Measures the average squared error between prediction and actual values.")
st.sidebar.markdown("**Revenue:** Estimated revenue generated from sorted materials.")

# --- Economic Section ---
st.header("💸 Economic Impact")
economic_data = generate_economic_data()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Sample Economic Data")
    st.dataframe(economic_data.head(), use_container_width=True)

with col2:
    model = revenue_prediction(economic_data)
    user_prediction_input(model)

plot_economic_charts(economic_data, model)

# --- Environmental Section ---
st.header("🌱 Environmental Impact")
environmental_data = generate_environmental_data()

with st.expander("📊 Carbon Footprint Comparison (Manual vs Automated)"):
    carbon_footprint_reduction(environmental_data)

with st.expander("📈 Annual Emissions Reduction Trend"):
    plot_emission_trend(environmental_data)

# --- Footer ---
st.markdown("---")
st.markdown("**Disclaimer:** Results shown are based on synthetic data. Real-world outcomes depend on actual deployment conditions and system efficiency.")
st.success("Dashboard Loaded Successfully ✅")