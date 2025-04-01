import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Impact Analysis Dashboard", layout="wide")
st.title(" Robotic Waste Sorting Impact Analysis 🤖")
st.markdown("### Economic and Environmental Impact Assessment")




# Info Boxes
st.sidebar.title("ℹ️ About the Metrics")
st.sidebar.markdown("**R-squared:** Indicates how well the model fits the data. A higher value means better accuracy.")
st.sidebar.markdown("**MSE (Mean Squared Error):** Measures the average squared difference between predicted and actual values. Lower is better.")
st.sidebar.markdown("**Revenue:** Estimated revenue generated from sorted materials.")

# Data Generation for Economic Impact
def generate_economic_data(num_samples=100):
    np.random.seed(42)
    sorting_accuracy = np.random.uniform(0.7, 0.99, num_samples)
    volume_processed = np.random.randint(1000, 5000, num_samples)
    material_price = np.random.uniform(50, 150, num_samples)
    revenue = sorting_accuracy * volume_processed * material_price * np.random.uniform(0.8, 1.2, num_samples)
    data = pd.DataFrame({
        'sorting_accuracy': sorting_accuracy,
        'volume_processed': volume_processed,
        'material_price': material_price,
        'revenue': revenue
    })
    data['revenue'] = data['revenue'].apply(lambda x: f"${x:,.2f}")
    return data

# Data Generation for Environmental Impact
def generate_environmental_data(num_years=10):
    np.random.seed(42)
    years = np.arange(2020, 2020 + num_years)
    manual_emissions = np.random.uniform(1000, 2000, num_years)
    automated_emissions = manual_emissions * np.random.uniform(0.4, 0.7, num_years)
    data = pd.DataFrame({
        'year': years,
        'manual_emissions': manual_emissions,
        'automated_emissions': automated_emissions
    })
    return data

# Economic Impact Analysis
def revenue_prediction(data):
    X = data[['sorting_accuracy', 'volume_processed', 'material_price']]
    y = data['revenue'].str.replace(r'[\$,]', '', regex=True).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write('**Revenue Prediction Metrics:**')
    st.write(f'**R-squared:** {r2_score(y_test, y_pred):.4f}')
    st.write(f'**MSE:** {mean_squared_error(y_test, y_pred):.2f}')
    return model

# Environmental Impact Assessment
def carbon_footprint_reduction(data):
    data['reduction'] = data['manual_emissions'] - data['automated_emissions']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_palette("coolwarm")  # Changed to a valid palette
    sns.barplot(data=data.melt('year', value_vars=['manual_emissions', 'automated_emissions'], var_name='Method', value_name='Emissions'), x='year', y='Emissions', hue='Method', ax=ax)
    plt.title('Carbon Footprint Comparison: Manual vs Automated')
    plt.ylabel('Emissions (kg CO₂)')
    plt.xlabel('Year')
    st.pyplot(fig)

# Run the App
st.write("#### Economic Impact Analysis")
economic_data = generate_economic_data()
st.dataframe(economic_data.head())
model = revenue_prediction(economic_data)

st.write("#### Environmental Impact Assessment")
environmental_data = generate_environmental_data()
carbon_footprint_reduction(environmental_data)

st.markdown("---")
st.markdown("**Disclaimer:** The results presented are based on synthetically generated data to demonstrate the potential impact of automated waste sorting. Actual outcomes may vary depending on real-world conditions and system efficiency.")

st.success("Impact Analysis Completed.")
