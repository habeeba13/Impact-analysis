# utils/environmental_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carbon_footprint_reduction(data):
    data['reduction'] = data['manual_emissions'] - data['automated_emissions']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_palette("coolwarm")
    sns.barplot(data=data.melt('year', value_vars=['manual_emissions', 'automated_emissions'], 
                                var_name='Method', value_name='Emissions'), 
                x='year', y='Emissions', hue='Method', ax=ax)
    plt.title('Carbon Footprint Comparison: Manual vs Automated')
    plt.ylabel('Emissions (kg CO₂)')
    plt.xlabel('Year')
    st.pyplot(fig)

def plot_emission_trend(data):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='year', y='reduction', marker='o', ax=ax)
    ax.set_title("Annual Carbon Reduction with Automation")
    ax.set_ylabel("Reduction in Emissions (kg CO₂)")
    st.pyplot(fig)
