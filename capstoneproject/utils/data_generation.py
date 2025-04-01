# utils/data_generation.py
import numpy as np
import pandas as pd

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
