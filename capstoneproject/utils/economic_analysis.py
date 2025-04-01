# utils/economic_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

def plot_economic_charts(data, model):
    numeric_data = data.copy()
    numeric_data['revenue'] = numeric_data['revenue'].str.replace(r'[\$,]', '', regex=True).astype(float)

    st.write("### Revenue vs Sorting Accuracy")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=numeric_data, x='sorting_accuracy', y='revenue', ax=ax1)
    st.pyplot(fig1)

    st.write("### Correlation Matrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu", ax=ax2)
    st.pyplot(fig2)

    st.write("### Model Feature Importance")
    feature_names = ['sorting_accuracy', 'volume_processed', 'material_price']
    coeffs = pd.DataFrame(model.coef_, index=feature_names, columns=['Coefficient'])
    st.dataframe(coeffs)
