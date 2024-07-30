import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
import pickle

warnings.filterwarnings('ignore')

# Streamlit app title
st.title('DPS Challenge Mission 1')

# Load data
data_path = r'C:\Users\amide\OneDrive\Documents\GitHub\mission_dps\monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv'
try:
    df_initial = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Data file not found at path: {data_path}")
    st.stop()

st.header('Initial Data')
st.write(df_initial.head())

# Data preprocessing
df = df_initial[df_initial['JAHR'] <= 2020].reset_index(drop=True)
df = df[['MONATSZAHL', 'AUSPRAEGUNG', 'MONAT', 'WERT']]
df['MONATSZAHL'] = df['MONATSZAHL'].astype("string")
df['AUSPRAEGUNG'] = df['AUSPRAEGUNG'].astype("string")
df['MONAT'] = pd.to_datetime(df['MONAT'].astype("string"), format='%Y%m', errors='coerce')
df = df.dropna(subset=['MONAT'])
df = df.sort_values(by='MONAT').reset_index(drop=True)

st.header('Processed Data')
st.write(df.head())

# Plotting historical number of accidents per category
st.header('Historical Number of Accidents per Category')
fig = px.line(df, x='MONAT', y='WERT', color='MONATSZAHL', title='Historical Number of Accidents per Category')
st.plotly_chart(fig)

# Linear Regression for 'Alkoholunfälle'
filtered_df = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')]
filtered_df['MONTH_NUMERIC'] = filtered_df['MONAT'].apply(lambda x: x.toordinal())

features = ['MONTH_NUMERIC']
target = 'WERT'

X_train, X_test, y_train, y_test = train_test_split(filtered_df[features], filtered_df[target], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')

# Forecasting for December 2023
forecast_date = pd.to_datetime('2023-12')
forecast_month_numeric = forecast_date.toordinal()
forecasted_value = model.predict([[forecast_month_numeric]])[0]
st.write(f'Forecasted Value for January 2024: {forecasted_value}')

# Get ground truth value for January 2024
try:
    ground_truth = df_initial[
        (df_initial['MONATSZAHL'] == 'Alkoholunfälle') &
        (df_initial['AUSPRAEGUNG'] == 'insgesamt') &
        (df_initial['MONAT'] == 202401)
    ]['WERT'].values[0]
    st.write(f'Ground Truth Value for January 2024: {ground_truth}')
except IndexError:
    st.write('Ground Truth Value not found for January 2024')

# Plotting historical data, model prediction, and actual value
st.header('Historical Data, Model Prediction, and Actual Value')
fig = px.line(filtered_df, x='MONAT', y='WERT', title='Historical Data, Model Prediction, and Actual Value')
fig.add_scatter(x=[forecast_date], y=[forecasted_value], mode='markers', marker=dict(color='orange', size=10), name='Forecasted Value')
if 'ground_truth' in locals():
    fig.add_scatter(x=[pd.to_datetime('2024-01')], y=[ground_truth], mode='markers', marker=dict(color='red', size=10), name='Actual Value')
st.plotly_chart(fig)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
st.write("Model saved successfully.")
