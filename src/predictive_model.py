import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

MODEL_FILE = 'delivery_time_model.pkl'
SCALER_FILE = 'scaler.pkl'

def train_model(df):
    """
    Train a normalized Linear Regression model to predict Delivery Time.
    """
    features = ['Quantity', 'Unit_Price', 'Order_Value']
    target = 'Delivery_Time'

    X = df[features]
    y = df[target]

    # Remove outliers (delivery time > 100 days)
    df = df[df['Delivery_Time'] < 100]
    X = df[features]
    y = df[target]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler

def predict_delivery_time(model, scaler, input_data):
    """
    Predict delivery time (in days) for new orders.
    """
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)
    # Clamp predictions to a reasonable range
    prediction = np.clip(prediction, 0, 60)
    return prediction
