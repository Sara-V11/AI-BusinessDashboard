import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

MODEL_FILE = "delivery_time_model.pkl"
SCALER_FILE = "scaler.pkl"

def train_delivery_model(df: pd.DataFrame):
    features = ["Quantity", "Unit_Price", "Negotiated_Price", "Defect_Rate", "Efficiency", "Order_Value"]
    target = "Delivery_Time"

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler
