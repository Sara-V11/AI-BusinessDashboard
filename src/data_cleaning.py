import pandas as pd

def preprocess_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert dates
    for col in ["Order_Date", "Delivery_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derived metrics
    df["Order_Value"] = df["Quantity"] * df["Unit_Price"]
    df["Defect_Rate"] = df["Defective_Units"] / (df["Quantity"] + 1e-5)
    df["Delivery_Time"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days.fillna(0)
    df["Efficiency"] = (1 - df["Defect_Rate"]) * (df["Order_Value"] / df["Order_Value"].max())

    # Filter out invalid rows
    df = df[df["Delivery_Time"] >= 0].fillna(0)

    return df
