import pandas as pd

def compute_kpis(df: pd.DataFrame):
    return {
        "avg_order_value": df["Order_Value"].mean(),
        "avg_efficiency": df["Efficiency"].mean() * 100,
        "avg_delivery_time": df["Delivery_Time"].mean(),
        "total_defective_units": df["Defective_Units"].sum(),
        "category_counts": df["Item_Category"].value_counts(),
    }
