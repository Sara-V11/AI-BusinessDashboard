import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from src.data_cleaning import preprocess_dataset
from src.kpi_analysis import compute_kpis
from src.model_training import train_delivery_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Process Automation Dashboard", layout="wide", page_icon="🤖")
st.markdown("""
    <style>
        body {background-color: #0e1117; color: #fafafa;}
        .metric-card {background-color: #1c1f26; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
        .metric-card h2 {color: #00b4d8; margin-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Based Process Automation Dashboard")
st.caption("Smart KPI Monitoring • Automated Insights • Predictive Analytics")

# ---------------- DATA LOADING ----------------
DATA_PATH = "data/processed_kpi.csv"
df = preprocess_dataset(DATA_PATH)

# ---------------- MODEL ----------------
if not os.path.exists("delivery_time_model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("Training AI model..."):
        model, scaler = train_delivery_model(df)
else:
    model = joblib.load("delivery_time_model.pkl")
    scaler = joblib.load("scaler.pkl")

# ---------------- KPI METRICS ----------------
kpis = compute_kpis(df)

st.subheader("📊 Business Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-card'><h2>Average Order Value</h2><h3>${kpis['avg_order_value']:.2f}</h3></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h2>Average Efficiency</h2><h3>{kpis['avg_efficiency']:.2f}%</h3></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h2>Avg Delivery Time</h2><h3>{kpis['avg_delivery_time']:.2f} days</h3></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h2>Total Defective Units</h2><h3>{int(kpis['total_defective_units'])}</h3></div>", unsafe_allow_html=True)

# ---------------- VISUALS ----------------
st.divider()
st.subheader("📦 Order Distribution by Category")

# Prepare category counts safely
if "Item_Category" in df.columns and df["Item_Category"].nunique() > 0:
    cat_counts = df["Item_Category"].value_counts().reset_index()
    cat_counts.columns = ["Item_Category", "Count"]
else:
    cat_counts = pd.DataFrame({"Item_Category": ["No data"], "Count": [0]})

# Try Plotly for interactive charts, fallback to Matplotlib
try:
    import plotly.express as px

    fig1 = px.bar(
        cat_counts,
        x="Item_Category",
        y="Count",
        title="Order Distribution by Category",
        labels={"Item_Category": "Item Category", "Count": "Orders"},
        hover_data=["Count"]
    )
    fig1.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(14,17,23,1)",
        paper_bgcolor="rgba(14,17,23,1)",
        font=dict(color="white")
    )
    st.plotly_chart(fig1, use_container_width=True)

except Exception:
    # Matplotlib fallback
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(cat_counts["Item_Category"], cat_counts["Count"])
    ax1.set_ylabel("Orders")
    ax1.set_xlabel("Item Category")
    plt.xticks(rotation=30)
    st.pyplot(fig1)


st.divider()
st.subheader("⚙️ Efficiency vs Project Progress")

# Ensure Project_Progress exists (fallback to index-based progress)
if "Project_Progress" not in df.columns:
    df["Project_Progress"] = (df.index - df.index.min()) / max(1, (df.index.max() - df.index.min()))

# Safe default columns
color_col = "Defect_Rate" if "Defect_Rate" in df.columns else None
size_col = "Order_Value" if "Order_Value" in df.columns else None
hover_cols = []
for c in ["Quantity", "Item_Category", "Delivery_Time"]:
    if c in df.columns:
        hover_cols.append(c)
hover_cols = hover_cols if hover_cols else None

try:
    import plotly.express as px

    fig2 = px.scatter(
        df,
        x="Project_Progress",
        y="Efficiency",
        color=color_col,
        size=size_col,
        hover_data=hover_cols,
        title="Efficiency vs Project Progress",
        labels={"Project_Progress": "Project Progress", "Efficiency": "Efficiency"}
    )
    fig2.update_layout(
        plot_bgcolor="rgba(14,17,23,1)",
        paper_bgcolor="rgba(14,17,23,1)",
        font=dict(color="white")
    )
    st.plotly_chart(fig2, use_container_width=True)

except Exception:
    # Matplotlib fallback (colored by defect rate if present)
    import matplotlib.pyplot as plt
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if color_col:
        sc = ax2.scatter(df["Project_Progress"], df["Efficiency"], c=df[color_col], cmap="coolwarm", s=(df[size_col] / (df[size_col].max()+1) * 100 + 10) if size_col else 40)
        plt.colorbar(sc, label=color_col)
    else:
        ax2.scatter(df["Project_Progress"], df["Efficiency"], s=40)
    ax2.set_xlabel("Project Progress")
    ax2.set_ylabel("Efficiency")
    st.pyplot(fig2)

# ---------------- AI PREDICTION ----------------
st.divider()
st.subheader("🤖 Predict Delivery Time")

colA, colB = st.columns(2)
with colA:
    quantity = st.number_input("Quantity", 1, 1000, 10)
    unit_price = st.number_input("Unit Price ($)", 0.1, 10000.0, 200.0)
    negotiated_price = st.number_input("Negotiated Price ($)", 0.1, 10000.0, 180.0)
with colB:
    defect_rate = st.number_input("Defect Rate", 0.0, 1.0, 0.02)
    efficiency = st.number_input("Efficiency", 0.0, 1.0, 0.85)

order_value = quantity * unit_price

if st.button("🚀 Predict Now"):
    X_input = pd.DataFrame([[quantity, unit_price, negotiated_price, defect_rate, efficiency, order_value]],
                           columns=["Quantity", "Unit_Price", "Negotiated_Price", "Defect_Rate", "Efficiency", "Order_Value"])
    X_scaled = scaler.transform(X_input)
    predicted_time = model.predict(X_scaled)[0]
    st.success(f"🕒 Predicted Delivery Time: **{predicted_time:.2f} days**")

    if predicted_time < 5:
        st.info("✅ Excellent! Fast delivery expected.")
    elif predicted_time < 10:
        st.warning("⚠️ Moderate delivery time. Possible delay risk.")
    else:
        st.error("🚨 High delay risk detected! Consider supplier optimization.")
