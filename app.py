import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Online Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")
SAMPLE_PATH = os.path.join(BASE, "sample_data", "online_retail_sample_rfm.csv")

@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    required = {"pipeline", "segment_map", "feature_order"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(
            "model.pkl has an unexpected format. Use the model.pkl supplied "
            "with this project package."
        )
    return artifact

@st.cache_data
def load_sample():
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)
    return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary", "Segment"])

def predict_segment(artifact, recency, frequency, monetary):
    values = [[recency, frequency, monetary]]
    X = np.log1p(np.asarray(values, dtype=float))
    cluster = int(artifact["pipeline"].predict(X)[0])
    segment = artifact["segment_map"].get(cluster, f"Cluster {cluster}")
    return cluster, segment

artifact = load_model()
st.title("🛍️ Online Retail Customer Segmentation")
st.caption(
    "RFM-based customer segmentation using K-Means clustering. "
    "This deployed model has two learned segments."
)

with st.sidebar:
    st.header("About the model")
    st.write("The model groups customers using Recency, Frequency and Monetary (RFM) behaviour.")
    st.markdown(
        "- **Recency:** days since the latest purchase\n"
        "- **Frequency:** number of orders\n"
        "- **Monetary:** total customer spend"
    )
    st.info("Input values are transformed using log1p and then passed through the fitted StandardScaler + K-Means pipeline.")

left, right = st.columns(2)

with left:
    st.subheader("Test a customer")
    recency = st.number_input(
        "Recency (days)", min_value=1, max_value=3650, value=30, step=1,
        help="Lower recency means a more recent purchase."
    )
    frequency = st.number_input(
        "Frequency (orders)", min_value=1, max_value=1000, value=20, step=1,
        help="Number of distinct orders."
    )
    monetary = st.number_input(
        "Monetary value (£)", min_value=1.0, max_value=1000000.0, value=8000.0, step=50.0,
        help="Total customer spend."
    )
    if st.button("Predict Customer Segment", type="primary", use_container_width=True):
        cluster, segment = predict_segment(artifact, recency, frequency, monetary)
        st.success(f"Predicted segment: **{segment}**")
        st.metric("Cluster", cluster)

with right:
    st.subheader("Built-in customer sample")
    sample_df = load_sample()
    if not sample_df.empty:
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
    else:
        st.info("Sample customer file is not present. Use the examples below.")

st.divider()
st.subheader("Test examples")
example_rows = [
    ("High-value / recent", 5, 12, 2500),
    ("High-value / frequent spender", 30, 20, 8000),
    ("Low-value / inactive", 150, 1, 75),
]
example_display = []
for label, r, f, m in example_rows:
    cluster, segment = predict_segment(artifact, r, f, m)
    example_display.append({
        "Example": label,
        "Recency (days)": r,
        "Frequency (orders)": f,
        "Monetary (£)": m,
        "Model prediction": segment,
    })
st.dataframe(pd.DataFrame(example_display), use_container_width=True, hide_index=True)

st.subheader("How to interpret the result")
st.markdown(
    "**High-Value Customer:** the cluster with relatively lower recency and higher "
    "frequency and/or monetary value.\n\n"
    "**Low-Value Customer:** the cluster with relatively higher recency and lower "
    "purchasing activity/value."
)
st.caption(
    "Model artifact: model.pkl | Input transformation: log1p(RFM) | "
    "Algorithm: StandardScaler + K-Means (2 clusters)"
)
