import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Online Retail Customer Segmentation', page_icon='🛍️', layout='wide')

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'model.pkl')
SAMPLE_PATH = os.path.join(BASE, 'sample_data', 'online_retail_sample_rfm.csv')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)

bundle = load_model()
pipeline = bundle['pipeline']
segment_map = bundle['segment_map']

st.title('🛍️ Online Retail Customer Segmentation')
st.caption('RFM-based customer segmentation using K-Means clustering — recruiter-friendly demo with built-in sample inputs.')

with st.sidebar:
    st.header('About the model')
    st.write('The model groups customers using Recency, Frequency and Monetary (RFM) behaviour.')
    st.markdown('**RFM**')
    st.write('- Recency: days since the latest purchase\n- Frequency: number of orders\n- Monetary: total customer spend')
    st.info('The included sample data lets you test the deployed app without uploading a file.')

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader('Test a customer')
    recency = st.number_input('Recency (days)', min_value=1, max_value=3650, value=18, step=1, help='Typical demo range: 1–365 days. Lower is more recent.')
    frequency = st.number_input('Frequency (orders)', min_value=1, max_value=1000, value=6, step=1, help='Typical demo range: 1–100 orders.')
    monetary = st.number_input('Monetary value (£)', min_value=1.0, max_value=1000000.0, value=1200.0, step=50.0, help='Total customer revenue/spend. Example range: £50–£20,000.')
    if st.button('Predict Customer Segment', type='primary', use_container_width=True):
        X = np.log1p([[recency, frequency, monetary]])
        cluster = int(pipeline.predict(X)[0])
        segment = segment_map.get(cluster, f'Cluster {cluster}')
        st.success(f'Predicted segment: **{segment}**')
        st.metric('Cluster', cluster)

with col2:
    st.subheader('Built-in sample data')
    sample = load_sample()
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.caption('Select a sample row and test its values above, or use the examples below.')
    examples = pd.DataFrame({
        'Example': ['High-value / recent', 'Regular customer', 'Low-value / inactive'],
        'Recency (days)': [5, 30, 150],
        'Frequency (orders)': [12, 20, 1],
        'Monetary (£)': [2500, 8000, 75]
    })
    st.dataframe(examples, use_container_width=True, hide_index=True)

st.divider()
st.subheader('How to interpret the result')
st.markdown('**High-Value Customer:** relatively recent activity, higher order frequency and/or higher monetary value.  \n**Low-Value Customer:** less recent activity and lower purchasing activity/value.')
st.caption('Model artifact: model.pkl • Input transformation: log1p(RFM) • Algorithm: StandardScaler + K-Means')
