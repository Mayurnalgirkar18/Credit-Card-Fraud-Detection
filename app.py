import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to check fraud risk.")

# Input fields
distance_from_home = st.number_input("Distance From Home", min_value=0.0)
distance_from_last_transaction = st.number_input("Distance From Last Transaction", min_value=0.0)
ratio_to_median_purchase_price = st.number_input("Ratio To Median Purchase Price", min_value=0.0)
repeat_retailer = st.selectbox("Repeat Retailer", [0,1])
used_chip = st.selectbox("Used Chip", [0,1])
used_pin_number = st.selectbox("Used PIN Number", [0,1])
online_order = st.selectbox("Online Order", [0,1])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("card_transdata.csv")
    return df

df = load_data()

X = df.drop(columns=['fraud'])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train model
model = KMeans(n_clusters=5, random_state=42)
model.fit(X_pca)

# Prediction
if st.button("Check Transaction"):

    input_data = np.array([[distance_from_home,
                            distance_from_last_transaction,
                            ratio_to_median_purchase_price,
                            repeat_retailer,
                            used_chip,
                            used_pin_number,
                            online_order]])

    scaled = scaler.transform(input_data)
    pca_input = pca.transform(scaled)

    cluster = model.predict(pca_input)

    st.success(f"Transaction belongs to cluster: {cluster[0]}")

    if cluster[0] in [3,4]:
        st.error("⚠️ Possible Fraud Transaction")
    else:
        st.success("✅ Likely Normal Transaction")