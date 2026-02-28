# app.py — Streamlit demo for Fraud Detection
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from joblib import load

OUTDIR = Path("fraud_outputs")

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🔍", layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Upload a transaction or enter values manually to check if it's fraudulent.")

# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    model_path = OUTDIR / "best_model.joblib"
    preprocess_path = OUTDIR / "preprocess.pkl"
    if not model_path.exists() or not preprocess_path.exists():
        return None, None
    model = load(model_path)
    with open(preprocess_path, "rb") as f:
        preprocess = pickle.load(f)
    return model, preprocess

model, preprocess = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please run `python main.py` first to train and save the model.")
    st.stop()

st.success("Model loaded successfully!")

tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

with tab1:
    st.subheader("Enter Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
        time   = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)
    with col2:
        threshold = st.slider("Detection Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01,
                              help="Lower = more sensitive to fraud, more false positives")

    st.markdown("**PCA Features (V1–V28)** — leave at 0 if unknown")
    v_cols = {}
    cols = st.columns(4)
    for i in range(1, 29):
        v_cols[f"V{i}"] = cols[(i-1) % 4].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

    if st.button("🔎 Predict", use_container_width=True):
        row = {"Time": time, "Amount": amount, **v_cols}
        df_input = pd.DataFrame([row])
        Xp = preprocess['preprocess_new'](df_input)
        proba = model.predict_proba(Xp)[:, 1][0]
        pred  = int(proba >= threshold)

        st.divider()
        if pred == 1:
            st.error(f"🚨 **FRAUD DETECTED** — Confidence: {proba*100:.1f}%")
        else:
            st.success(f"✅ **Legitimate Transaction** — Fraud probability: {proba*100:.1f}%")

        st.progress(float(proba), text=f"Fraud probability: {proba:.4f}")

with tab2:
    st.subheader("Batch Prediction via CSV")
    st.markdown("Upload a CSV with the same columns as the training data (excluding `Class`).")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'Class' in df_up.columns:
            df_up = df_up.drop(columns=['Class'])
        try:
            Xp = preprocess['preprocess_new'](df_up)
            probas = model.predict_proba(Xp)[:, 1]
            threshold_batch = st.slider("Threshold", 0.1, 0.9, 0.5, 0.01, key="batch_thresh")
            preds = (probas >= threshold_batch).astype(int)
            df_up['Fraud_Probability'] = probas
            df_up['Prediction'] = preds
            df_up['Label'] = df_up['Prediction'].map({0: '✅ Legit', 1: '🚨 Fraud'})

            fraud_count = preds.sum()
            st.metric("Transactions", len(df_up))
            st.metric("Flagged as Fraud", int(fraud_count))
            st.metric("Fraud Rate", f"{fraud_count/len(df_up)*100:.2f}%")

            st.dataframe(df_up[['Fraud_Probability', 'Label']].head(50))
            csv = df_up.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.divider()
st.caption("Fraud Detection Capstone | Model: Random Forest | Dataset: Kaggle Credit Card Fraud")