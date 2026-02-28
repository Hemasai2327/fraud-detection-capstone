# deployment.py
import time
from joblib import dump, load
import pickle
import pandas as pd
from pathlib import Path

OUTDIR = Path("fraud_outputs")
OUTDIR.mkdir(exist_ok=True)

class FraudDetectionSystem:
    def __init__(self, model, preprocess_pipeline, threshold: float = 0.5):
        # model: trained sklearn model or path to joblib file
        if isinstance(model, (str, Path)):
            self.model = load(model)
        else:
            self.model = model
        self.preprocess = preprocess_pipeline
        self.threshold = threshold

    def predict_fraud(self, transaction_df: pd.DataFrame):
        # expects a DataFrame with same columns as training features
        Xp = self.preprocess['preprocess_new'](transaction_df)
        try:
            proba = self.model.predict_proba(Xp)[:,1]
        except Exception:
            dec = self.model.decision_function(Xp)
            proba = (dec - dec.min())/(dec.max()-dec.min()+1e-9)
        preds = (proba >= self.threshold).astype(int)
        return proba, preds

    def latency_ms(self, transaction_df: pd.DataFrame, repeats=100):
        t0 = time.time()
        for _ in range(repeats):
            _ = self.predict_fraud(transaction_df)
        t1 = time.time()
        return ((t1 - t0)/repeats) * 1000.0

def save_artifacts(model, preprocess_pipeline, model_path="fraud_outputs/best_model.joblib", preprocess_path="fraud_outputs/preprocess.pkl"):
    dump(model, model_path)
    with open(preprocess_path, "wb") as f:
        pickle.dump(preprocess_pipeline, f)
    print("Saved model and preprocess pipeline.")
