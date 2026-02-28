# imbalance.py
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
from sklearn.base import clone
from models import train_and_evaluate_all
from sklearn.metrics import recall_score, precision_score, f1_score

def apply_smote_and_eval(model, X_train, y_train, X_test, y_test, strategy=0.1, random_state=42):
    sm = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE counts:", dict(zip(*np.unique(y_res, return_counts=True))))
    # clone model to avoid overwriting
    m = clone(model)
    m.fit(X_res, y_res)
    try:
        y_proba = m.predict_proba(X_test)[:,1]
    except Exception:
        y_proba = None
    y_pred = m.predict(X_test)
    return {'model': m, 'recall': recall_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0)}

def balanced_rf_eval(X_train, y_train, X_test, y_test, random_state=42):
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=random_state)
    brf.fit(X_train, y_train)
    y_pred = brf.predict(X_test)
    return {'model': brf, 'recall': recall_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0)}
