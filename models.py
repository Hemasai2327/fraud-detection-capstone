# models.py
import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from typing import Dict, Any

def train_and_evaluate_all(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train KNN, SVM, DT, RF and return results dict.
    All models are returned trained on X_train.
    """
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=random_state),
        'DT' : DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=random_state),
        'RF' : RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=random_state)
    }
    results = {}
    for name, clf in models.items():
        print(f"Training & evaluating {name} ...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_t = time.time() - t0

        t0 = time.time()
        try:
            y_proba = clf.predict_proba(X_test)[:,1]
        except Exception:
            try:
                dec = clf.decision_function(X_test)
                y_proba = (dec - dec.min())/(dec.max()-dec.min()+1e-9)
            except Exception:
                y_proba = None
        y_pred = clf.predict(X_test)
        pred_t = time.time() - t0
        pred_per_1000 = pred_t / max(1, (len(X_test)/1000.0))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': clf, 'train_time_s': train_t, 'pred_time_per_1000_s': pred_per_1000,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc,
            'confusion_matrix': cm, 'y_proba': y_proba, 'report': classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"{name} recall={rec:.4f}, precision={prec:.4f}, f1={f1:.4f}, roc_auc={roc if not np.isnan(roc) else 'nan'}")
    return results
