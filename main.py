# main.py

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)

from eda import run_eda
from preprocessing import split_and_preprocess
from models import train_and_evaluate_all
from tuning import tune_for_recall
from imbalance import apply_smote_and_eval, balanced_rf_eval
from interpretability import save_feature_importances, shap_summary
from deployment import FraudDetectionSystem, save_artifacts

BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "fraud_outputs"
OUTDIR.mkdir(exist_ok=True)


def find_best_threshold(model, X_val, y_val):
    """Find threshold that maximizes F1 on validation set."""
    try:
        proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        dec = model.decision_function(X_val)
        proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions[:-1], label='Precision')
    ax.plot(thresholds, recalls[:-1], label='Recall')
    ax.plot(thresholds, f1_scores[:-1], label='F1')
    ax.axvline(thresholds[np.argmax(f1_scores[:-1])], color='red', linestyle='--', label='Best threshold')
    ax.set_xlabel('Threshold')
    ax.set_title('Precision / Recall / F1 vs Threshold')
    ax.legend()
    fig.savefig(OUTDIR / "threshold_optimization.png")
    plt.close(fig)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    print(f"Best threshold: {best_threshold:.4f} — F1: {f1_scores[best_idx]:.4f}")
    return best_threshold


def evaluate_full(name, model, X_test, y_test, threshold=0.5):
    """Return a dict of full metrics for a model at a given threshold."""
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        dec = model.decision_function(X_test)
        proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'Model':     name,
        'Threshold': threshold,
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'F1':        f1_score(y_test, y_pred, zero_division=0),
        'ROC_AUC':   roc_auc_score(y_test, proba),
        'TP': int(cm[1, 1]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TN': int(cm[0, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Capstone Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard.csv",
        help="Path to creditcard.csv (default: ./creditcard.csv)"
    )
    args = parser.parse_args()

    DATA_PATH = Path(args.data)
    print("Loading dataset from:", DATA_PATH.resolve())

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and run: python main.py --data path/to/creditcard.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # Part 1: EDA
    df = run_eda(df)

    # Part 2: Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, preprocess = split_and_preprocess(df)

    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val   = pd.DataFrame(imputer.transform(X_val),       columns=X_val.columns,   index=X_val.index)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    # Part 3: Baseline Models
    results = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    rows = []
    for k, v in results.items():
        rows.append({
            'Algorithm':         k,
            'TrainTime_s':       v['train_time_s'],
            'PredTimePer1000_s': v['pred_time_per_1000_s'],
            'Accuracy':          v['accuracy'],
            'Precision':         v['precision'],
            'Recall':            v['recall'],
            'F1':                v['f1'],
            'ROC_AUC':           v['roc_auc']
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "part3_comparison.csv", index=False)

    best_name = max(results.items(), key=lambda kv: kv[1]['recall'])[0]
    print("Best baseline model by recall:", best_name)
    base_model = results[best_name]['model']

    # Part 4: Hyperparameter Tuning
    if best_name == 'RF':
        param_grid = {
            'n_estimators':      [100, 200, 300],
            'max_depth':         [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'class_weight':      ['balanced', {0: 1, 1: 50}, {0: 1, 1: 100}]
        }
        gs = tune_for_recall(base_model, param_grid, X_train, y_train, subsample_frac=0.5)
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators':      [100, 200],
            'max_depth':         [10, 20, None],
            'min_samples_split': [2, 5]
        }
        gs = tune_for_recall(rf, param_grid, X_train, y_train, subsample_frac=0.5)

    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train)

    # Threshold optimization using validation set
    best_threshold = find_best_threshold(best_model, X_val, y_val)

    # Part 5: Imbalance Handling
    smote_res = apply_smote_and_eval(best_model, X_train, y_train, X_test, y_test, strategy=0.1)
    brf_res   = balanced_rf_eval(X_train, y_train, X_test, y_test)

    # Full metrics comparison
    full_comparison = pd.DataFrame([
        evaluate_full("Tuned RF (threshold=0.5)",                        best_model,        X_test, y_test, threshold=0.5),
        evaluate_full(f"Tuned RF (optimized threshold={best_threshold:.3f})", best_model,   X_test, y_test, threshold=best_threshold),
        evaluate_full("SMOTE + Tuned RF",                                smote_res['model'], X_test, y_test, threshold=0.5),
        evaluate_full("Balanced Random Forest",                          brf_res['model'],   X_test, y_test, threshold=0.5),
    ])
    full_comparison.to_csv(OUTDIR / "part5_full_comparison.csv", index=False)
    print("\nFull model comparison:")
    print(full_comparison.to_string(index=False))

    # Part 6: Interpretability
    save_feature_importances(best_model, X_train, top_k=15)
    shap_summary(best_model, X_train, X_test, nsample=200)

    # Part 7: Save Artifacts & Demo Deployment
    save_artifacts(best_model, preprocess)

    fds = FraudDetectionSystem(best_model, preprocess, threshold=best_threshold)
    sample = X_test.iloc[[0]]
    proba, pred = fds.predict_fraud(sample)
    print("\nSample proba/pred:", proba, pred)
    print("Latency (ms):", fds.latency_ms(sample, repeats=100))
    print("\nPipeline complete. All outputs saved to:", OUTDIR.resolve())


if __name__ == "__main__":
    main()