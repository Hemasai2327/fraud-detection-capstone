# main.py

import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer

from eda import run_eda
from preprocessing import split_and_preprocess
from models import train_and_evaluate_all
from tuning import tune_for_recall
from imbalance import apply_smote_and_eval, balanced_rf_eval
from interpretability import save_feature_importances, shap_summary
from deployment import FraudDetectionSystem, save_artifacts


# Paths


BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "fraud_outputs"
OUTDIR.mkdir(exist_ok=True)

DATA_PATH = Path("C:/Users/hemas/fraud-detection-deployment/creditcard.csv")


def main():

    print("Loading dataset from:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Part 1: EDA
    df = run_eda(df)

    # Part 2: Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, preprocess = split_and_preprocess(df)

    imputer = SimpleImputer(strategy='median')

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Part 3: Baseline Models
    results = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    rows = []
    for k, v in results.items():
        rows.append({
            'Algorithm': k,
            'TrainTime_s': v['train_time_s'],
            'PredTimePer1000_s': v['pred_time_per_1000_s'],
            'Accuracy': v['accuracy'],
            'Precision': v['precision'],
            'Recall': v['recall'],
            'F1': v['f1'],
            'ROC_AUC': v['roc_auc']
        })

    pd.DataFrame(rows).to_csv(OUTDIR / "part3_comparison.csv", index=False)

    # Best model by Recall
    best_name = max(results.items(), key=lambda kv: kv[1]['recall'])[0]
    print("Best baseline model by recall:", best_name)

    base_model = results[best_name]['model']

    # Part 4: Hyperparameter Tuning
    if best_name == 'RF':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', {0: 1, 1: 50}, {0: 1, 1: 100}]
        }

        gs = tune_for_recall(base_model, param_grid, X_train, y_train, subsample_frac=0.5)
        best_model = gs.best_estimator_

    else:
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        gs = tune_for_recall(rf, param_grid, X_train, y_train, subsample_frac=0.5)
        best_model = gs.best_estimator_

    # Evaluate tuned model
    best_model.fit(X_train, y_train)

    from sklearn.metrics import recall_score, precision_score

    y_pred = best_model.predict(X_test)

    print("Tuned model recall on test:",
          recall_score(y_test, y_pred))

    print("Tuned model precision on test:",
          precision_score(y_test, y_pred, zero_division=0))

    # Part 5: Imbalance Handling
    smote_res = apply_smote_and_eval(
        best_model, X_train, y_train, X_test, y_test, strategy=0.1
    )

    brf_res = balanced_rf_eval(
        X_train, y_train, X_test, y_test
    )

    table5 = pd.DataFrame([
        {
            'Technique': 'Baseline',
            'Recall': results[best_name]['recall'],
            'Precision': results[best_name]['precision'],
            'F1': results[best_name]['f1'],
            'Threshold': 0.5
        },
        {
            'Technique': 'SMOTE',
            'Recall': smote_res['recall'],
            'Precision': smote_res['precision'],
            'F1': smote_res['f1'],
            'Threshold': 0.5
        },
        {
            'Technique': 'BalancedRF',
            'Recall': brf_res['recall'],
            'Precision': brf_res['precision'],
            'F1': brf_res['f1'],
            'Threshold': 0.5
        }
    ])

    table5.to_csv(OUTDIR / "part5_comparison.csv", index=False)

    # Part 6: Interpretability
    save_feature_importances(best_model, X_train, top_k=15)
    shap_summary(best_model, X_train, X_test, nsample=200)

    # Part 7: Save Artifacts & Demo Deployment
    save_artifacts(best_model, preprocess)

    fds = FraudDetectionSystem(best_model, preprocess, threshold=0.5)

    sample = X_test.iloc[[0]]
    proba, pred = fds.predict_fraud(sample)

    print("Sample proba/pred:", proba, pred)
    print("Latency (ms):", fds.latency_ms(sample, repeats=100))


if __name__ == "__main__":
    main()