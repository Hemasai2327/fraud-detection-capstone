# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
OUTDIR = Path("fraud_outputs")
OUTDIR.mkdir(exist_ok=True)

def split_and_preprocess(df: pd.DataFrame, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits into train/val/test (stratified) and returns processed DataFrames and a preprocess object.
    Preprocessing:
      - median impute Amount and Time if necessary
      - standard scale Amount and Time
      - leave V1..V28 as-is (already transformed in dataset)
    Returns:
      X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, preprocess_dict
    """
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random_state)

    numeric_cols = ['Amount','Time']
    other_cols = [c for c in X.columns if c not in numeric_cols]

    # Fit imputer & scaler on numeric cols
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_num = imputer.fit_transform(X_train[numeric_cols])
    X_train_num = scaler.fit_transform(X_train_num)

    X_val_num = scaler.transform(imputer.transform(X_val[numeric_cols]))
    X_test_num = scaler.transform(imputer.transform(X_test[numeric_cols]))

    # Rebuild DataFrames with same column order
    final_cols = numeric_cols + other_cols
    X_train_proc = pd.DataFrame(X_train_num, index=X_train.index, columns=numeric_cols).join(X_train[other_cols].reset_index(drop=True))[final_cols]
    X_val_proc = pd.DataFrame(X_val_num, index=X_val.index, columns=numeric_cols).join(X_val[other_cols].reset_index(drop=True))[final_cols]
    X_test_proc = pd.DataFrame(X_test_num, index=X_test.index, columns=numeric_cols).join(X_test[other_cols].reset_index(drop=True))[final_cols]

    preprocess = {
        'imputer': imputer,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'other_cols': other_cols,
        'final_cols': final_cols,
        'preprocess_new': lambda X_new: pd.DataFrame(scaler.transform(imputer.transform(X_new[numeric_cols])), index=X_new.index, columns=numeric_cols).join(X_new[other_cols].reset_index(drop=True))[final_cols]
    }

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, preprocess
