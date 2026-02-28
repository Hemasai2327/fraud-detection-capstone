# tuning.py
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from joblib import dump
import numpy as np

def tune_for_recall(base_model, param_grid, X_train, y_train, subsample_frac=0.5, random_state=42):
    """
    Subsamples training data (preserving class distribution) and performs GridSearchCV
    optimizing recall for fraud class (pos_label=1).
    """
    if subsample_frac < 1.0:
        Xy = X_train.copy()
        Xy['__y__'] = y_train.values
        Xy_sub = Xy.sample(frac=subsample_frac, random_state=random_state)
        y_sub = Xy_sub['__y__']
        X_sub = Xy_sub.drop(columns='__y__')
    else:
        X_sub = X_train
        y_sub = y_train

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scoring = make_scorer(recall_score, pos_label=1)
    gs = GridSearchCV(base_model, param_grid, scoring=scoring, cv=skf, n_jobs=-1, verbose=2)
    gs.fit(X_sub, y_sub)
    print("GridSearch best params:", gs.best_params_, "best recall CV:", gs.best_score_)
    return gs
