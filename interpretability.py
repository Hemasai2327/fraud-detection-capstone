# interpretability.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
OUTDIR = Path("fraud_outputs")
OUTDIR.mkdir(exist_ok=True)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def save_feature_importances(model, X_train, top_k=15):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        df = pd.DataFrame({'feature': X_train.columns, 'importance': fi}).sort_values('importance', ascending=False)
        top = df.head(top_k)
        top.to_csv(OUTDIR / "feature_importances_top15.csv", index=False)
        # plot
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(range(len(top)), top['importance'][::-1])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top['feature'][::-1])
        ax.set_title("Top feature importances")
        fig.savefig(OUTDIR / "feature_importances_top15.png")
        plt.close(fig)
        print("Saved feature importances and plot.")
    else:
        print("Model lacks feature_importances_. Can't save direct importances.")

def shap_summary(model, X_train, X_test, nsample=200):
    if not SHAP_AVAILABLE:
        print("SHAP not installed; skipping SHAP.")
        return
    explainer = shap.Explainer(model, X_train)
    sv = explainer(X_test[:nsample])
    shap.summary_plot(sv, X_test[:nsample], show=False)
    import matplotlib.pyplot as plt
    plt.savefig(OUTDIR / "shap_summary.png")
    plt.close()
    print("Saved SHAP summary plot.")
