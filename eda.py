# eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

OUTDIR = Path("fraud_outputs")
OUTDIR.mkdir(exist_ok=True)

def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run exploratory data analysis and save plots.
    Returns dataframe (with Hour column added).
    """
    print("EDA: dataset shape", df.shape)
    # Class distribution pie
    cls = df['Class'].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(cls, labels=[f"Legit ({cls[0]})", f"Fraud ({cls[1]})"], autopct="%1.2f%%", startangle=90)
    ax.set_title("Class distribution")
    fig.savefig(OUTDIR / "class_distribution_pie.png")
    plt.close(fig)

    # Amount distributions
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    sns.histplot(df[df['Class']==0]['Amount'], bins=100, ax=axes[0])
    axes[0].set_title("Amount distribution - Legit")
    sns.histplot(df[df['Class']==1]['Amount'], bins=100, ax=axes[1], color='orange')
    axes[1].set_title("Amount distribution - Fraud")
    fig.savefig(OUTDIR / "amount_distribution.png")
    plt.close(fig)

    # Correlation heatmap for V features + Amount
    corr_cols = [c for c in df.columns if c.startswith('V')] + ['Amount','Class']
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation heatmap (V features + Amount)")
    fig.savefig(OUTDIR / "correlation_heatmap.png")
    plt.close(fig)

    # Time-based patterns
    df = df.copy()
    df['Hour'] = ((df['Time'] % (24*3600)) / 3600).astype(int)
    fraud_by_hour = df.groupby('Hour')['Class'].sum()
    legit_by_hour = df.groupby('Hour')['Class'].count() - fraud_by_hour
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(fraud_by_hour.index, fraud_by_hour.values, label='Fraud', color='red')
    ax.plot(legit_by_hour.index, legit_by_hour.values, label='Legit', color='green')
    ax.legend(); ax.set_title("Transactions by hour (fraud vs legit)")
    fig.savefig(OUTDIR / "fraud_by_hour.png")
    plt.close(fig)

    print("EDA plots saved to", OUTDIR.resolve())
    return df
