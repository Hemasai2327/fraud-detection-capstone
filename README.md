# Fraud Detection Capstone

A machine learning project to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Project Structure

```
fraud-detection-capstone/
├── main.py               # Entry point — runs the full pipeline
├── eda.py                # Exploratory data analysis & visualizations
├── preprocessing.py      # Data splitting, imputation, and scaling
├── models.py             # Baseline model training & evaluation (KNN, SVM, DT, RF)
├── tuning.py             # Hyperparameter tuning with GridSearchCV (optimized for recall)
├── imbalance.py          # Imbalance handling via SMOTE and Balanced Random Forest
├── interpretability.py   # Feature importances and SHAP explainability
├── deployment.py         # Model saving, loading, and inference system
└── fraud_outputs/        # Generated outputs (plots, CSVs, saved models)
```

## Pipeline Overview

1. EDA — Class distribution, amount distributions, correlation heatmap, fraud by hour
2. Preprocessing — Stratified train/val/test split, median imputation, standard scaling of Amount and Time
3. Baseline Models — KNN, SVM, Decision Tree, Random Forest evaluated on accuracy, precision, recall, F1, ROC-AUC
4. Hyperparameter Tuning — GridSearchCV optimized for fraud recall using stratified k-fold CV
5. Imbalance Handling — SMOTE oversampling and Balanced Random Forest comparison
6. Interpretability — Top 15 feature importances and SHAP summary plots
7. Deployment — Artifacts saved with joblib/pickle; FraudDetectionSystem class for inference with latency benchmarking

 Setup

 1. Clone the repository
bash
git clone https://github.com/Hemasai2327/fraud-detection-capstone.git
cd fraud-detection-capstone


 2. Install dependencies
bash
pip install -r requirements.txt


 3. Download the dataset
Download 'creditcard.csv' from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and update the DATA_PATH in main.py to point to its location.

 4. Run the pipeline
bash
python main.py


All outputs (plots, CSVs, saved model) will be saved to the `fraud_outputs/` folder.



 Technologies

- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn
- Matplotlib, Seaborn
- SHAP, Joblib
