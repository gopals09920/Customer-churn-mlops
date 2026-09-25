import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from preprocessing import load_and_clean_data
def train_model():
    print("1. Loading and cleaning data...")
    df = load_and_clean_data('data/churn.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("2. Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   Original Traning set size: {X_train.shape[0]}")
    print(f"   Balanced Traning set size (after SMOTE): {X_train_res.shape[0]}")
    print("3. Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_res, y_train_res)
    print("4. Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    aus_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n===========================================================")
    print(f"ROC AUC Score: {aus_score* 100:.2f}%")
    print("===========================================================")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/xgb_churn_model.pkl')
    joblib.dump(list(X.columns), 'models/features_columns.pkl')
    print("Model saved to 'models/xgb_churn_model.pkl'")
if __name__ == "__main__":
    train_model()