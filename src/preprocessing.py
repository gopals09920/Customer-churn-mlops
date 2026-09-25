import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] =df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
if __name__ == "__main__":
    df_clean = load_and_clean_data('data/churn.csv')
    print("Cleaned Dataset Shape:", df_clean.shape)
    print("Columns sample:", df_clean.columns[:5])