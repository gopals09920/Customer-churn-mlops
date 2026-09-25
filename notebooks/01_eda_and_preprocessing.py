import pandas as pd
import numpy as np
df = pd.read_csv('data/churn.csv')
print("--- Data Summary ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Churn Target Count ---")
print(df['Churn'].value_counts())