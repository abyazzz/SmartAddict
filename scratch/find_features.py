import json
import os
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load model
m = load('dt2_classifier.pkl')
best_est = m.best_estimator_ if hasattr(m, 'best_estimator_') else m

# Load dataset
df_path = r'C:\Users\Muhammad Zaqi\Downloads\Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
df = pd.read_csv(df_path, keep_default_na=False)

# Strip and title object columns
for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.title()
    
# Convert numeric columns to float
numeric_cols = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time', 'addicted_label'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN
df = df.dropna()
df = df.drop_duplicates()

# Label encode categorical columns
le_dict = {}
categorical_cols = ['gender', 'stress_level', 'academic_work_impact', 'addiction_level']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Potential features
all_cols = list(df.columns)
exclude = ['transaction_id', 'user_id', 'addiction_level', 'addicted_label']
feature_pool = [c for c in all_cols if c not in exclude]

y = df['addiction_level']

original_9 = ['age', 'gender', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
remaining = [c for c in feature_pool if c not in original_9]

print("Remaining features in pool:", remaining)

scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

for rem in remaining:
    candidate_features = original_9 + [rem]
    X_candidate = df[candidate_features]
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X_candidate, y, test_size=0.2, random_state=42)
    
    for s_name, scaler in scalers.items():
        if scaler is not None:
            # Fit on train, transform test
            # Make copies to not modify raw data
            X_tr_scaled = scaler.fit_transform(X_train)
            X_te_scaled = scaler.transform(X_test)
        else:
            X_tr_scaled = X_train.values
            X_te_scaled = X_test.values
            
        try:
            y_pred = best_est.predict(X_te_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"Candidate: {rem} | Scaler: {s_name} | Test Accuracy: {acc:.4f}")
        except Exception as e:
            pass
