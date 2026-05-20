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
df_path = r'C:\Users\Muhammad Zaqi\Downloads\smart_addict_dataset.csv'
df = pd.read_csv(df_path, keep_default_na=False)

# Preprocessing from Tubes_FIX.ipynb
# Strip and title object columns
for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.title()
    
# Convert numeric columns to float
numeric_cols = [
    'jam_smartphone_per_hari', 'jam_media_sosial_per_hari', 'frekuensi_kehilangan_tidur',
    'frekuensi_cek_hp_per_jam', 'merasa_cemas_tanpa_hp', 'gangguan_aktivitas_harian',
    'penggunaan_saat_makan', 'penggunaan_sebelum_tidur', 'pernah_mencoba_berhenti', 'usia'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN
df = df.dropna()
df = df.drop_duplicates()

# Label encode categorical columns
le_dict = {}
categorical_cols = ['level_kecanduan']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Features and target
feature_cols = [c for c in df.columns if c != 'level_kecanduan']
print("Features:", feature_cols)

X = df[feature_cols]
y = df['level_kecanduan']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

for s_name, scaler in scalers.items():
    if scaler is not None:
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)
    else:
        X_tr_scaled = X_train.values
        X_te_scaled = X_test.values
        
    try:
        y_pred = best_est.predict(X_te_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"Scaler: {s_name} | Test Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Scaler: {s_name} failed: {e}")
