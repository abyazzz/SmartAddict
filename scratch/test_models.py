import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df_path = r'C:\Users\Muhammad Zaqi\Downloads\Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
df = pd.read_csv(df_path, keep_default_na=False)

# Preprocessing
for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.title()
    
# Convert numeric columns to float
numeric_cols = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN
df = df.dropna()
df = df.drop_duplicates()

# Label encode categorical columns
df['gender'] = LabelEncoder().fit_transform(df['gender'].astype(str))
df['addiction_level'] = LabelEncoder().fit_transform(df['addiction_level'].astype(str))

features = [
    'age', 'gender', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time'
]

X = df[features]
y = df['addiction_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

models = {
    'DT': load('c:/Dasar Ilmu Data/Tubes Final/SmartAddict/dt2_classifier.pkl'),
    'KNN': load('c:/Dasar Ilmu Data/Tubes Final/SmartAddict/knn2_classifier.pkl'),
    'NN': load('c:/Dasar Ilmu Data/Tubes Final/SmartAddict/nn2_classifier.pkl'),
    'SVM': load('c:/Dasar Ilmu Data/Tubes Final/SmartAddict/svm2_classifier.pkl')
}

for m_name, m in models.items():
    est = m.best_estimator_ if hasattr(m, 'best_estimator_') else m
    for s_name, scaler in scalers.items():
        if scaler is not None:
            # Scaler fit on train, transform test
            scaler.fit(X_train)
            X_te_scaled = scaler.transform(X_test)
        else:
            X_te_scaled = X_test.values
            
        try:
            y_pred = est.predict(X_te_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"Model: {m_name} | Scaler: {s_name} | Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Model: {m_name} | Scaler: {s_name} failed: {e}")
