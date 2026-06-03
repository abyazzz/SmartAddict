#!/usr/bin/env python
# coding: utf-8

# Auto-generated from Tubes_FIX.ipynb
# Generated at: 2026-06-04 03:08:11.983850

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import datetime as _datetime_module

# Set output directory
output_model_dir = r'D:/Tel-U/Tugas/semester 4/Dasar Ilmu Data/TUBES-DASILDAT-main/model/model_FINAL_20260604_030811'
os.makedirs(output_model_dir, exist_ok=True)

# Set dataset path
DATASET_PATH = r'D:/Tel-U/Tugas/semester 4/Dasar Ilmu Data/TUBES-DASILDAT-main/dataset-notebook/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'

# Helper for display() function (used in notebook)
def display(obj):
    print(obj)



# Cell 0
try:
    import json
    import os
    from datetime import datetime as _dt
    
    RETRAIN_JOB_ID = os.environ.get('RETRAIN_JOB_ID') or globals().get('job_id', 'unknown')
    RETRAIN_STATUS_FILE = os.environ.get('RETRAIN_STATUS_FILE') or globals().get('status_file_path') or os.path.join(
        'instance',
        'retrain_statuses',
        f"{RETRAIN_JOB_ID}.json",
    )
    
    RETRAIN_STEP_PLAN = [
        'Load library',
        'Load dataset',
        'Label encoding kolom kategorikal',
        'Visualisasi sederhana EDA',
        'Ambil feature dan label',
        'Split train/test 80/20',
        'SMOTE',
        'Feature scaling',
        'PCA',
        'SVM',
        'HPO SVM',
        'Evaluation',
        'k-NN classifier',
        'HPO dan evaluasi k-NN',
        'Decision Tree',
        'HPO Decision Tree',
        'Neural Network',
        'HPO Neural Network',
        'Tabel perbandingan sebelum HPO',
        'Tabel perbandingan setelah HPO',
        'Confusion matrix Decision Tree',
        'Deploy',
    ]
    
    
    def _load_status():
        try:
            with open(RETRAIN_STATUS_FILE, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception:
            return {
                'job_id': RETRAIN_JOB_ID,
                'status': 'running',
                'progress': 0,
                'current_step': None,
                'steps': [{'name': name, 'status': 'pending'} for name in RETRAIN_STEP_PLAN],
                'logs': [],
            }
    
    
    def _save_status(payload):
        os.makedirs(os.path.dirname(RETRAIN_STATUS_FILE), exist_ok=True)
        with open(RETRAIN_STATUS_FILE, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    
    
    def checkpoint(step_name, finalize=False, message=None):
        payload = _load_status()
        steps = payload.get('steps') or [{'name': name, 'status': 'pending'} for name in RETRAIN_STEP_PLAN]
        previous = payload.get('current_step')
    
        for step in steps:
            if previous and step.get('name') == previous and step.get('status') == 'running':
                step['status'] = 'done'
            if step.get('name') == step_name:
                step['status'] = 'done' if finalize else 'running'
    
        payload['steps'] = steps
        payload['current_step'] = None if finalize else step_name
        payload['status'] = 'success' if finalize else 'running'
        payload['progress'] = round((sum(1 for step in steps if step.get('status') == 'done') / max(len(steps), 1)) * 100, 2)
        payload['finished_at'] = _dt.utcnow().isoformat() + 'Z' if finalize else payload.get('finished_at')
        logs = payload.get('logs', [])
        logs.append({
            'ts': _dt.utcnow().isoformat() + 'Z',
            'level': 'INFO',
            'message': message or (f'RETRAIN SELESAI' if finalize else f'Sedang menjalankan: {step_name}')
        })
        payload['logs'] = logs
        _save_status(payload)
except: pass


# Cell 1
output_model_dir = 'model/model_default'


# Cell 4
try:
    checkpoint('Load library')
except: pass


# Cell 7
try:
    checkpoint('Load dataset')
except: pass


# Cell 8
# Load data dari file clean_dataset.csv
import os
file_path = DATASET_PATH

df = pd.read_csv(file_path)

# Convert non-numeric columns to object dtype to ensure compatibility with pandas 3.x
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].astype('object')

print(f"Bentuk dataset: {df.shape}")


# Cell 9
print(df['addiction_level'].unique())
print(df['addiction_level'].value_counts(dropna=False))


# Cell 10
df.head()


# Cell 12
try:
    checkpoint('Label encoding kolom kategorikal')
except: pass


# Cell 13
from sklearn.preprocessing import LabelEncoder

# Hapus missing value
df = df.dropna()

# Hapus spasi berlebih di semua kolom object/string
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

# Samakan format huruf
# Contoh: severe -> Severe
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.title()

# Hapus data duplikat
df = df.drop_duplicates()


# Inisialisasi LabelEncoder
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(f"Bentuk dataset setelah label encoding: {df.head}")


# Cell 14
df.head()


# Cell 16
try:
    checkpoint('Visualisasi sederhana EDA')
except: pass


# Cell 17
import matplotlib.pyplot as plt
import seaborn as sns # Make sure seaborn is imported

# Mempersiapkan data untuk visualisasi addiction_level
# Karena addiction_level sudah di-encode menjadi 0, 1, 2,
# dan berdasarkan df.head() sebelumnya, urutannya adalah Mild, Moderate, Severe.
labels = ['Mild', 'Moderate', 'Severe']
ticks = range(len(labels))

print(df['addiction_level'].value_counts())

# Membuat barchart dari value_counts() kolom 'addiction_level'
plt.figure(figsize=(8, 5))
addiction_counts = df['addiction_level'].value_counts()
sns.barplot(x=addiction_counts.index, y=addiction_counts.values, palette='viridis')

plt.title('Distribusi Level Adiksi') # Corrected title
plt.xlabel('Level Adiksi (0 = Mild, 1 = Moderate, 2 = Severe)')
plt.ylabel('Jumlah Sampel')
plt.xticks(ticks=ticks, labels=labels)
plt.show()


# Cell 19
try:
    checkpoint('Ambil feature dan label')
except: pass


# Cell 20
# Ambil semua kolom
X = df.loc[:,'age':'weekend_screen_time']

# Ambil kolom 'addiction_level' sebagai target (label)
y = df['addiction_level']


print("Fitur (X) head:")
display(X.head())
print("Label (y) head:")
display(y.head())


# Cell 22
try:
    checkpoint('Split train/test 80/20')
except: pass


# Cell 23
# Membagi data jadi data training dan data testing
# X = fitur (input), y = label (output)
# 20% data jadi data testing, 80% jadi training

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)


# Cell 25
try:
    checkpoint('SMOTE')
except: pass


# Cell 26
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

# Inisialisasi SMOTE
smote = SMOTE(random_state=42)

# Terapkan SMOTE pada data training
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Bentuk X_train sebelum SMOTE: {X_train.shape}")
print(f"Bentuk y_train sebelum SMOTE: {y_train.shape}")
print(f"Jumlah kelas di y_train sebelum SMOTE:\n{y_train.value_counts()}")

print(f"\nBentuk X_train setelah SMOTE: {X_train_smote.shape}")
print(f"Bentuk y_train setelah SMOTE: {y_train_smote.shape}")
print(f"Jumlah kelas di y_train setelah SMOTE:\n{y_train_smote.value_counts()}")

# Perbarui X_train dan y_train dengan hasil SMOTE
X_train = X_train_smote
y_train = y_train_smote


# Cell 27
import matplotlib.pyplot as plt
import seaborn as sns

# Mempersiapkan data untuk visualisasi y_train_smote
# Karena addiction_level sudah di-encode menjadi 0, 1, 2,
# dan berdasarkan df.head() sebelumnya, urutannya adalah Mild, Moderate, Severe.
labels = ['Mild', 'Moderate', 'Severe']
ticks = range(len(labels))

print(y_train_smote.value_counts())

# Membuat barchart dari value_counts() kolom y_train_smote
plt.figure(figsize=(8, 5))
smote_counts = y_train_smote.value_counts().sort_index()
sns.barplot(x=smote_counts.index, y=smote_counts.values, palette='viridis')

plt.title('Distribusi Level Adiksi Setelah SMOTE')
plt.xlabel('Level Adiksi (0 = Mild, 1 = Moderate, 2 = Severe)')
plt.ylabel('Jumlah Sampel')
plt.xticks(ticks=ticks, labels=labels)
plt.show()


# Cell 29
try:
    checkpoint('Feature scaling')
except: pass


# Cell 30
scaler = StandardScaler()
# Fit scaler pada X_train dan transform X_train
X_train = scaler.fit_transform(X_train);
# Transform X_test dengan scaler yang sama
X_test = scaler.transform(X_test)
print(f"Bentuk X_train setelah Standard Scaling: {X_train.shape}")
print(f"Bentuk X_test setelah Standard Scaling: {X_test.shape}")


# Cell 32
try:
    checkpoint('HPO SVM')
except: pass


# Cell 33
from sklearn.svm import SVC

# isi jawaban parameter
param_grid = [
    {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': [0.01, 0.001, 0.0001]}
]

# Inisialisasi SVC dengan probability=True dan random_state=42 seperti yang Anda sebutkan
svc_estimator = SVC(probability=True, random_state=42)

#tolong perhatikan parameter scoring dan cv
classifier = GridSearchCV(SVC(), param_grid, scoring='recall_macro', cv=5, refit = True, verbose = 3)


classifier.fit(X_train, y_train)


print('Parameter terbaik:',classifier.best_params_)

# print how our model looks after hyper-parameter tuning
print(classifier.best_estimator_)


# Cell 35
try:
    checkpoint('Evaluation')
except: pass


# Cell 36
from sklearn.metrics import accuracy_score, classification_report

predictions = classifier.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('\nClassification Report:\n', classification_report(y_test, predictions))


# Cell 37
# # Regression plot (commented out as it's not suitable for classification)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# ax.text(1, 9.5,'$R^2=$'+str(round(r2_score(y_test, predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')
# ax.text(1, 9,'$MSE=$'+str(round(mean_squared_error(y_test, predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')

# ax.set_xlim(xmin=1)
# ax.set_ylim(ymin=1)
# ax.set_xlim(xmax=10)
# ax.set_ylim(ymax=10)

# ax.set_xlabel('Actual Value', fontsize=14)
# ax.set_ylabel('Predicted Value', fontsize=14)
# ax.scatter(y_test, predictions, s=100, c=y_test, cmap='viridis')

# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]

# ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.grid(True, which='both')

# xvalue = np.linspace(1,10,10)
# print(xvalue)
# lsigma = ax.fill_between(xvalue, xvalue+1, xvalue-1, color='blue', alpha=0.3)

# plt.show()


# Cell 39
try:
    checkpoint('HPO dan evaluasi k-NN')
except: pass


# Cell 40
from sklearn.neighbors import KNeighborsClassifier

# Definisikan parameter grid untuk k-NN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski'] # 'chebyshev', 'wminkowski', 'seuclidean', 'mahalanobis' could be problematic with some data/versions
}

# Inisialisasi GridSearchCV untuk k-NN
knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='recall_macro', cv=5, refit=True, verbose=3)

# Latih model
knn_classifier.fit(X_train, y_train)

print('Parameter terbaik k-NN:', knn_classifier.best_params_)
print('Estimator terbaik k-NN:', knn_classifier.best_estimator_)


# Cell 41
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model k-NN terbaik
knn_predictions = knn_classifier.predict(X_test)

print('Accuracy k-NN:', accuracy_score(y_test, knn_predictions))
print('\nClassification Report k-NN:\n', classification_report(y_test, knn_predictions))


# Cell 43
try:
    checkpoint('HPO Decision Tree')
except: pass


# Cell 44
from sklearn.tree import DecisionTreeClassifier

# Definisikan parameter grid untuk Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi GridSearchCV untuk Decision Tree
dt_classifier = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid_dt, scoring='recall_macro', cv=5, refit=True, verbose=3)

# Latih model
dt_classifier.fit(X_train, y_train)

print('Parameter terbaik Decision Tree:', dt_classifier.best_params_)
print('Estimator terbaik Decision Tree:', dt_classifier.best_estimator_)


# Cell 45
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model Decision Tree terbaik
dt_predictions = dt_classifier.predict(X_test)

print('Accuracy Decision Tree:', accuracy_score(y_test, dt_predictions))
print('\nClassification Report Decision Tree:\n', classification_report(y_test, dt_predictions))


# Cell 47
try:
    checkpoint('HPO Neural Network')
except: pass


# Cell 48
# isi jawaban di sini
from sklearn.neural_network import MLPClassifier

# defining parameter range
param_grid_nn = [
    {'hidden_layer_sizes': [(10,), (15, 10)], 'max_iter': [2000], 'activation': ['relu', 'tanh', 'logistic'], 'solver': ['adam']}
]

#tolong perhatikan parameter scoring dan cv
nn_classifier = GridSearchCV(MLPClassifier(random_state=42), param_grid_nn, scoring='recall_macro', cv=5, refit=True, verbose=3)

# fitting the model for grid search
nn_classifier.fit(X_train, y_train)

# print best parameter after tuning
print('parameter terbaik:', nn_classifier.best_params_)

# print how our model looks after hyper-parameter tuning
print(nn_classifier.best_estimator_)


# Cell 49
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model Neural Network terbaik
mlp_predictions = nn_classifier.predict(X_test)

print('Accuracy Neural Network:', accuracy_score(y_test, mlp_predictions))
print('\nClassification Report Neural Network:\n', classification_report(y_test, mlp_predictions))


# Cell 51
try:
    checkpoint('Tabel perbandingan setelah HPO')
except: pass


# Cell 52
import pandas as pd

# Collect the accuracy scores from each model
# SVM accuracy (from DsVMW2BttcU4 which used 'predictions')
svm_accuracy = accuracy_score(y_test, predictions)

# k-NN accuracy (from 579e6d3b which used 'knn_predictions')
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Decision Tree accuracy (from 8d29bedf which used 'dt_predictions')
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Neural Network accuracy (from b61e40b5 which used 'mlp_predictions')
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

# Create a DataFrame to compare the models
results_df = pd.DataFrame({
    'Model': ['SVM', 'k-NN', 'Decision Tree', 'Neural Network'],
    'Accuracy': [svm_accuracy, knn_accuracy, dt_accuracy, mlp_accuracy]
})

print("\n===== MODEL COMPARISON ===\n")
print(results_df.to_string(index=False))
print("\n")
display(results_df)


# Cell 54
try:
    checkpoint('Confusion matrix Decision Tree')
except: pass


# Cell 55
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion matrix untuk model Decision Tree (performa terbaik)
cm = confusion_matrix(y_test, dt_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Tree')
plt.show()


# Cell 57
try:
    checkpoint('Deploy')
except: pass


# Cell 58
import joblib
import os
import json
import datetime

# Check output_model_dir
output_model_dir = os.environ.get('OUTPUT_MODEL_DIR') or globals().get('output_model_dir') or '/content/drive/My Drive/dataset_tubes/model2'

os.makedirs(output_model_dir, exist_ok=True)

# Simpan model yang sudah terlatih saja, bukan seluruh GridSearchCV object
svm_artifact = classifier.best_estimator_ if hasattr(classifier, 'best_estimator_') else classifier
dt_artifact = dt_classifier.best_estimator_ if hasattr(dt_classifier, 'best_estimator_') else dt_classifier
knn_artifact = knn_classifier.best_estimator_ if hasattr(knn_classifier, 'best_estimator_') else knn_classifier
nn_artifact = nn_classifier.best_estimator_ if hasattr(nn_classifier, 'best_estimator_') else nn_classifier
joblib.dump(svm_artifact, os.path.join(output_model_dir, 'svm2_classifier.pkl'))
joblib.dump(dt_artifact, os.path.join(output_model_dir, 'dt2_classifier.pkl'))
joblib.dump(knn_artifact, os.path.join(output_model_dir, 'knn2_classifier.pkl'))
joblib.dump(nn_artifact, os.path.join(output_model_dir, 'nn2_classifier.pkl'))
joblib.dump(scaler, os.path.join(output_model_dir, 'scaler.pkl'))

# Save metrics.json
timestamp = os.path.basename(output_model_dir).replace("model_", "")
if not timestamp or timestamp == "model2":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

metrics_data = {
    "dt": float(dt_accuracy),
    "knn": float(knn_accuracy),
    "nn": float(mlp_accuracy),
    "svm": float(svm_accuracy),
    "timestamp": timestamp
}

metrics_path = os.path.join(output_model_dir, 'metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

print(f"Model and metrics saved successfully to {output_model_dir}")


# Cell 59
try:
    checkpoint('Deploy', finalize=True, message='RETRAIN SELESAI')
except: pass
