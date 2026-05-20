# CELL 0
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression # Import untuk seleksi fitur
from sklearn.metrics import mean_squared_error, r2_score


from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd /content/drive/My Drive/dataset_tubes
!ls

# CELL 1
# Load data dari file clean_dataset.csv
file_path = '/content/drive/MyDrive/dataset_tubes/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
df = pd.read_csv(file_path)

print(f"Bentuk dataset: {df.shape}")

# CELL 2
print(df['addiction_level'].unique())
print(df['addiction_level'].value_counts(dropna=False))

# CELL 3
df.head()

# CELL 4
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

# CELL 5
df.head()

# CELL 6
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

# CELL 7
# Ambil semua kolom
X = df.loc[:, ['age','gender','daily_screen_time_hours','social_media_hours','gaming_hours','sleep_hours','notifications_per_day','app_opens_per_day','weekend_screen_time']]

# Ambil kolom 'addiction_level' sebagai target (label)
y = df['addiction_level']


print("Fitur (X) head:")
display(X.head())
print("Label (y) head:")
display(y.head())

# CELL 8
# Membagi data jadi data training dan data testing
# X = fitur (input), y = label (output)
# 30% data jadi data testing, 70% jadi training

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# CELL 9
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

# CELL 10
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

# CELL 11
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
print(X_train)
print(f"Bentuk X_train setelah scaling: {X_train.shape}")

# CELL 12
# from sklearn.decomposition import PCA

# # Inisialisasi PCA. Kita bisa memilih jumlah komponen atau membiarkan PCA menentukannya
# # Misalnya, kita bisa mencoba 2 komponen utama untuk visualisasi, atau 'None' untuk melihat varians
# # Di sini, kita akan mencoba mengurangi menjadi 2 komponen untuk demonstrasi.
# pca = PCA(n_components=2)

# # Terapkan PCA pada data training dan testing
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# print(f"Bentuk X_train setelah PCA: {X_train_pca.shape}")
# print(f"Bentuk X_test setelah PCA: {X_test_pca.shape}")

# # Jelaskan variansi yang dijelaskan oleh komponen utama
# print(f"Variansi yang dijelaskan oleh masing-masing komponen utama: {pca.explained_variance_ratio_}")
# print(f"Total variansi yang dijelaskan oleh {pca.n_components_} komponen utama: {sum(pca.explained_variance_ratio_):.2f}")

# # Tampilkan data setelah PCA (beberapa baris pertama)
# print("\nX_train_pca head:")
# # display(pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)]).head())


# CELL 13
from sklearn.svm import SVC
from sklearn.metrics import classification_report
model_default = SVC(kernel='rbf', C=1000, gamma=0.01, random_state=0)
model_default.fit(X_train, y_train)
pred_default_svm = model_default.predict(X_test)
print(classification_report(y_test, pred_default_svm))

# CELL 14
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

# CELL 15
from sklearn.metrics import accuracy_score, classification_report

predictions = classifier.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('\nClassification Report:\n', classification_report(y_test, predictions, zero_division=0))

# CELL 16
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

# CELL 17
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

knn_before =  KNeighborsClassifier(n_neighbors=5)
knn_before.fit(X_train, y_train)
y_pred = knn_before.predict(X_test)
print(classification_report(y_test, y_pred))

# CELL 18
from sklearn.neighbors import KNeighborsClassifier

# Definisikan parameter grid untuk k-NN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski'] # 'chebyshev', 'wminkowski', 'seuclidean', 'mahalanobis' could be problematic with some data/versions
}

# Inisialisasi GridSearchCV untuk k-NN
knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='accuracy', cv=5, refit=True, verbose=3)

# Latih model
knn_classifier.fit(X_train, y_train)

print('Parameter terbaik k-NN:', knn_classifier.best_params_)
print('Estimator terbaik k-NN:', knn_classifier.best_estimator_)

# CELL 19
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model k-NN terbaik
knn_predictions = knn_classifier.predict(X_test)

print('Accuracy k-NN:', accuracy_score(y_test, knn_predictions))
print('\nClassification Report k-NN:\n', classification_report(y_test, knn_predictions))

# CELL 20
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Inisialisasi model Decision Tree dengan parameter default
dt_before_hpo = DecisionTreeClassifier(random_state=0)

# Latih model pada data training
dt_before_hpo.fit(X_train, y_train)

# Lakukan prediksi pada data testing
y_pred_dt = dt_before_hpo.predict(X_test)

# Cetak laporan klasifikasi
print('Classification Report (Decision Tree before HPO):')
print(classification_report(y_test, y_pred_dt))

# CELL 21
from sklearn.tree import DecisionTreeClassifier

# Definisikan parameter grid untuk Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi GridSearchCV untuk Decision Tree
dt_classifier = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid_dt, scoring='accuracy', cv=5, refit=True, verbose=3)

# Latih model
dt_classifier.fit(X_train, y_train)

print('Parameter terbaik Decision Tree:', dt_classifier.best_params_)
print('Estimator terbaik Decision Tree:', dt_classifier.best_estimator_)

# CELL 22
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model Decision Tree terbaik
dt_predictions = dt_classifier.predict(X_test)

print('Accuracy Decision Tree:', accuracy_score(y_test, dt_predictions))
print('\nClassification Report Decision Tree:\n', classification_report(y_test, dt_predictions))

# CELL 23
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Model Neural Network tanpa HPO
model_default = MLPClassifier(
    random_state=42,
    max_iter=400
)

# Training
model_default.fit(X_train, y_train)

# Prediksi
pred_default_nn = model_default.predict(X_test)

# Evaluasi
print(classification_report(y_test, pred_default_nn))

# CELL 24
# isi jawaban di sini
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV

# defining parameter range
param_grid = [
    {'hidden_layer_sizes':[(10,),(15,10)], 'max_iter':[2000], 'activation': ['relu','tanh','logistic'], 'solver': ['adam']}
 ]

#tolong perhatikan parameter scoring dan cv
nn_classifier = GridSearchCV(MLPClassifier(), param_grid, scoring='recall_macro', cv=5, refit = True, verbose = 3)

# fitting the model for grid search
nn_classifier.fit(X_train, y_train)

# print best parameter after tuning
print('parameter terbaik:',classifier.best_params_)

# print how our model looks after hyper-parameter tuning
print(classifier.best_estimator_)

# CELL 25
from sklearn.metrics import accuracy_score, classification_report

# Prediksi dengan model Neural Network terbaik
mlp_predictions = classifier.predict(X_test)

print('Accuracy Neural Network:', accuracy_score(y_test, mlp_predictions))
print('\nClassification Report Neural Network:\n', classification_report(y_test, mlp_predictions))

# CELL 26
import pandas as pd
from sklearn.metrics import accuracy_score

# Accuracy for SVM before HPO (from pjgPhh7Besq8)
svm_accuracy_before_hpo = accuracy_score(y_test, pred_default_svm)

# Accuracy for k-NN before HPO (from D8Dd2AZIneA2)
knn_accuracy_before_hpo = accuracy_score(y_test, y_pred)

# Accuracy for Decision Tree before HPO (from 9HantjUr_Pcx)
dt_accuracy_before_hpo = accuracy_score(y_test, y_pred_dt)

# Accuracy for Neural Network before HPO (from SiSHiBJA_XlZ)
mlp_accuracy_before_hpo = accuracy_score(y_test, pred_default_nn)

# Create a DataFrame to compare the models before HPO
results_before_hpo_df = pd.DataFrame({
    'Model': ['SVM', 'k-NN', 'Decision Tree', 'Neural Network'],
    'Accuracy Before HPO': [svm_accuracy_before_hpo, knn_accuracy_before_hpo, dt_accuracy_before_hpo, mlp_accuracy_before_hpo]
})

print("\n===== MODEL COMPARISON (Before HPO) ===\n")
print(results_before_hpo_df.to_string(index=False))
print("\n")
display(results_before_hpo_df)

# CELL 27
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

# CELL 28
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion matrix untuk model Decision Tree (performa terbaik)
cm = confusion_matrix(y_test, dt_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# CELL 29
import joblib
import os

# Pastikan folder model di dalam Google Drive
os.makedirs('/content/drive/My Drive/dataset_tubes', exist_ok=True)

# Simpan model dan preprocessor ke Google Drive
joblib.dump(classifier, '/content/drive/My Drive/dataset_tubes/svm2_classifier.pkl')
joblib.dump(dt_classifier, '/content/drive/My Drive/dataset_tubes/dt2_classifier.pkl')
joblib.dump(knn_classifier, '/content/drive/My Drive/dataset_tubes/knn2_classifier.pkl')
joblib.dump(nn_classifier, '/content/drive/My Drive/dataset_tubes/nn2_classifier.pkl')

