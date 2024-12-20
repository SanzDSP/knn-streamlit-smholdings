import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io

# Fungsi untuk memuat data dari file HDF5
def load_data(file_path):
    X_train = pd.read_hdf(file_path, key='fitur_training')
    y_train = pd.read_hdf(file_path, key='label_training')
    X_test = pd.read_hdf(file_path, key='fitur_testing')
    y_test = pd.read_hdf(file_path, key='label_testing')
    return X_train, y_train, X_test, y_test

# Fungsi untuk menyeimbangkan data menggunakan SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Fungsi untuk memilih k terbaik berdasarkan f1-score
def find_best_k(X_train, y_train):
    scores = {}
    for k in range(1, 21):  # Mencoba k dari 1 hingga 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        scores[k] = f1_score(y_train, y_pred, average='weighted')
    best_k = max(scores, key=scores.get)
    return best_k

# Fungsi untuk membuat dan melatih model
@st.cache_resource
def load_model(file_path):
    X_train, y_train, _, _ = load_data(file_path)

    # Menyeimbangkan data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_balanced, y_train_balanced = balance_data(X_train_scaled, y_train)

    # Mencari k terbaik
    best_k = find_best_k(X_train_balanced, y_train_balanced)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_balanced, y_train_balanced)

    return knn, scaler, best_k

# Judul aplikasi
st.title("Klasifikasi Saham Samsung")
st.write("Masukkan fitur saham untuk mengetahui apakah saham baik atau buruk.")

# Input fitur dari pengguna
open_price = st.number_input("Harga Open", min_value=0.0, step=10.0)
high_price = st.number_input("Harga High", min_value=0.0, step=10.0)
low_price = st.number_input("Harga Low", min_value=0.0, step=10.0)
close_price = st.number_input("Harga Close", min_value=0.0, step=10.0)
volume = st.number_input("Volume", min_value=0.0, step=10000.0)

# Path file model
file_path = "samsungholdings_classification.h5"

# Memuat model
knn_model, scaler, best_k = load_model(file_path)
st.write(f"Model menggunakan k={best_k}")

# Prediksi
if st.button("Klasifikasikan"):
    if open_price > 0 and high_price > 0 and low_price > 0 and close_price > 0 and volume > 0:
        # Data input pengguna
        user_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Close': [close_price],
            'Volume': [volume]
        })
        user_data_scaled = scaler.transform(user_data)
        probabilities = knn_model.predict_proba(user_data_scaled)
        st.write(f"Probabilitas: {probabilities}")

        # Threshold untuk prediksi
        threshold = 0.5
        prediction = 1 if probabilities[0][1] > threshold else 0
        st.success(f"Hasil klasifikasi: {'Baik' if prediction == 1 else 'Buruk'} saham")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Visualisasi distribusi data
st.subheader("Distribusi Data")
X_train, y_train, _, _ = load_data(file_path)
st.write("Distribusi Label (Sebelum Penyeimbangan):")
st.write(y_train.value_counts())

# Visualisasi grafik saham
st.subheader("Grafik Saham")
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Close'], y_train, label='Data Training', alpha=0.7)
plt.axhline(y=0.5, color='red', linestyle='--', label='Batas Kategori')
plt.xlabel('Close Price')
plt.ylabel('Kategori (baik/buruk)')
plt.legend()
buffer = io.BytesIO()
plt.savefig(buffer, format="png")
st.image(buffer)

st.write("Aplikasi ini membantu dalam memprediksi kategori saham berdasarkan model KNN.")
