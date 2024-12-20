import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import io

# Fungsi untuk memuat data dari file HDF5
def load_data(file_path):
    X_train = pd.read_hdf(file_path, key='fitur_training')
    y_train = pd.read_hdf(file_path, key='label_training')
    X_test = pd.read_hdf(file_path, key='fitur_testing')
    y_test = pd.read_hdf(file_path, key='label_testing')
    return X_train, y_train, X_test, y_test

# Fungsi untuk membuat dan melatih model
@st.cache_resource
def load_model(file_path):
    X_train, y_train, _, _ = load_data(file_path)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)
    return knn, scaler

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
knn_model, scaler = load_model(file_path)

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
        prediction = knn_model.predict(user_data_scaled)

        st.success(f"Hasil klasifikasi: {prediction[0].capitalize()} saham")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Visualisasi data
st.subheader("Grafik Saham")
X_train, y_train, X_test, y_test = load_data(file_path)
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Close'], y_train, label='Data Training', alpha=0.7)
plt.scatter(X_test['Close'], y_test, label='Data Testing', alpha=0.7)
plt.axhline(y=0.5, color='red', linestyle='--', label='Batas Kategori')
plt.xlabel('Close Price')
plt.ylabel('Kategori (baik/buruk)')
plt.legend()
buffer = io.BytesIO()
plt.savefig(buffer, format="png")
st.image(buffer)

st.write("Aplikasi ini membantu dalam memprediksi kategori saham berdasarkan model KNN.")
