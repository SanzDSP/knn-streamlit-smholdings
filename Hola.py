import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import io

# Fungsi untuk memuat data dari file HDF5
def load_data(file_path):
    X_train = pd.read_hdf(file_path, key='fitur_training')
    y_train = pd.read_hdf(file_path, key='label_training')
    X_test = pd.read_hdf(file_path, key='fitur_testing')
    y_test = pd.read_hdf(file_path, key='label_testing')
    return X_train, y_train, X_test, y_test

# Fungsi untuk memilih k terbaik
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Menyeimbangkan data untuk mengatasi bias
    y_train_balanced = pd.Series(y_train).replace({0: 1, 1: 0})
    best_k = find_best_k(X_train_scaled, y_train_balanced)  # Cari k terbaik
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train_balanced)
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
        prediction = knn_model.predict(user_data_scaled)

        st.success(f"Hasil klasifikasi: {'Baik' if prediction[0] == 1 else 'Buruk'} saham")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Visualisasi data
st.subheader("Grafik Keuntungan dan Kerugian Saham")
X_train, y_train, X_test, y_test = load_data(file_path)

# Hitung jumlah keuntungan dan kerugian
keuntungan = (y_train == 1).sum()
kerugian = (y_train == 0).sum()

# Visualisasi bar plot
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Keuntungan', 'Kerugian'], [keuntungan, kerugian], color=['green', 'red'])
ax.bar_label(bars, fmt='%d')

# Tambahkan bingkai sesuai kategori
ax.patches[0].set_edgecolor('green')
ax.patches[0].set_linewidth(2)
ax.patches[1].set_edgecolor('red')
ax.patches[1].set_linewidth(2)

# Tampilkan grafik
plt.title('Distribusi Keuntungan dan Kerugian')
buffer = io.BytesIO()
plt.savefig(buffer, format="png")
st.image(buffer)

st.write("Aplikasi ini membantu dalam memprediksi kategori saham berdasarkan model KNN.")


