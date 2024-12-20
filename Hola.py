import streamlit as st
import requests
import gdown
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Title aplikasi
st.title("Klasifikasi Saham - Samsung Holdings")
st.write("Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan saham apakah 'baik' atau 'buruk' berdasarkan input fitur.")

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Memuat model dari file .h5
        model = load_model("samsungholdings_classification.h5")
        
        # Memuat data untuk inisialisasi scaler
        X_train = pd.read_csv("fitur_training.csv")
        scaler = StandardScaler()
        scaler.fit(X_train)  # Inisialisasi scaler berdasarkan data training
        
        return model, scaler
    except FileNotFoundError:
        st.error("File model 'samsungholdings_classification.h5' atau data training tidak ditemukan.")
        return None, None

# Memuat model dan scaler
model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    # Form untuk input data saham
    st.sidebar.header("Masukkan Data Saham")
    open_price = st.sidebar.number_input("Open Price", value=1000.0, step=0.01)
    high_price = st.sidebar.number_input("High Price", value=1050.0, step=0.01)
    low_price = st.sidebar.number_input("Low Price", value=950.0, step=0.01)
    close_price = st.sidebar.number_input("Close Price", value=1020.0, step=0.01)
    volume = st.sidebar.number_input("Volume", value=500000.0, step=1000.0)

    # Membuat input array dari user
    input_features = np.array([[open_price, high_price, low_price, close_price, volume]])

    # Prediksi kelas
    if st.sidebar.button("Klasifikasi"):
        # Scaling input
        scaled_features = scaler.transform(input_features)
        
        # Melakukan prediksi menggunakan model deep learning
        prediction = model.predict(scaled_features)
        prediction_class = "baik" if prediction[0][0] > 0.5 else "buruk"  # Asumsi output berupa probabilitas

        # Menampilkan hasil prediksi
        st.subheader("Hasil Klasifikasi")
        st.write(f"Prediksi Kategori Saham: **{prediction_class.capitalize()}**")
        
        # Visualisasi
        st.subheader("Grafik Prediksi")
        st.write("Visualisasi berikut memperlihatkan prediksi kategori saham berdasarkan input:")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Baik', 'Buruk'], [prediction[0][0], 1 - prediction[0][0]], color=['green', 'red'])
        ax.set_ylabel('Probabilitas')
        ax.set_xlabel('Kategori')
        ax.set_title('Hasil Prediksi Saham')
        st.pyplot(fig)

# Menampilkan dokumentasi
st.sidebar.info("""
### Panduan
- Masukkan data fitur saham ke sidebar kiri.
- Tekan tombol **Klasifikasi** untuk memproses data.
- Hasil prediksi akan ditampilkan di tengah layar.
- Aplikasi ini menggunakan model deep learning dengan data Samsung Holdings.
""")
