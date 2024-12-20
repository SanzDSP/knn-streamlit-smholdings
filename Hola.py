import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title aplikasi
st.title("Klasifikasi Saham - Samsung Holdings")
st.write("Aplikasi ini menggunakan algoritma K-Nearest Neighbors (KNN) untuk mengklasifikasikan saham apakah 'baik' atau 'buruk' berdasarkan input fitur.")

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_model():
    # Memuat dataset model dari file .h5
    X_train = pd.read_hdf('samsungholdings_classification.h5', key='fitur_training')
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler berdasarkan data training
    model = joblib.load('knn_model.pkl')  # Load model yang telah disimpan
    return model, scaler

# Memuat model dan scaler
model, scaler = load_model()

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
    prediction = model.predict(scaled_features)

    # Menampilkan hasil prediksi
    st.subheader("Hasil Klasifikasi")
    st.write(f"Prediksi Kategori Saham: **{prediction[0].capitalize()}**")
    
    # Visualisasi
    st.subheader("Grafik Prediksi")
    st.write("Visualisasi berikut memperlihatkan prediksi kategori saham berdasarkan input:")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Baik', 'Buruk'], [1 if prediction[0] == 'baik' else 0, 1 if prediction[0] == 'buruk' else 0], color=['green', 'red'])
    ax.set_ylabel('Prediksi')
    ax.set_xlabel('Kategori')
    ax.set_title('Hasil Prediksi Saham')
    st.pyplot(fig)

# Menampilkan dokumentasi
st.sidebar.info("""
### Panduan
- Masukkan data fitur saham ke sidebar kiri.
- Tekan tombol **Klasifikasi** untuk memproses data.
- Hasil prediksi akan ditampilkan di tengah layar.
- Aplikasi ini menggunakan algoritma KNN dengan data Samsung Holdings.
""")
