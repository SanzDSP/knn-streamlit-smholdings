import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import h5py
import io

# Fungsi untuk memuat model klasterisasi dari file HDF5
def load_clusters(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data_scaled = h5_file['data_scaled'][:]
        clusters = h5_file['clusters'][:]
        centroids = h5_file['centroids'][:]
    return data_scaled, clusters, centroids

# Fungsi untuk memuat model KNN
def train_knn(data_scaled, clusters):
    knn = KNeighborsClassifier(n_neighbors=4)  # Menggunakan 4 klaster
    knn.fit(data_scaled, clusters)
    return knn

# Judul aplikasi
st.title("Klasifikasi dan Klasterisasi Saham Samsung")
st.write("Masukkan fitur saham untuk mengetahui klaster saham Anda.")

# Input fitur dari pengguna
open_price = st.number_input("Harga Open", min_value=0.0, step=10.0)
high_price = st.number_input("Harga High", min_value=0.0, step=10.0)
low_price = st.number_input("Harga Low", min_value=0.0, step=10.0)
close_price = st.number_input("Harga Close", min_value=0.0, step=10.0)
volume = st.number_input("Volume", min_value=0.0, step=10000.0)

# Memuat model klasterisasi
file_path = "samsung_clusters.h5"
data_scaled, clusters, centroids = load_clusters(file_path)

# Latih model KNN untuk klasifikasi
knn = train_knn(data_scaled, clusters)

# Standarisasi data input
scaler = StandardScaler()
scaler.fit(data_scaled)

# Prediksi klaster berdasarkan input pengguna
if st.button("Klasifikasikan"):
    if open_price > 0 and high_price > 0 and low_price > 0 and close_price > 0 and volume > 0:
        user_data = np.array([[open_price, high_price, low_price, close_price, volume]])
        user_data_scaled = scaler.transform(user_data)
        cluster = knn.predict(user_data_scaled)[0]

        # Tampilkan hasil klaster
        cluster_labels = {
            0: "High Volume and Price",
            1: "Moderate Activity",
            2: "Low Price, Low Volume",
            3: "Very Low Activity"
        }
        st.success(f"Hasil klasifikasi: {cluster_labels[cluster]} (Cluster {cluster})")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Visualisasi klasterisasi
st.subheader("Visualisasi Klasterisasi")
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.scatter(
        data_scaled[clusters == i, 0],  # Fitur pertama (Open Price)
        data_scaled[clusters == i, 3],  # Fitur keempat (Close Price)
        label=f"Cluster {i}"
    )
plt.scatter(
    centroids[:, 0],
    centroids[:, 3],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)
plt.title("Klasterisasi Saham Samsung")
plt.xlabel("Scaled Open Price")
plt.ylabel("Scaled Close Price")
plt.legend()
st.pyplot(plt)

st.write("Aplikasi ini membantu dalam memprediksi kategori saham berdasarkan model KNN.")


