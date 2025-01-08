import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import h5py
import io

# Fungsi untuk memuat model klasterisasi dari file HDF5
def load_clusters(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data_scaled = h5_file['data_scaled'][:]
        clusters = h5_file['clusters'][:]
        centroids = h5_file['centroids'][:]
    return data_scaled, clusters, centroids

# Fungsi untuk melatih model KNN
def train_knn(data_scaled, clusters):
    knn = KNeighborsClassifier(n_neighbors=4)  # Menggunakan 4 klaster
    knn.fit(data_scaled, clusters)
    return knn

# Fungsi untuk menghasilkan data simulasi keuntungan/kerugian
def generate_simulation_data(cluster, days=30):
    np.random.seed(cluster)
    base_value = 100 + cluster * 20  # Nilai awal berdasarkan klaster
    growth = np.cumsum(np.random.uniform(-5, 15, days))  # Menambah variasi pada perubahan
    return base_value + growth

# Judul aplikasi
st.title("Klasifikasi dan Visualisasi Saham Samsung")
st.write("Masukkan fitur saham untuk mengetahui klaster saham Anda dan simulasi investasi.")

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

# Standarisasi data input dengan scaler yang sama yang digunakan pada data pelatihan
scaler = StandardScaler()
scaler.fit(data_scaled)  # Fit dengan data pelatihan yang sudah distandarisasi

# Prediksi klaster berdasarkan input pengguna
if st.button("Klasifikasikan"):
    if open_price > 0 and high_price > 0 and low_price > 0 and close_price > 0 and volume > 0:
        user_data = np.array([[open_price, high_price, low_price, close_price, volume]])
        user_data_scaled = scaler.transform(user_data)  # Transformasi input pengguna

        cluster = knn.predict(user_data_scaled)[0]

        # Tampilkan hasil klaster
        cluster_labels = {
            0: "High Volume and Price",
            1: "Moderate Activity",
            2: "Low Price, Low Volume",
            3: "Very Low Activity"
        }
        st.success(f"Hasil klasifikasi: {cluster_labels[cluster]} (Cluster {cluster})")

        # Simulasi data keuntungan/kerugian
        st.subheader("Simulasi Pertumbuhan Investasi")
        days = 30  # Simulasi untuk 30 hari
        simulated_growth = generate_simulation_data(cluster, days)
        time = np.arange(1, days + 1)

        # Visualisasi interaktif dengan Plotly
        fig = go.Figure()

        # Menambahkan garis simulasi
        fig.add_trace(go.Scatter3d(
            x=time,
            y=simulated_growth,
            z=[cluster] * days,
            mode='lines+markers',
            marker=dict(size=5, color=simulated_growth, colorscale='Viridis', opacity=0.8),
            line=dict(color='blue', width=2),
            name=f"Cluster {cluster}"
        ))

        # Pengaturan layout
        fig.update_layout(
            title="Grafik 3D Pertumbuhan Investasi",
            scene=dict(
                xaxis_title="Hari",
                yaxis_title="Nilai Investasi",
                zaxis_title="Klaster",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # Tampilkan grafik
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Keterangan tambahan
st.write("Aplikasi ini membantu memprediksi klaster saham dan memberikan simulasi investasi berdasarkan klaster.")


