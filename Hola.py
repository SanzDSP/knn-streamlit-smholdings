import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
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

# Fungsi untuk menyeimbangkan data
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

# Fungsi untuk memilih k terbaik
def find_best_k(X_train, y_train):
    scores = {}
    for k in range(1, 21):  # Mencoba k dari 1 hingga 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        scores[k] = f1_score(y_train, y_pred)
    best_k = max(scores, key=scores.get)
    return best_k

# Fungsi untuk membuat dan melatih model
@st.cache_resource
def load_model(file_path):
    X_train, y_train, _, _ = load_data(file_path)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Menyeimbangkan data
    X_balanced, y_balanced = balance_data(X_train_scaled, y_train)
    
    # Menentukan k terbaik
    best_k = find_best_k(X_balanced, y_balanced)
    
    # Melatih model
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_balanced, y_balanced)
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

        result = 'Baik' if prediction[0] == 1 else 'Buruk'
        st.success(f"Hasil klasifikasi: {result} saham")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

# Visualisasi data
st.subheader("Grafik Keuntungan dan Kerugian Saham")
X_train, y_train, _, _ = load_data(file_path)

# Menghitung total keuntungan/kerugian
X_train['Profit/Loss'] = X_train['Close'] - X_train['Open']

# Grafik histogram keuntungan dan kerugian
plt.figure(figsize=(10, 6))
profits = X_train['Profit/Loss'][X_train['Profit/Loss'] > 0]
losses = X_train['Profit/Loss'][X_train['Profit/Loss'] <= 0]
plt.hist(profits, bins=20, color='green', edgecolor='black', label='Keuntungan', alpha=0.7)
plt.hist(losses, bins=20, color='red', edgecolor='black', label='Kerugian', alpha=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title("Distribusi Keuntungan dan Kerugian")
plt.xlabel("Profit/Loss")
plt.ylabel("Frekuensi")
plt.legend()

# Simpan grafik ke buffer untuk ditampilkan
buffer = io.BytesIO()
plt.savefig(buffer, format="png", bbox_inches='tight')
buffer.seek(0)
st.image(buffer)

# Menampilkan grafik data saham dengan bingkai warna
st.subheader("Grafik Tingkat Kerugian dan Keuntungan Saham")
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Close'], X_train['Profit/Loss'], c=np.where(X_train['Profit/Loss'] > 0, 'green', 'red'), label='Data Saham', edgecolors='black')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Tingkat Kerugian dan Keuntungan")
plt.xlabel("Harga Close")
plt.ylabel("Profit/Loss")
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Simpan grafik ke buffer untuk ditampilkan
buffer = io.BytesIO()
plt.savefig(buffer, format="png", bbox_inches='tight')
buffer.seek(0)
st.image(buffer)

st.write("Aplikasi ini membantu dalam memprediksi kategori saham berdasarkan model KNN.")

