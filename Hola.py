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

# Fungsi untuk memilih k terbaik
def find_best_k(X_train, y_train):
    scores = {}
    for k in range(1, 21):  # Mencoba k dari 1 hingga 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        
        # Validasi label
        if set(np.unique(y_train)) == set(np.unique(y_pred)):
            scores[k] = f1_score(y_train, y_pred)
        else:
            scores[k] = 0  # Jika label tidak cocok, beri skor 0

    best_k = max(scores, key=scores.get)
    return best_k

# Fungsi untuk membuat dan melatih model
@st.cache_resource
def load_model(file_path):
    X_train, y_train, _, _ = load_data(file_path)
    
    # Balancing data
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Scaling data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    
    # Cari k terbaik
    best_k = find_best_k(X_train_scaled, y_train_balanced)
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
        user_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Close': [close_price],
            'Volume': [volume]
        })
        user_data_scaled = scaler.transform(user_data)
        prediction = knn_model.predict(user_data_scaled)

        st.success(f"Hasil klasifikasi: {'Baik' if prediction[0] == 'baik' else 'Buruk'} saham")
    else:
        st.error("Silakan masukkan semua fitur dengan nilai yang valid.")

