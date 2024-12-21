import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    return X, y, None, None

# Find the best k using cross-validation
def find_best_k(X_train, y_train):
    scores = {}
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        
        # Log label distribution for debugging
        logging.info(f"Iteration k={k}, y_train={np.unique(y_train)}, y_pred={np.unique(y_pred)}")

        # Validate label consistency
        unique_labels_y_train = set(np.unique(y_train))
        unique_labels_y_pred = set(np.unique(y_pred))

        if len(unique_labels_y_train) > 1 and unique_labels_y_train == unique_labels_y_pred:
            scores[k] = f1_score(y_train, y_pred, average="weighted")
        else:
            scores[k] = 0  # Assign score of 0 if labels are not valid

    best_k = max(scores, key=scores.get)
    return best_k

# Load model
@st.cache_resource
def load_model(file_path):
    X_train, y_train, _, _ = load_data(file_path)

    # Log initial label distribution
    logging.info(f"Before balancing: {np.unique(y_train, return_counts=True)}")

    # Handle data imbalance
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Log label distribution after balancing
    logging.info(f"After balancing: {np.unique(y_train_balanced, return_counts=True)}")

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)

    # Find the best k
    best_k = find_best_k(X_train_scaled, y_train_balanced)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train_balanced)

    return knn, scaler, best_k

# Streamlit app
st.title("Klasifikasi Saham Samsung")
st.write("Masukkan fitur saham untuk mengetahui apakah saham baik atau buruk.")

# Input fields
open_price = st.number_input("Harga Open", min_value=0.0, step=0.01)
high_price = st.number_input("Harga High", min_value=0.0, step=0.01)
low_price = st.number_input("Harga Low", min_value=0.0, step=0.01)
close_price = st.number_input("Harga Close", min_value=0.0, step=0.01)
volume = st.number_input("Volume", min_value=0.0, step=0.01)

# Predict button
if st.button("Kategorikan"):
    file_path = "samsung.csv"
    try:
        knn_model, scaler, best_k = load_model(file_path)

        # Prepare input data
        input_data = np.array([[open_price, high_price, low_price, close_price, volume]])
        input_data_scaled = scaler.transform(input_data)

        # Prediction
        prediction = knn_model.predict(input_data_scaled)[0]
        st.success(f"Saham dikategorikan sebagai: {'Baik' if prediction == 1 else 'Buruk'}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Display performance graphs
if st.button("Tampilkan Grafik"):
    try:
        X_train, y_train, _, _ = load_data("samsung.csv")
        smote = SMOTE()
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        # Calculate profit and loss statistics
        profit_count = np.sum(y_balanced == 1)
        loss_count = np.sum(y_balanced == 0)

        # Plot profit vs loss
        fig, ax = plt.subplots()
        ax.bar(["Keuntungan", "Kerugian"], [profit_count, loss_count], color=["green", "red"], edgecolor=["darkgreen", "darkred"], linewidth=2)
        ax.set_title("Jumlah Keuntungan dan Kerugian")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menampilkan grafik: {e}")


