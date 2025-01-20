import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan scaler
model_path = 'toyota_modelnew.sav'  # Pastikan file model .sav ada di path ini
scaler_path = 'scalernew.sav'  # Path ke scaler yang disimpan

# Memuat model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Memuat scaler

# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas menggunakan Random Forest")

# Input data pengguna
year = st.number_input("Tahun", min_value=1980, max_value=2025, value=2016)
mileage = st.number_input("Jarak Tempuh", min_value=0, value=30000)
tax = st.number_input("Pajak", min_value=0, value=150)
mpg = st.number_input("Efisiensi Bahan Bakar", min_value=0.0, value=20.0)
engine_size = st.number_input("Ukuran Mesin", min_value=0.0, value=2.0)

# Button untuk prediksi
if st.button("Prediksi Harga"):
    # Membentuk array input
    input_data = np.array([[year, mileage, tax, mpg, engine_size]])
    
    # Melakukan scaling pada data input (jika ada scaler)
    input_data_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi
    prediction = model.predict(input_data_scaled)
    
    # Menampilkan hasil prediksi
    st.write(f"Prediksi Harga Mobil dalam £: {prediction[0]:.2f}")
    st.write(f"Predicted price in Rupiah: Rp {prediction[0] * 19110 * 1e-6:.2f} Juta")

# Penjelasan aplikasi
st.write("Masukkan data mobil bekas untuk mendapatkan prediksi harga.")