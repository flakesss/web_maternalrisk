# api/index.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# Mendapatkan path ke root direktori
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path ke folder 'model'
MODEL_FOLDER = os.path.join(ROOT_DIR, 'model')

# Inisialisasi Flask app dengan path absolut untuk templates dan static
app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, 'templates'), static_folder=os.path.join(ROOT_DIR, 'static'))

# Memuat model, scaler, dan label encoder
model = joblib.load(os.path.join(MODEL_FOLDER, 'model.pkl'))
scaler = joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_FOLDER, 'label_encoder.pkl'))

# Fungsi untuk memberikan saran berdasarkan tingkat risiko
def give_advice(risk_level):
    if risk_level == 'low risk':
        advice = "Risiko Anda rendah. Tetap jaga pola hidup sehat dan rutin periksa ke dokter."
    elif risk_level == 'mid risk':
        advice = "Risiko Anda sedang. Disarankan untuk konsultasi lebih lanjut dengan dokter dan memonitor kondisi kesehatan Anda."
    elif risk_level == 'high risk':
        advice = "Risiko Anda tinggi. Segera hubungi profesional medis untuk mendapatkan penanganan lebih lanjut."
    else:
        advice = "Data tidak valid. Silakan coba lagi."
    return advice

# Fungsi untuk menentukan kelas CSS berdasarkan tingkat risiko
def get_risk_class(risk_level):
    if risk_level == 'low risk':
        return 'low-risk'
    elif risk_level == 'mid risk':
        return 'mid-risk'
    elif risk_level == 'high risk':
        return 'high-risk'
    else:
        return ''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Mengambil data dari form
            age = float(request.form['age'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            bs = float(request.form['bs'])
            body_temp = float(request.form['body_temp'])
            heart_rate = float(request.form['heart_rate'])

            # Membuat DataFrame dari input pengguna
            user_input = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                                      columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

            # Standarisasi input pengguna
            user_input_scaled = scaler.transform(user_input)

            # Memprediksi tingkat risiko
            risk_encoded = model.predict(user_input_scaled)
            risk_level = label_encoder.inverse_transform(risk_encoded)[0]

            # Memberikan saran berdasarkan tingkat risiko
            advice = give_advice(risk_level)

            # Mendapatkan kelas CSS untuk tingkat risiko
            risk_class = get_risk_class(risk_level)

            return render_template('index.html', prediction=risk_level, advice=advice, risk_class=risk_class)
        except ValueError:
            error_message = "Input tidak valid. Pastikan Anda memasukkan angka."
            return render_template('index.html', error=error_message)
    else:
        return render_template('index.html')

# Menjalankan aplikasi secara lokal
if __name__ == "__main__":
    app.run(debug=True)
