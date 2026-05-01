import os
import requests
import psycopg2
import pandas as pd
import joblib
from datetime import datetime

# --- CONFIG SECRETS ---
DB_URL = os.environ.get("SUPABASE_DB_URL")
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url).json()
    rain = resp.get('rain', {}).get('1h', 0)
    rh = resp.get('main', {}).get('humidity', 0)
    return rain, rh

def send_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"})

# 1. AMBIL DATA CUACA (KONSISTEN LOKASI)
# Hulu Tukka: 1.72, 98.92
rain_tuk, rh_tuk = get_weather(1.72, 98.92)
# Hulu Batang Toru: 1.55, 99.08
rain_btr, rh_btr = get_weather(1.55, 99.08)

# 2. FEATURE ENGINEERING UNTUK MODEL AI
max_rain_t = max(rain_tuk, rain_btr)
max_rh = max(rh_tuk, rh_btr)
rh_avg = (rh_tuk + rh_btr) / 2
# Akumulasi 3H (Jika API tidak sedia, sementara pakai rain saat ini atau 0 untuk demo)
rain3_tuk, rain3_btr = rain_tuk, rain_btr 

# 3. LOGIKA HYBRID (ATURAN 50/90)
if max_rain_t >= 50 and max_rh >= 90:
    prediksi_final = "TINGGI"
    metode = "Override Manual (Ekstrem)"
else:
    try:
        model = joblib.load('model_banjir_tapteng_final.pkl')
        encoder = joblib.load('label_encoder_final.pkl')
        
        input_data = pd.DataFrame([[
            rain_tuk, rain3_tuk, rh_tuk, rain_btr, 
            rain3_btr, rh_btr, rh_avg, max_rain_t, max_rain_t, max_rh
        ]], columns=['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 
                    'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'MAX_RAIN_T', 
                    'MAX_RAIN3', 'MAX_RH'])
        
        prediksi_num = model.predict(input_data)
        prediksi_final = encoder.inverse_transform(prediksi_num)[0]
    except:
        prediksi_final = "AMAN"

# 4. SIMPAN KE DATABASE SUPABASE
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()
query = """
INSERT INTO histori_harian (rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, prediksi)
VALUES (%s, %s, %s, %s, %s)
"""
cur.execute(query, (rain_tuk, rain_btr, rh_tuk, rh_btr, prediksi_final))
conn.commit()
cur.close()
conn.close()

# 5. KIRIM NOTIFIKASI TELEGRAM JIKA TINGGI
if prediksi_final == "TINGGI":
    send_telegram(f"🚨 *EWS BANJIR TAPTENG*\nStatus: TINGGI!\nLokasi: Hulu Tukka & Batang Toru\nHujan Terdeteksi: {max_rain_t} mm\nRH Terdeteksi: {max_rh} %")

print(f"Update Berhasil. Lokasi Konsisten. Prediksi: {prediksi_final}")
