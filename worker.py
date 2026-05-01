import os
import requests
import psycopg2
import pandas as pd
import joblib

# --- CONFIG ---
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

# 1. AMBIL DATA (KOORDINAT PRESISI)
rain_tuk, rh_tuk = get_weather(1.72, 98.92) # Tukka
rain_btr, rh_btr = get_weather(1.55, 99.08) # Batang Toru

# 2. FEATURE ENGINEERING
max_rain_t = max(rain_tuk, rain_btr)
max_rh = max(rh_tuk, rh_btr)
rh_avg = (rh_tuk + rh_btr) / 2
# Untuk rain3, karena data ditarik tiap jam, kita asumsikan akumulasi bertahap
rain3_tuk, rain3_btr = rain_tuk, rain_btr 

# 3. LOGIKA HYBRID
if max_rain_t >= 50 and max_rh >= 90:
    prediksi_final = "TINGGI"
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

# 4. SIMPAN (Pastikan tabel Supabase siap)
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

if prediksi_final == "TINGGI":
    send_telegram(f"🚨 *EWS TAPTENG (WIB)*\nStatus: TINGGI!\nMax Hujan: {max_rain_t}mm\nMax RH: {max_rh}%")

print(f"Berhasil Update Lokasi & Prediksi: {prediksi_final}")
