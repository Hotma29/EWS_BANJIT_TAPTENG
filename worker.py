import os
import requests
import psycopg2
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIG (GitHub Secrets) ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

LOCS = {"Tukka": {"lat": 1.72, "lon": 98.92}, "BTR": {"lat": 1.55, "lon": 99.10}}

def run_system():
    # Timezone WIB (UTC+7)
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Fetch Weather Data
    def get_weather(lat, lon):
        res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric").json()
        return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
    
    rt, rht = get_weather(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_weather(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])

    # 2. Supabase Connection
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 3. UPSERT Data Harian
    cur.execute("""
        INSERT INTO histori_harian (tanggal, created_at, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, entry_count)
        VALUES (%s, %s, %s, %s, %s, %s, 1)
        ON CONFLICT (tanggal) DO UPDATE SET
            rain_tuk = histori_harian.rain_tuk + EXCLUDED.rain_tuk,
            rain_btr = histori_harian.rain_btr + EXCLUDED.rain_btr,
            rh_tuk_avg = (histori_harian.rh_tuk_avg * histori_harian.entry_count + EXCLUDED.rh_tuk_avg) / (histori_harian.entry_count + 1),
            rh_btr_avg = (histori_harian.rh_btr_avg * histori_harian.entry_count + EXCLUDED.rh_btr_avg) / (histori_harian.entry_count + 1),
            entry_count = histori_harian.entry_count + 1,
            created_at = EXCLUDED.created_at;
    """, (tgl, waktu_lengkap, rt, rb, rht, rhb))
    conn.commit()

    # 4. Ambil Histori untuk Akumulasi 3 Hari
    cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    acc3_t = sum(r[0] for r in rows)
    acc3_b = sum(r[1] for r in rows)

    # 5. PREDIKSI AI (9 Fitur Sesuai Model model_ews.pkl)
    input_data = pd.DataFrame([{
        'RAIN_TUKKA': rt, 'RAIN3_TUKKA': acc3_t, 'RH_TUKKA': rht,
        'RAIN_BTR': rb, 'RAIN3_BTR': acc3_b, 'RHBTR': rhb,
        'RAIN_MAX': max(rt, rb), 'RAIN3_MAX': max(acc3_t, acc3_b), 'RH_MAX': max(rht, rhb)
    }])

    try:
        model = joblib.load('model_ews.pkl')[cite: 4]
        le = joblib.load('label_encoder_ews.pkl')[cite: 3]
        
        pred_numeric = model.predict(input_data)
        status = le.inverse_transform(pred_numeric)[0][cite: 3]
    except Exception as e:
        print(f"Error AI: {e}")
        status = "RENDAH"

    # 6. Update Status ke Database
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    
    # 7. Notifikasi Telegram
    if status == "TINGGI":
        msg = f"🚨 *SIAGA BANJIR!* Status: TINGGI\nHujan Max: {max(rt, rb)}mm\nRH Max: {max(rht, rhb)}%"
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_system()
