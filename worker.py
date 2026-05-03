import os
import requests
import psycopg2
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. KONFIGURASI (DIAMBIL DARI GITHUB SECRETS) ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Koordinat Lokasi
LOCS = {
    "Tukka": {"lat": 1.72, "lon": 98.92}, 
    "BTR": {"lat": 1.55, "lon": 99.10}
}

def fetch_weather():
    """Mengambil data cuaca real-time dari API OpenWeather"""
    def get_data(lat, lon):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            res = requests.get(url).json()
            rain_1h = res.get('rain', {}).get('1h', 0.0)
            humidity = res['main']['humidity']
            return rain_1h, humidity
        except Exception as e:
            print(f"Gagal mengambil data cuaca: {e}")
            return 0.0, 0
    
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    # Pengaturan Waktu WIB
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Tarik Data Cuaca Terbaru
    rt, rb, rht, rhb = fetch_weather()
    
    # 2. Hubungkan ke Database Supabase
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"Gagal koneksi database: {e}")
        return

    # 3. UPSERT Data Harian (Update jika tanggal sama)
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

    # 4. Ambil Data Histori untuk Perhitungan Akumulasi 3 Hari (RAIN3)
    cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    
    acc3_t = sum(r[0] for r in rows) # Akumulasi Tukka
    acc3_b = sum(r[1] for r in rows) # Akumulasi Batang Toru
    
    # 5. FEATURE ENGINEERING (9 Fitur Sesuai Model ews.pkl)
    rain_max = max(rt, rb)
    rain3_max = max(acc3_t, acc3_b)
    rh_max = max(rht, rhb)
    
    # Susun DataFrame dengan nama kolom persis saat training
    input_data = pd.DataFrame([{
        'RAIN_TUKKA': rt,
        'RAIN3_TUKKA': acc3_t,
        'RH_TUKKA': rht,
        'RAIN_BTR': rb,
        'RAIN3_BTR': acc3_b,
        'RHBTR': rhb,           # Sesuai model: RH Batang Toru tanpa underscore
        'RAIN_MAX': rain_max,
        'RAIN3_MAX': rain3_max,
        'RH_MAX': rh_max
    }])

    # 6. PREDIKSI MENGGUNAKAN AI
    try:
        model = joblib.load('model_ews.pkl')
        le = joblib.load('label_encoder_ews.pkl')
        
        pred_numeric = model.predict(input_data)
        status = le.inverse_transform(pred_numeric)[0]
    except Exception as e:
        print(f"Gagal melakukan prediksi AI: {e}")
        status = "RENDAH"

    # 7. UPDATE HASIL PREDIKSI KE DATABASE
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    
    print(f"[{waktu_lengkap}] Data Tersimpan. Status: {status}")

    # 8. NOTIFIKASI TELEGRAM (Hanya jika TINGGI)
    if status == "TINGGI":
        msg = (f"🚨 *EWS BANJIR TAPTENG: SIAGA* 🚨\n\n"
               f"Status: *TINGGI*\n"
               f"Waktu: {wib_now.strftime('%H:%M')} WIB\n"
               f"📍 Hujan Maks: {rain_max} mm\n"
               f"📍 Akumulasi 3 Hari: {rain3_max} mm\n\n"
               f"Harap waspada untuk wilayah hilir!")
        
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_system()
