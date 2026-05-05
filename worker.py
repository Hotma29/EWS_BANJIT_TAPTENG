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

    # 4. Ambil Data Histori untuk Perhitungan Akumulasi 3 Hari (RAIN3)
    cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    
    acc3_t = sum(r[0] for r in rows) # Akumulasi Tukka
    acc3_b = sum(r[1] for r in rows) # Akumulasi Batang Toru
    
    # 5. FEATURE ENGINEERING (Wajib 10 Fitur untuk model terbaru)
    rain_max = max(rt, rb)
    rain3_max = max(acc3_t, acc3_b)
    rh_max = max(rht, rhb)
    rata_rh = (rht + rhb) / 2  # Fitur ke-7 yang baru
    
    # --- SUSUN DATAFRAME SESUAI URUTAN MODEL ---
    # Nama kolom harus persis dengan model.feature_names_in_
    features = [
        'RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 
        'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 
        'RATA-RATA_RH', 
        'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX'
    ]
    
    input_values = [[
        rt, acc3_t, rht, 
        rb, acc3_b, rhb, 
        rata_rh, 
        rain_max, rain3_max, rh_max
    ]]
    
    input_df = pd.DataFrame(input_values, columns=features)

    # 6. PREDIKSI MENGGUNAKAN AI
    try:
        # Load model dan label encoder terbaru
        model = joblib.load('model_ews_mytapteng.pkl')
        le = joblib.load('label_encoder_ews_mytapteng.pkl')
        
        pred_numeric = model.predict(input_df)
        status = le.inverse_transform(pred_numeric)[0]
    except Exception as e:
        print(f"Gagal melakukan prediksi AI: {e}")
        status = "RENDAH"

    # 7. UPDATE HASIL PREDIKSI KE DATABASE
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    
    print(f"[{waktu_lengkap}] Sync Berhasil. Status AI: {status}")

    # 8. NOTIFIKASI TELEGRAM (Jika SIAGA/SEDANG atau TINGGI)
    if status in ["SEDANG", "TINGGI"]:
        emoji = "⚠️" if status == "SEDANG" else "🚨"
        msg = (f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
               f"Status: *{status}*\n"
               f"Waktu: {wib_now.strftime('%H:%M')} WIB\n"
               f"📍 Hujan Saat Ini: {rain_max} mm\n"
               f"📍 Akumulasi 3 Hari: {rain3_max} mm\n"
               f"📍 Kelembapan (RH): {rh_max}%\n\n"
               f"Harap tetap waspada!")
        
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_system()
