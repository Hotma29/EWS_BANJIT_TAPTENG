import os
import requests
import psycopg2
import joblib
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# --- 1. KONFIGURASI (GITHUB SECRETS) ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

LOCS = {
    "Tukka": {"lat": 1.72, "lon": 98.92}, 
    "BTR": {"lat": 1.55, "lon": 99.10}
}

# --- 2. FUNGSI FETCH DATA (RETRY LOGIC) ---
def fetch_weather_with_retry(retries=3, delay=5):
    def get_data(lat, lon):
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        for i in range(retries):
            try:
                res = requests.get(url, timeout=10)
                res.raise_for_status()
                data = res.json()
                # Ambil data hujan 1 jam terakhir, jika tidak ada default 0.0
                rain_1h = data.get('rain', {}).get('1h', 0.0)
                humidity = data['main']['humidity']
                return rain_1h, humidity
            except Exception as e:
                print(f"Percobaan {i+1} gagal (Lat: {lat}): {e}")
                if i < retries - 1: time.sleep(delay)
        return 0.0, 0 # Fallback jika gagal total

    print("Mengambil data cuaca dari OpenWeather API...")
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

# --- 3. SISTEM UTAMA ---
def run_system():
    # Penentuan Waktu (WIB)
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n--- SIKLUS EKSEKUSI: {waktu_lengkap} WIB ---")
    
    # A. Ambil Data Cuaca
    rt_hour, rb_hour, rht, rhb = fetch_weather_with_retry()
    
    # B. Koneksi Database dengan Retry
    conn = None
    for _ in range(3):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10)
            cur = conn.cursor()
            break
        except Exception as e:
            print(f"Koneksi Database Gagal: {e}")
            time.sleep(5)
    
    if not conn: return # Stop jika DB tidak bisa diakses

    try:
        # C. UPSERT DATA (Simpan Akumulasi & Data Terakhir)
        # Menangani data Harian, Rata-rata, dan Latest sekaligus
        cur.execute("""
            INSERT INTO histori_harian (
                tanggal, created_at, 
                rain_tuk, rain_btr, 
                rh_tuk_avg, rh_btr_avg, 
                rh_tuk_latest, rh_btr_latest,
                rain_tuk_latest, rain_btr_latest,
                entry_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (tanggal) DO UPDATE SET
                -- Update Akumulasi Hujan
                rain_tuk = histori_harian.rain_tuk + EXCLUDED.rain_tuk,
                rain_btr = histori_harian.rain_btr + EXCLUDED.rain_btr,
                -- Update Rata-rata Kelembapan
                rh_tuk_avg = (histori_harian.rh_tuk_avg * histori_harian.entry_count + EXCLUDED.rh_tuk_avg) / (histori_harian.entry_count + 1),
                rh_btr_avg = (histori_harian.rh_btr_avg * histori_harian.entry_count + EXCLUDED.rh_btr_avg) / (histori_harian.entry_count + 1),
                -- Update Nilai Terakhir (Untuk UI)
                rh_tuk_latest = EXCLUDED.rh_tuk_latest,
                rh_btr_latest = EXCLUDED.rh_btr_latest,
                rain_tuk_latest = EXCLUDED.rain_tuk_latest,
                rain_btr_latest = EXCLUDED.rain_btr_latest,
                
                entry_count = histori_harian.entry_count + 1,
                created_at = EXCLUDED.created_at
            RETURNING rain_tuk, rain_btr;
        """, (tgl, waktu_lengkap, rt_hour, rb_hour, rht, rhb, rht, rhb, rt_hour, rb_hour))
        
        db_res = cur.fetchone()
        total_harian_tuk, total_harian_btr = db_res[0], db_res[1]

        # D. Ambil Histori 3 Hari (Untuk Fitur RAIN3)
        cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
        rows = cur.fetchall()
        acc3_t = sum(r[0] for r in rows)
        acc3_b = sum(r[1] for r in rows)
        
        # E. Feature Engineering (Integritas Spasial)
        skor_tukka = max(total_harian_tuk, acc3_t)
        skor_btr = max(total_harian_btr, acc3_b)
        
        if skor_tukka >= skor_btr:
            rain_max, rain3_max, rh_max = total_harian_tuk, acc3_t, rht
        else:
            rain_max, rain3_max, rh_max = total_harian_btr, acc3_b, rhb
            
        rata_rh = (rht + rhb) / 2

        # F. Logika Prediksi (Hybrid)
        if (rain_max >= 50 or rain3_max >= 100 or (rain_max >= 40 and rh_max >= 90)):
            status = "TINGGI"
            logika = "Hard Rule: Ambang Batas Kritis"
        else:
            try:
                model = joblib.load('model_banjir_mine.pkl')
                le = joblib.load('label_encoder_mine.pkl')
                
                features = ['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX']
                input_df = pd.DataFrame([[total_harian_tuk, acc3_t, rht, total_harian_btr, acc3_b, rhb, rata_rh, rain_max, rain3_max, rh_max]], columns=features)
                
                pred = model.predict(input_df)
                status = le.inverse_transform(pred)[0]
                logika = "Analisis AI Random Forest"
            except:
                status, logika = "RENDAH", "Fallback: Model Loading Error"

        # G. Update Prediksi ke Database
        cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
        conn.commit()
        print(f"Hasil: {status} | {logika}")

        # H. Notifikasi Telegram
        if status in ["SEDANG", "TINGGI"]:
            emoji = "⚠️" if status == "SEDANG" else "🚨"
            msg = (f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
                   f"Lokasi Kritis: {'Tukka' if skor_tukka >= skor_btr else 'B. Toru'}\n"
                   f"Hujan Harian: {rain_max:.1f} mm\n"
                   f"Kelembapan: {rh_max}%\n"
                   f"Mekanisme: {logika}\n\n"
                   f"Waspada potensi banjir bandang!")
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    except Exception as e:
        print(f"Error Operasional: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    run_system()
