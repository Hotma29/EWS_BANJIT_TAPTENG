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

# Fungsi Fetch Weather dengan Retry Logic (3x Percobaan)
def fetch_weather_with_retry(retries=3, delay=5):
    def get_data(lat, lon):
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        for i in range(retries):
            try:
                res = requests.get(url, timeout=10)
                res.raise_for_status() # Cek jika ada error HTTP
                data = res.json()
                rain_1h = data.get('rain', {}).get('1h', 0.0)
                humidity = data['main']['humidity']
                return rain_1h, humidity
            except Exception as e:
                print(f"Percobaan {i+1} gagal untuk lat {lat}: {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    return 0.0, 0 # Fallback jika semua retry gagal
        return 0.0, 0

    print("Mengambil data cuaca dari hulu...")
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    # Sinkronisasi Waktu
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"--- Eksekusi Sistem: {waktu_lengkap} WIB ---")
    
    # 1. Tarik Data dengan Retry
    rt_hour, rb_hour, rht, rhb = fetch_weather_with_retry()
    
    # 2. Koneksi Database dengan Retry
    conn = None
    for i in range(3):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10)
            cur = conn.cursor()
            break
        except Exception as e:
            print(f"Koneksi DB Gagal (Percobaan {i+1}): {e}")
            time.sleep(5)
    
    if not conn:
        print("CRITICAL: Gagal terhubung ke Database setelah 3x percobaan.")
        return

    try:
        # 3. UPSERT & AMBIL TOTAL HARIAN
        cur.execute("""
            INSERT INTO histori_harian (tanggal, created_at, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, entry_count)
            VALUES (%s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (tanggal) DO UPDATE SET
                rain_tuk = histori_harian.rain_tuk + EXCLUDED.rain_tuk,
                rain_btr = histori_harian.rain_btr + EXCLUDED.rain_btr,
                rh_tuk_avg = (histori_harian.rh_tuk_avg * histori_harian.entry_count + EXCLUDED.rh_tuk_avg) / (histori_harian.entry_count + 1),
                rh_btr_avg = (histori_harian.rh_btr_avg * histori_harian.entry_count + EXCLUDED.rh_btr_avg) / (histori_harian.entry_count + 1),
                entry_count = histori_harian.entry_count + 1,
                created_at = EXCLUDED.created_at
            RETURNING rain_tuk, rain_btr;
        """, (tgl, waktu_lengkap, rt_hour, rb_hour, rht, rhb))
        
        db_res = cur.fetchone()
        total_hari_ini_tuk = db_res[0]
        total_hari_ini_btr = db_res[1]
        
        # 4. AMBIL DATA HISTORI 3 HARI (ACC3)
        cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
        rows = cur.fetchall()
        acc3_t = sum(r[0] for r in rows)
        acc3_b = sum(r[1] for r in rows)
        
        # 5. FEATURE ENGINEERING
        skor_tukka = max(total_hari_ini_tuk, acc3_t)
        skor_btr = max(total_hari_ini_btr, acc3_b)
        
        if skor_tukka >= skor_btr:
            rain_max, rain3_max, rh_max = total_hari_ini_tuk, acc3_t, rht
        else:
            rain_max, rain3_max, rh_max = total_hari_ini_btr, acc3_b, rhb
            
        rata_rh = (rht + rhb) / 2

        # 6. LOGIKA HYBRID (HARD RULE + AI)
        # Sesuai argumen skripsi Bang Hotma: Prioritas Keselamatan
        if (rain_max >= 50 or rain3_max >= 100 or (rain_max >= 40 and rh_max >= 90)):
            status = "TINGGI"
            logika = f"Hard Rule: Kondisi Kritis Terdeteksi"
        else:
            try:
                model = joblib.load('model_banjir_mine.pkl')
                le = joblib.load('label_encoder_mine.pkl')
                
                features = [
                    'RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 
                    'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 
                    'RATA-RATA_RH', 
                    'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX'
                ]
                
                input_data = [[
                    total_hari_ini_tuk, acc3_t, rht, 
                    total_hari_ini_btr, acc3_b, rhb, 
                    rata_rh, 
                    rain_max, rain3_max, rh_max
                ]]
                
                input_df = pd.DataFrame(input_data, columns=features)
                pred_angka = model.predict(input_df)
                status = le.inverse_transform(pred_angka)[0]
                logika = "Analisis AI Random Forest"
            except Exception as e:
                print(f"AI Prediction Error: {e}")
                status = "RENDAH"
                logika = "Fallback: Sistem Standar"

        # 7. SIMPAN HASIL PREDIKSI
        cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
        conn.commit()
        print(f"Berhasil! Status: {status} | Mekanisme: {logika}")

        # 8. NOTIFIKASI TELEGRAM
        if status in ["SEDANG", "TINGGI"]:
            emoji = "⚠️" if status == "SEDANG" else "🚨"
            msg = (f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
                   f"Waktu: {waktu_lengkap} WIB\n"
                   f"Hujan Harian: {rain_max:.2f} mm\n"
                   f"Akumulasi 3 Hari: {rain3_max:.2f} mm\n"
                   f"Kelembapan: {rh_max}%\n"
                   f"Logika: {logika}\n\n"
                   f"Harap tetap waspada!")
            
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                         params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    except Exception as e:
        print(f"Error dalam proses sistem: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    run_system()
