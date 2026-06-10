import os
import requests
import psycopg2
import joblib
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz

# --- 1. KONFIGURASI (GITHUB SECRETS) ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Konfigurasi Titik Pantau Berdasarkan Analisis Spasial
LOCS = {
    "Tukka": {
        "lat": 1.699608, # Hutanabolon (Hulu Kritis)
        "lon": 98.910028
    }, 
    "Sibabangun": {
        "lat": 1.541647, # Muara Sibuntoan (Biang Banjir)
        "lon": 98.993431
    }
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
                rain_1h = data.get('rain', {}).get('1h', 0.0)
                humidity = data['main']['humidity']
                return rain_1h, humidity
            except Exception as e:
                print(f"Percobaan {i+1} gagal (Lat: {lat}): {e}")
                if i < retries - 1: time.sleep(delay)
        return 0.0, 0 

    print("Mengambil data cuaca di Hutanabolon & Sibabangun...")
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rs, rhs = get_data(LOCS["Sibabangun"]["lat"], LOCS["Sibabangun"]["lon"])
    return rt, rs, rht, rhs

# --- 3. SISTEM UTAMA ---
def run_system():
    # Penentuan Waktu (WIB) menggunakan pytz agar presisi di server maupun database
    wib = pytz.timezone('Asia/Jakarta')
    wib_now = datetime.now(wib)
    
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S%z')
    
    print(f"\n--- SIKLUS EKSEKUSI: {waktu_lengkap} ---")
    
    # A. Ambil Data Cuaca
    rt_hour, rs_hour, rht, rhs = fetch_weather_with_retry()
    
    # B. Koneksi Database
    conn = None
    for _ in range(3):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10)
            cur = conn.cursor()
            break
        except Exception as e:
            print(f"Koneksi Database Gagal: {e}")
            time.sleep(5)
    
    if not conn: return 

    try:
        # C. UPSERT DATA (Memakai skema tabel terbaru dengan kolom _sbbn)
        cur.execute("""
            INSERT INTO histori_harian (
                tanggal, created_at, 
                rain_tuk, rain_tuk_latest, rh_tuk_latest, 
                rain_sbbn, rain_sbbn_latest, rh_sbbn_latest,
                entry_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (tanggal) DO UPDATE SET
                rain_tuk = histori_harian.rain_tuk + EXCLUDED.rain_tuk_latest,
                rain_sbbn = histori_harian.rain_sbbn + EXCLUDED.rain_sbbn_latest,
                rain_tuk_latest = EXCLUDED.rain_tuk_latest,
                rain_sbbn_latest = EXCLUDED.rain_sbbn_latest,
                rh_tuk_latest = EXCLUDED.rh_tuk_latest,
                rh_sbbn_latest = EXCLUDED.rh_sbbn_latest,
                entry_count = histori_harian.entry_count + 1,
                created_at = EXCLUDED.created_at
            RETURNING rain_tuk, rain_sbbn;
        """, (tgl, waktu_lengkap, rt_hour, rt_hour, rht, rs_hour, rs_hour, rhs))
        
        db_res = cur.fetchone()
        total_harian_tuk, total_harian_sibabangun = db_res[0], db_res[1]

        # D. Ambil Histori 3 Hari (Untuk hitung RAIN3)
        cur.execute("SELECT rain_tuk, rain_sbbn FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
        rows = cur.fetchall()
        acc3_t = sum(r[0] for r in rows)
        acc3_s = sum(r[1] for r in rows)
        
        # E. PENCARIAN TITIK TERPARAH (REPRESENTATIF / REP)
        skor_tukka = max(total_harian_tuk, acc3_t)
        skor_sibabangun = max(total_harian_sibabangun, acc3_s)
        
        if skor_tukka >= skor_sibabangun:
            rain_rep, rain3_rep, rh_rep = total_harian_tuk, acc3_t, rht
            lokasi_nama = 'Tukka (Hutanabolon)'
        else:
            rain_rep, rain3_rep, rh_rep = total_harian_sibabangun, acc3_s, rhs
            lokasi_nama = 'Sibabangun (Muara)'

        # F. LOGIKA PREDIKSI (HYBRID: FAIL-SAFE BMKG + AI)
        
        # 1. Cek Hujan Instan Sangat Lebat (Per Jam) - BAHAYA MUTLAK
        if rt_hour >= 10.0 or rs_hour >= 10.0:
            status = "TINGGI"
            logika = "Fail-Safe BMKG: Hujan Instan Lebat/Sangat Lebat (>=10 mm/jam)"
            
        # 2. Cek Titik Buta Model (Boundary Patching) - PAS DI ANGKA 50 & 90
        elif rain_rep == 50.0 and rh_rep == 90.0:
            status = "TINGGI"
            logika = "Fail-Safe BMKG: Menambal Boundary Model (Hujan Tepat 50 mm & RH 90%)"
            
        # 3. Cek Waspada Hujan Instan Sedang (Per Jam) - WASPADA INSTAN
        elif rt_hour >= 5.0 or rs_hour >= 5.0:
            status = "SEDANG"
            logika = "Fail-Safe BMKG: Hujan Instan Sedang (5 - 10 mm/jam)"
            
        # 4. Cek Titik Buta Model (Boundary Patching) - PAS DI ANGKA 20 & 85
        elif rain_rep == 20.0 and rh_rep == 85.0:
            status = "SEDANG"
            logika = "Fail-Safe BMKG: Menambal Boundary Model (Hujan Tepat 20 mm & RH 85%)"
            
        # 5. Model AI mengambil alih sisa probabilitas lainnya
        else:
            try:
                model = joblib.load('model_banjirrrr.pkl')
                le = joblib.load('label_encoderrrr.pkl')
                
                features = ['RAIN', 'RAIN3', 'RH']
                input_df = pd.DataFrame([[rain_rep, rain3_rep, rh_rep]], columns=features)
                
                # Menggunakan predict_proba sesuai di UI
                probabilitas = model.predict_proba(input_df)[0]
                
                # Mendapatkan letak index class otomatis
                idx_rendah = list(le.classes_).index('RENDAH')
                idx_sedang = list(le.classes_).index('SEDANG')
                idx_tinggi = list(le.classes_).index('TINGGI')

                prob_rendah = probabilitas[idx_rendah]
                prob_sedang = probabilitas[idx_sedang]
                prob_tinggi = probabilitas[idx_tinggi]

                # Threshold Tuning / Silent Killer
                if prob_tinggi >= 0.30:
                    status = "TINGGI"
                    logika = f"Analisis AI (Confidence TINGGI: {prob_tinggi*100:.1f}%)"
                elif prob_sedang >= 0.40:
                    status = "SEDANG"
                    logika = f"Analisis AI (Confidence SEDANG: {prob_sedang*100:.1f}%)"
                else:
                    status = "RENDAH"
                    logika = f"Analisis AI (Confidence RENDAH: {prob_rendah*100:.1f}%)"
                    
            except Exception as ai_err:
                print(f"Error AI: {ai_err}")
                status, logika = "RENDAH", "Fallback: Model Loading Error"

        # G. Update Prediksi ke Database
        cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
        conn.commit()
        print(f"Hasil Prediksi: {status} | Lokasi REP: {lokasi_nama} | {logika}")

        # H. Notifikasi Telegram Cerdas
        if status in ["SEDANG", "TINGGI"]:
            # Identifikasi jenis pemicu berdasarkan kata kunci 'Instan' di variabel 'logika'
            is_instan = "Instan" in logika
            
            if status == "SEDANG":
                emoji = "⚠️"
                if is_instan:
                    pesan_himbauan = (
                        "*STATUS: WASPADA (SEDANG)*\n"
                        "Terdeteksi hujan mendadak (instan) yang cukup lebat di wilayah hulu. "
                        "Waspadai potensi debit air sungai yang bisa naik dengan cepat."
                    )
                else:
                    pesan_himbauan = (
                        "*STATUS: WASPADA (SEDANG)*\n"
                        "Terdeteksi peningkatan akumulasi air atau probabilitas cuaca memburuk. "
                         "Waspadai potensi debit air sungai yang bisa naik dengan cepat."
                    )
            
            else:  # Jika STATUS == TINGGI
                emoji = "🚨"
                if is_instan:
                    pesan_himbauan = (
                        "*🚨 STATUS: BAHAYA (TINGGI) 🚨*\n"
                        "PERINGATAN! Terjadi hujan badai mendadak (lebat) di wilayah hulu. "
                        "Risiko BANJIR BANDANG KILAT sangat tinggi.\n"
                        "Mohon segera lakukan langkah antisipasi dan evakuasi jika diperlukan!"
                    )
                else:
                    pesan_himbauan = (
                        "*🚨 STATUS: BAHAYA (TINGGI) 🚨*\n"
                        "PERINGATAN! Akumulasi curah hujan harian/3 hari atau probabilitas cuaca telah mencapai titik kritis. "
                        "Kapasitas sungai berpotensi meluap secara masif.\n"
                        "Mohon segera lakukan langkah antisipasi dan evakuasi jika diperlukan!"
                    )
            
            msg = (f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
                   f"📍 *Titik Pantau Terparah:* {lokasi_nama}\n"
                   f"🌧️ *Hujan Hari Ini (Akumulasi):* {rain_rep:.1f} mm\n"
                   f"🌊 *Hujan Akumulasi (3 Hari):* {rain3_rep:.1f} mm\n"
                   f"💧 *Kelembapan:* {rh_rep}%\n"
                   f"⚙️ *Mekanisme Trigger:* {logika}\n\n"
                   f"📢 *Info:* {pesan_himbauan}")
            
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                         params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    except Exception as e:
        print(f"Error Operasional: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    run_system()
