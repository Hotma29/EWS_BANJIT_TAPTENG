import os
import requests
import psycopg2
import joblib
import pandas as pd
import time
from datetime import datetime, timedelta

# --- 1. KONFIGURASI (GITHUB SECRETS) ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Titik Pantau
LOCS = {
    "Tukka": {"lat": 1.699608, "lon": 98.910028}, 
    "Sibabangun": {"lat": 1.541647, "lon": 98.993431}
}

# --- 2. FUNGSI FETCH DATA OPEN-METEO API ---
def fetch_open_meteo(lat, lon, retries=3, delay=5):
    """Menarik data cuaca aktual dan total harian dari Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["precipitation", "relative_humidity_2m", "temperature_2m", "wind_speed_10m"],
        "daily": ["precipitation_sum"],
        "timezone": "Asia/Jakarta",
        "wind_speed_unit": "ms"
    }
    
    for i in range(retries):
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            # Data Instan (Saat Ini)
            ch_latest = data['current']['precipitation']
            rh = data['current']['relative_humidity_2m']
            t2m = data['current']['temperature_2m']
            ws10m = data['current']['wind_speed_10m']
            
            # Total Curah Hujan Hari Ini
            ch_harian = data['daily']['precipitation_sum'][0] 
            
            return ch_harian, ch_latest, rh, t2m, ws10m
            
        except Exception as e:
            print(f"Percobaan {i+1} gagal (Lat: {lat}): {e}")
            if i < retries - 1: time.sleep(delay)
            
    return 0.0, 0.0, 0.0, 0.0, 0.0

# --- 3. SISTEM UTAMA ---
def run_system():
    # Penentuan Waktu (WIB)
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n--- SIKLUS EKSEKUSI: {waktu_lengkap} ---")
    
    # [ALUR 1] Open-Meteo API
    print("Menarik data dari Open-Meteo API...")
    ch_t, chl_t, rh_t, t2m_t, ws_t = fetch_open_meteo(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    ch_s, chl_s, rh_s, t2m_s, ws_s = fetch_open_meteo(LOCS["Sibabangun"]["lat"], LOCS["Sibabangun"]["lon"])
    
    # Koneksi Database
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
        # [ALUR 2] Simpan Data ke Supabase (UPSERT)
        print("Menyimpan data awal ke Supabase...")
        cur.execute("""
            INSERT INTO histori_harian (
                tanggal, created_at, 
                ch_tuk, ch_tuk_latest, rh_tuk, t2m_tuk, ws10m_tuk,
                ch_sbbn, ch_sbbn_latest, rh_sbbn, t2m_sbbn, ws10m_sbbn,
                entry_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (tanggal) DO UPDATE SET
                ch_tuk = EXCLUDED.ch_tuk,
                ch_tuk_latest = EXCLUDED.ch_tuk_latest,
                rh_tuk = EXCLUDED.rh_tuk,
                t2m_tuk = EXCLUDED.t2m_tuk,
                ws10m_tuk = EXCLUDED.ws10m_tuk,
                ch_sbbn = EXCLUDED.ch_sbbn,
                ch_sbbn_latest = EXCLUDED.ch_sbbn_latest,
                rh_sbbn = EXCLUDED.rh_sbbn,
                t2m_sbbn = EXCLUDED.t2m_sbbn,
                ws10m_sbbn = EXCLUDED.ws10m_sbbn,
                entry_count = histori_harian.entry_count + 1,
                created_at = EXCLUDED.created_at;
        """, (tgl, waktu_lengkap, ch_t, chl_t, rh_t, t2m_t, ws_t, ch_s, chl_s, rh_s, t2m_s, ws_s))
        conn.commit()

        # [ALUR 3] Hitung CH3 (Akumulasi Hujan 3 Hari) via SELECT
        print("Menghitung akumulasi CH3 dari database...")
        cur.execute("SELECT ch_tuk, ch_sbbn FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
        rows = cur.fetchall()
        
        # Menjumlahkan riwayat 3 hari (mengabaikan nilai None jika ada)
        ch3_tuk = sum(r[0] for r in rows if r[0] is not None)
        ch3_sbbn = sum(r[1] for r in rows if r[1] is not None)

       # Penentuan Lokasi Representatif
        skor_tukka = max(ch_t, ch3_tuk)
        skor_sibabangun = max(ch_s, ch3_sbbn)

        if skor_tukka >= skor_sibabangun:
            rep_features = {'CH': ch_t, 'CH3': ch3_tuk, 'RH': rh_t, 'T2M': t2m_t, 'WS10M': ws_t}
            chl_rep = chl_t  # <--- Simpan nilai 1 jam untuk Tukka
            lokasi_nama = 'Hulu Tukka'
        else:
            rep_features = {'CH': ch_s, 'CH3': ch3_sbbn, 'RH': rh_s, 'T2M': t2m_s, 'WS10M': ws_s}
            chl_rep = chl_s  # <--- Simpan nilai 1 jam untuk Sibabangun
            lokasi_nama = 'Hulu Sibabangun'
        # [ALUR 4 & 5] Load model.pkl & Random Forest Classification (PURE PREDICT)
        print("Menjalankan inferensi AI Random Forest...")
        try:
            model = joblib.load('random_forest_model.pkl')
            le = joblib.load('label_encoder.pkl')
            
            input_df = pd.DataFrame([rep_features])
            
            # PURE PREDICT
            prediksi_encoded = model.predict(input_df)[0]
            status = le.inverse_transform([prediksi_encoded])[0]
            logika = "Murni Klasifikasi Default Random Forest"
                
        except Exception as ai_err:
            print(f"Error AI: {ai_err}")
            status, logika = "RENDAH", "Error Loading Model"

        # [ALUR 6] Update Database (Menyimpan Prediksi dan nilai CH3)
        print("Update status prediksi ke database...")
        cur.execute("""
            UPDATE histori_harian 
            SET prediksi = %s, ch3_tuk = %s, ch3_sbbn = %s 
            WHERE tanggal = %s
        """, (status, ch3_tuk, ch3_sbbn, tgl))
        conn.commit()
        
        print(f"Hasil: {status} | Acuan: {lokasi_nama} | {logika}")

       # [ALUR 7] Telegram Bot Notifikasi
        if status in ["SEDANG", "TINGGI"]:
            
            # Format pesan resmi dan profesional tanpa emoji
            msg = (
                "*INFORMASI POTENSI BANJIR - KAB. TAPANULI TENGAH*\n"
                "--------------------------------------------------\n\n"
                f"*Status Prediksi     :* {status}\n"
                f"*Titik Pemantauan :* Hulu {lokasi_nama}\n\n"
                "*Data Hidrometeorologi:*\n"
                f"- Curah Hujan (1 Jam) : {rep_features['chl_rep']} mm\n"
                f"- Curah Hujan (1 Jam) : {rep_features['CH']} mm\n"
                f"- Akumulasi Hujan (CH3) : {rep_features['CH3']:.1f} mm\n"
                f"- Kelembapan Udara (RH) : {rep_features['RH']} %\n"
                f"- Suhu Udara (T2M)      : {rep_features['T2M']} °C\n"
                f"- Kecepatan Angin       : {rep_features['WS10M']} m/s\n\n"
                f"*Waktu Pembaruan:* {waktu_lengkap} WIB\n"
                "--------------------------------------------------\n"
                "_Pesan ini dihasilkan secara otomatis oleh sistem._"
            )
            
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
