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
            
            # Data Instan (Saat Ini / 1 Jam Terakhir)
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
    # Penentuan Waktu (WIB) murni
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n--- SIKLUS EKSEKUSI: {waktu_lengkap} ---")
    
    # [ALUR 1] Open-Meteo API 
    print("Menarik data dari Open-Meteo API...")
    ch_tuk, ch_tuk_latest, rh_tuk, t2m_tuk, ws10m_tuk = fetch_open_meteo(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    ch_sbbn, ch_sbbn_latest, rh_sbbn, t2m_sbbn, ws10m_sbbn = fetch_open_meteo(LOCS["Sibabangun"]["lat"], LOCS["Sibabangun"]["lon"])
    
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
        """, (tgl, waktu_lengkap, ch_tuk, ch_tuk_latest, rh_tuk, t2m_tuk, ws10m_tuk, 
              ch_sbbn, ch_sbbn_latest, rh_sbbn, t2m_sbbn, ws10m_sbbn))
        conn.commit()

        # [ALUR 3] Hitung CH3 via SELECT
        print("Menghitung akumulasi CH3 dari database...")
        cur.execute("SELECT ch_tuk, ch_sbbn FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
        rows = cur.fetchall()
        
        ch3_tuk = sum(r[0] for r in rows if r[0] is not None)
        ch3_sbbn = sum(r[1] for r in rows if r[1] is not None)

        # [ALUR 4 & 5] Random Forest Classification (Inferensi Ganda)
        print("Menjalankan inferensi AI Random Forest untuk kedua hulu...")
        try:
            model = joblib.load('random_forest_model.pkl')
            le = joblib.load('label_encoder.pkl')
            
            # Siapkan fitur kedua lokasi
            features_tukka = {'CH': ch_tuk, 'CH3': ch3_tuk, 'RH': rh_tuk, 'T2M': t2m_tuk, 'WS10M': ws10m_tuk}
            features_sbbn = {'CH': ch_sbbn, 'CH3': ch3_sbbn, 'RH': rh_sbbn, 'T2M': t2m_sbbn, 'WS10M': ws10m_sbbn}
            
            # Prediksi Hulu Tukka
            df_tukka = pd.DataFrame([features_tukka])
            status_tukka = le.inverse_transform([model.predict(df_tukka)[0]])[0].upper()
            
            # Prediksi Hulu Sibabangun
            df_sbbn = pd.DataFrame([features_sbbn])
            status_sbbn = le.inverse_transform([model.predict(df_sbbn)[0]])[0].upper()
            
            # Penentuan Lokasi Representatif (Worst-Case Scenario)
            hierarki = {"RENDAH": 1, "SEDANG": 2, "TINGGI": 3}
            
            if hierarki[status_tukka] > hierarki[status_sbbn]:
                status = status_tukka
                lokasi_nama = "Hulu Tukka"
                rep_features = features_tukka
                chl_rep = ch_tuk_latest  # Ambil CH 1 Jam dari Tukka
            elif hierarki[status_sbbn] > hierarki[status_tukka]:
                status = status_sbbn
                lokasi_nama = "Hulu Sibabangun"
                rep_features = features_sbbn
                chl_rep = ch_sbbn_latest  # Ambil CH 1 Jam dari Sibabangun
            else:
                # Jika status sama (Tie-Breaker: Pilih curah hujan harian tertinggi)
                status = status_tukka
                if features_tukka['CH'] >= features_sbbn['CH']:
                    lokasi_nama = "Hulu Tukka"
                    rep_features = features_tukka
                    chl_rep = ch_tuk_latest
                else:
                    lokasi_nama = "Hulu Sibabangun"
                    rep_features = features_sbbn
                    chl_rep = ch_sbbn_latest
                
        except Exception as ai_err:
            print(f"Error AI: {ai_err}")
            status = "RENDAH"
            lokasi_nama = "Hulu Tukka"
            rep_features = {'CH': ch_tuk, 'CH3': ch3_tuk, 'RH': rh_tuk, 'T2M': t2m_tuk, 'WS10M': ws10m_tuk}
            chl_rep = ch_tuk_latest

        # [ALUR 6] Update Database
        print("Update status prediksi ke database...")
        cur.execute("""
            UPDATE histori_harian 
            SET prediksi = %s, ch3_tuk = %s, ch3_sbbn = %s 
            WHERE tanggal = %s
        """, (status, ch3_tuk, ch3_sbbn, tgl))
        conn.commit()
        
        print(f"Hasil: {status} | Acuan: {lokasi_nama}")

        # [ALUR 7] Telegram Bot Notifikasi
        if status in ["SEDANG", "TINGGI"]:
            if status == "TINGGI":
                pesan_himbauan = "BAHAYA: Mohon segera lakukan langkah antisipasi dan evakuasi jika diperlukan!"
                level_ancaman = "(BAHAYA)"
            elif status == "SEDANG":
                pesan_himbauan = "WASPADA: Pantau terus kondisi cuaca secara berkala!"
                level_ancaman = "(WASPADA)"
            
            msg = (
                "*INFORMASI POTENSI BANJIR BANDANG - KAB. TAPANULI TENGAH*\n"
                "--------------------------------------------------\n\n"
                f"*Potensi Banjir  :* {status} {level_ancaman}\n"
                f"*Titik Sumber   :* {lokasi_nama}\n\n"
                "*Data Meteorologi:*\n"
                f"- Curah Hujan (1 Jam)   : {chl_rep} mm\n"
                f"- Curah Hujan (Harian)  : {rep_features['CH']} mm\n"
                f"- Akumulasi Hujan (CH3) : {rep_features['CH3']:.1f} mm\n"
                f"- Kelembapan Udara (RH) : {rep_features['RH']} %\n"
                f"- Suhu Udara (T2M)      : {rep_features['T2M']} °C\n"
                f"- Kecepatan Angin       : {rep_features['WS10M']} m/s\n\n"
                f"*Instruksi Mitigasi:*\n{pesan_himbauan}\n\n"
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
