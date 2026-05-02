import os
import requests
import psycopg2
import joblib
import numpy as np
from datetime import datetime, timedelta

# 1. KONFIGURASI (Diambil dari GitHub Secrets)
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

LOCS = {
    "Tukka": {"lat": 1.72, "lon": 98.92}, 
    "BTR": {"lat": 1.55, "lon": 99.10}
}

def fetch_weather():
    """Mengambil data cuaca real-time"""
    def get_data(lat, lon):
        res = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        ).json()
        return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
    
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    # --- FIX TIMEZONE KE WIB (Tapteng) ---
    # Kita ambil UTC lalu tambah 7 jam agar jam 01:11 WIB tidak dianggap jam 18:11 kemarin.
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    rt, rb, rht, rhb = fetch_weather()
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 2. UPSERT: 1 Tanggal = 1 Baris. 
    # Kita masukkan waktu_lengkap (WIB) ke created_at agar dashboard Supabase benar.
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

    # 3. AMBIL DATA HISTORI (3 Hari Terakhir untuk Acc3)
    cur.execute("SELECT rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    
    # Hitung Akumulasi 3 Hari (Acc3)
    acc3_t = sum(r[0] for r in rows)
    acc3_b = sum(r[1] for r in rows)
    rh_today_t, rh_today_b = rows[0][2], rows[0][3]
    
    # 4. LOGIKA STATUS (Guardrail AND & AI)
    rain_max_hour = max(rt, rb)
    rain_max_today = max(rows[0][0], rows[0][1])
    rh_max = max(rh_today_t, rh_today_b)
    acc3_max = max(acc3_t, acc3_b)
    
    status = "RENDAH" # Default
    
    # Syarat Guardrail: Hujan Ekstrem DAN Udara sangat Lembap (AND)
    if (rain_max_hour >= 50 or rain_max_today >= 50) and rh_max >= 90:
        status = "TINGGI"
    else:
        # Jika tidak ekstrem, serahkan ke Random Forest
        try:
            model = joblib.load('model_banjir_tapteng_final.pkl')
            le = joblib.load('label_encoder_final.pkl')
            # Susun 10 Fitur untuk AI
            feat = np.array([[rt, acc3_t, rh_today_t, rb, acc3_b, rh_today_b, rh_max, rain_max_hour, acc3_max, rh_max]])
            status = le.inverse_transform(model.predict(feat))[0]
        except:
            status = "RENDAH"

    # 5. UPDATE STATUS & KIRIM NOTIFIKASI
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    conn.close()

    if status == "TINGGI":
        msg = (f"🚨 *PERINGATAN DINI BANJIR TAPTENG* 🚨\n\n"
               f"Status: *TINGGI*\n"
               f"Waktu: {wib_now.strftime('%H:%M')} WIB\n"
               f"📍 Hujan Maks: {rain_max_hour} mm\n"
               f"📍 Total Hari Ini: {rain_max_today} mm\n"
               f"📍 Kelembapan: {rh_max}%\n\n"
               f"Masyarakat dihimbau waspada!")
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_system()
