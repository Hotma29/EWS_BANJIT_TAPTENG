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

# Koordinat Hulu
LOCS = {
    "Tukka": {"lat": 1.72, "lon": 98.92}, 
    "BTR": {"lat": 1.55, "lon": 99.10}
}

def fetch_weather():
    """Mengambil data cuaca real-time dari OpenWeatherMap"""
    def get_data(lat, lon):
        res = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        ).json()
        return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
    
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    # --- STANDARISASI WAKTU WIB (Tapteng) ---
    # Karena server GitHub Actions pakai UTC+0, kita tambah 7 jam
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    
    # Ambil data cuaca jam ini
    rt, rb, rht, rhb = fetch_weather()
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 2. LOGIKA UPSERT: 1 Tanggal = 1 Baris (Menabung data harian)
    cur.execute("""
        INSERT INTO histori_harian (tanggal, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, entry_count)
        VALUES (%s, %s, %s, %s, %s, 1)
        ON CONFLICT (tanggal) DO UPDATE SET
            rain_tuk = histori_harian.rain_tuk + EXCLUDED.rain_tuk,
            rain_btr = histori_harian.rain_btr + EXCLUDED.rain_btr,
            rh_tuk_avg = (histori_harian.rh_tuk_avg * histori_harian.entry_count + EXCLUDED.rh_tuk_avg) / (histori_harian.entry_count + 1),
            rh_btr_avg = (histori_harian.rh_btr_avg * histori_harian.entry_count + EXCLUDED.rh_btr_avg) / (histori_harian.entry_count + 1),
            entry_count = histori_harian.entry_count + 1;
    """, (tgl, rt, rb, rht, rhb))
    conn.commit()

    # 3. AMBIL DATA HISTORI (3 Hari Terakhir untuk Acc3)
    cur.execute("SELECT rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    
    # Hitung Akumulasi 3 Hari ($Acc3$)
    acc3_t = sum(r[0] for r in rows)
    acc3_b = sum(r[1] for r in rows)
    
    # Ambil nilai hari ini (baris paling atas)
    today_t, today_b = rows[0][0], rows[0][1]
    rh_today_t, rh_today_b = rows[0][2], rows[0][3]
    
    # 4. LOGIKA PENENTUAN STATUS (Guardrail & AI)
    rain_max_hour = max(rt, rb)
    rain_max_today = max(today_t, today_b)
    rh_max = max(rh_today_t, rh_today_b)
    acc3_max = max(acc3_t, acc3_b)
    
    status = "RENDAH" # Default awal
    
    # LOGIKA GUARDRAIL EKSTREM (Sesuai Permintaan: Hujan DAN RH)
    if (rain_max_hour >= 50 or rain_max_today >= 50) and rh_max >= 90:
        status = "TINGGI"
    else:
        # JIKA TIDAK EKSTREM, GUNAKAN PREDIKSI RANDOM FOREST (10 FITUR)
        try:
            model = joblib.load('model_banjir_tapteng_final.pkl')
            le = joblib.load('label_encoder_final.pkl')
            
            # Susun Fitur: rt, acc3_t, rht, rb, acc3_b, rhb, rh_max, rain_max_hour, acc3_max, rh_max
            feat = np.array([[rt, acc3_t, rh_today_t, rb, acc3_b, rh_today_b, rh_max, rain_max_hour, acc3_max, rh_max]])
            
            pred = model.predict(feat)
            status = le.inverse_transform(pred)[0]
        except Exception as e:
            print(f"Error AI: {e}")
            status = "RENDAH"

    # 5. UPDATE STATUS KE DATABASE
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    conn.close()

    # 6. KIRIM NOTIFIKASI TELEGRAM (Hanya jika status TINGGI)
    if status == "TINGGI":
        waktu_pesan = wib_now.strftime('%H:%M')
        msg = (f"🚨 *PERINGATAN DINI BANJIR TAPTENG* 🚨\n\n"
               f"Status: *TINGGI*\n"
               f"Waktu: {waktu_pesan} WIB\n"
               f"--------------------------------\n"
               f"📍 Hujan Maks (Jam ini): {rain_max_hour} mm\n"
               f"📍 Total Hujan Hari Ini: {rain_max_today} mm\n"
               f"📍 Kelembapan (RH): {rh_max}%\n"
               f"📍 Akumulasi 3 Hari: {acc3_max} mm\n\n"
               f"⚠️ Kondisi berpotensi menyebabkan banjir. Harap waspada!")
        
        requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
            params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"}
        )

if __name__ == "__main__":
    run_system()
