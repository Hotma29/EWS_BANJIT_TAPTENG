import os
import requests
import psycopg2
import joblib
import numpy as np
from datetime import datetime

# Ambil konfigurasi dari GitHub Secrets
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

LOCS = {"Tukka": {"lat": 1.72, "lon": 98.92}, "BTR": {"lat": 1.55, "lon": 99.10}}

def fetch_weather():
    def get_data(lat, lon):
        res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric").json()
        return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    rt, rb, rht, rhb = fetch_weather()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    tgl = datetime.now().strftime('%Y-%m-%d')
    
    # 1. LOGIKA UPSERT: 1 Tanggal = 1 Baris (Mencegah data skip)
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

    # 2. AMBIL DATA 3 HARI TERAKHIR
    cur.execute("SELECT rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    
    acc3_t = sum(r[0] for r in rows)
    acc3_b = sum(r[1] for r in rows)
    today_t, today_b = rows[0][0], rows[0][1]
    
    # 3. LOGIKA MAX & GUARDRAIL (Sesuai Permintaan: Pakai AND)
    rain_max_hour = max(rt, rb)
    rain_max_today = max(today_t, today_b)
    rh_max = max(rows[0][2], rows[0][3])
    acc3_max = max(acc3_t, acc3_b)
    
    status = "RENDAH" # Default
    # Logika Guardrail: Jika Hujan Ekstrem DAN Udara sangat Lembap
    if (rain_max_hour >= 50 or rain_max_today >= 50) and rh_max >= 90:
        status = "TINGGI"
    else:
        # Jika syarat ekstrem tidak terpenuhi, biarkan AI yang memutuskan
        try:
            model = joblib.load('model_banjir_tapteng_final.pkl')
            le = joblib.load('label_encoder_final.pkl')
            # 10 Fitur untuk AI
            feat = np.array([[rt, acc3_t, rows[0][2], rb, acc3_b, rows[0][3], rh_max, rain_max_hour, acc3_max, rh_max]])
            status = le.inverse_transform(model.predict(feat))[0]
        except:
            status = "PENDING"

    # Update Status Terakhir ke Database
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    conn.close()

    # 4. NOTIFIKASI TELEGRAM (Hanya jika TINGGI)
    if status == "TINGGI":
        msg = (f"🚨 *EWS BANJIR TAPTENG: SIAGA* 🚨\n\nStatus: *{status}*\n"
               f"📍 Hujan Maks (Jam Ini): {rain_max_hour} mm\n"
               f"📍 Total Hujan Hari Ini: {rain_max_today} mm\n"
               f"📍 Kelembapan (RH): {rh_max}%\n"
               f"📍 Akumulasi 3 Hari: {acc3_max} mm\n\n"
               f"Masyarakat dihimbau untuk tetap waspada!")
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_system()
