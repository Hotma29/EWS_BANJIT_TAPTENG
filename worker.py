import os
import requests
import psycopg2
import joblib
import numpy as np
from datetime import datetime

# Ambil dari GitHub Secrets
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

LOCS = {"Tukka": {"lat": 1.72, "lon": 98.92}, "Batang Toru": {"lat": 1.55, "lon": 99.10}}

def fetch_weather():
    def get_data(lat, lon):
        res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric").json()
        return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["Batang Toru"]["lat"], LOCS["Batang Toru"]["lon"])
    return rt, rb, rht, rhb

def update_db_and_get_features(rt, rb, rht, rhb):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    tgl = datetime.now().strftime('%Y-%m-%d')
    
    # Update data harian
    cur.execute("SELECT * FROM histori_harian WHERE tanggal = %s", (tgl,))
    row = cur.fetchone()
    if row:
        new_count = row[5] + 1
        cur.execute("UPDATE histori_harian SET rain_tuk=%s, rain_btr=%s, rh_tuk_avg=%s, rh_btr_avg=%s, entry_count=%s WHERE tanggal=%s",
                    (row[1]+rt, row[2]+rb, ((row[3]*row[5])+rht)/new_count, ((row[4]*row[5])+rhb)/new_count, new_count, tgl))
    else:
        cur.execute("INSERT INTO histori_harian VALUES (%s, %s, %s, %s, %s, 1)", (tgl, rt, rb, rht, rhb))
    conn.commit()
    
    # Ambil data 3 hari terakhir untuk fitur AI
    cur.execute("SELECT rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    conn.close()
    return rows

def run_prediction(rows):
    if len(rows) < 1: return
    last = rows[0]
    acc3_t, acc3_b = sum(r[0] for r in rows), sum(r[1] for r in rows)
    rh_max = max(last[2], last[3])
    rain_max = max(last[0], last[1])
    
    # Logika Pengaman Ekstrem (Hard Limit)
    if rain_max >= 50 and rh_max >= 90:
        res = "TINGGI"
    else:
        # Gunakan Model Random Forest
        model = joblib.load('model_banjir_tapteng_final.pkl')
        le = joblib.load('label_encoder_final.pkl')
        feat = np.array([[last[0], acc3_t, last[2], last[1], acc3_b, last[3], rh_max, rain_max, max(acc3_t, acc3_b), rh_max]])
        res = le.inverse_transform(model.predict(feat))[0]
    
    if res == "TINGGI":
        msg = (f"🚨 *PERINGATAN DINI BANJIR TAPTENG* 🚨\n\nStatus: *TINGGI (SIAGA)*\n"
               f"📍 Hulu Tukka: Hujan {last[0]}mm, Acc3 {acc3_t}mm\n"
               f"📍 Hulu BTR: Hujan {last[1]}mm, Acc3 {acc3_b}mm\n"
               f"💧 Kelembapan Max: {rh_max}%\n\n"
               f"⚠️ Kondisi berpotensi banjir. SEGERA SIAGA!")
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    rt, rb, rht, rhb = fetch_weather()
    data_rows = update_db_and_get_features(rt, rb, rht, rhb)
    run_prediction(data_rows)