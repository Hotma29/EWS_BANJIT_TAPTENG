import os
import requests
import psycopg2
import joblib
import pandas as pd
import numpy as np
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

def fetch_weather():
    def get_data(lat, lon):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            res = requests.get(url).json()
            return res.get('rain', {}).get('1h', 0.0), res['main']['humidity']
        except:
            return 0.0, 0
    rt, rht = get_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_data(LOCS["BTR"]["lat"], LOCS["BTR"]["lon"])
    return rt, rb, rht, rhb

def run_system():
    wib_now = datetime.utcnow() + timedelta(hours=7)
    tgl = wib_now.strftime('%Y-%m-%d')
    waktu_lengkap = wib_now.strftime('%Y-%m-%d %H:%M:%S')
    
    rt_hour, rb_hour, rht, rhb = fetch_weather()
    
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"Database Error: {e}")
        return

    # 3. UPSERT & AMBIL TOTAL HARIAN TERBARU
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
    conn.commit()

    # 4. AMBIL DATA HISTORI 3 HARI (RAIN3)
    cur.execute("SELECT rain_tuk, rain_btr FROM histori_harian ORDER BY tanggal DESC LIMIT 3")
    rows = cur.fetchall()
    acc3_t = sum(r[0] for r in rows)
    acc3_b = sum(r[1] for r in rows)
    
    # ==========================================
    # 5. FEATURE ENGINEERING (INTEGRITAS SPASIAL)
    # ==========================================
    # Hitung Skor untuk menentukan stasiun representative
    skor_tukka = max(total_hari_ini_tuk, acc3_t)
    skor_btr = max(total_hari_ini_btr, acc3_b)
    
    # Ambil 1 paket data dari stasiun paling berbahaya
    if skor_tukka >= skor_btr:
        rain_max = total_hari_ini_tuk
        rain3_max = acc3_t
        rh_max = rht
    else:
        rain_max = total_hari_ini_btr
        rain3_max = acc3_b
        rh_max = rhb
        
    rata_rh = (rht + rhb) / 2
    
    # --- 6. LOGIKA HYBRID (HARD RULE + AI) ---
    # Update Hard Rule agar sinkron dengan Logika Excel (TINGGI: Rain>=50 OR Rain3>=100 OR (Rain>=40 AND RH>=90))
    if (rain_max >= 50 or rain3_max >= 100 or (rain_max >= 40 and rh_max >= 90)):
        status = "TINGGI"
        logika = f"Hard Rule: Kondisi Ekstrem (Rain: {rain_max}mm, RH: {rh_max}%)"
    else:
        try:
            # Sesuaikan nama file .pkl terbaru Abang
            model = joblib.load('model_ews_flood.pkl')
            le = joblib.load('label_encoder.pkl')
            
            # Urutan fitur harus sama persis dengan saat training
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
            print(f"Prediction Error: {e}")
            status = "RENDAH"
            logika = "Fallback: Error Prediction"

    # 7. UPDATE HASIL KE SUPABASE
    cur.execute("UPDATE histori_harian SET prediksi = %s WHERE tanggal = %s", (status, tgl))
    conn.commit()
    print(f"[{waktu_lengkap}] Status: {status} | Logika: {logika}")

    # 8. TELEGRAM NOTIF
    if status in ["SEDANG", "TINGGI"]:
        emoji = "⚠️" if status == "SEDANG" else "🚨"
        msg = (f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
               f"Status: *{status}*\n"
               f"Total Hujan Harian: {rain_max} mm\n"
               f"Kelembapan: {rh_max} %\n"
               f"Logika: {logika}\n\n"
               f"Harap tetap waspada!")
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"})

    cur.close(); conn.close()

if __name__ == "__main__":
    run_system()
