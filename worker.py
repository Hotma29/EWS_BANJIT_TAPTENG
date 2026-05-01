import os
import requests
import psycopg2
import joblib
import pandas as pd
from datetime import datetime

# 1. KONFIGURASI ENVIRONTMENT (Diambil dari GitHub Secrets)
API_KEY = os.getenv("OPENWEATHER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Koordinat Lokasi Tapteng
LOCS = {
    "Tukka": {"lat": 1.7371, "lon": 98.8681},
    "Batang Toru": {"lat": 1.5035, "lon": 99.0667}
}

def get_weather_data(lat, lon):
    """Mengambil data cuaca dari OpenWeather dengan penanganan error"""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        res = response.json()
        
        if response.status_code == 200:
            # Mengambil curah hujan 1 jam terakhir (default 0.0 jika tidak hujan)
            rain = res.get('rain', {}).get('1h', 0.0)
            # Mengambil kelembapan (humidity)
            humidity = res.get('main', {}).get('humidity', 0)
            return rain, humidity
        else:
            print(f"⚠️ API Error ({response.status_code}): {res.get('message', 'Gagal data')}")
            return 0.0, 0
    except Exception as e:
        print(f"❌ Koneksi Error: {e}")
        return 0.0, 0

def run_worker():
    # --- STEP 1: AMBIL DATA CUACA REAL-TIME ---
    rt, rht = get_weather_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_weather_data(LOCS["Batang Toru"]["lat"], LOCS["Batang Toru"]["lon"])
    
    print(f"Data Terambil - Tukka: {rt}mm, B.Toru: {rb}mm")

    # --- STEP 2: PREDIKSI MENGGUNAKAN MODEL AI ---
    try:
        # Pastikan file .pkl ini sudah ada di root repository GitHub Abang
        model = joblib.load('model_banjir_tapteng_final.pkl')
        encoder = joblib.load('label_encoder_final.pkl')

        # Siapkan DataFrame (Urutan kolom HARUS sama dengan saat training model)
        # Urutan: Rain_Tukka, Rain_BatangToru, Hum_Tukka, Hum_BatangToru
        input_data = pd.DataFrame([[rt, rb, rht, rhb]], 
                                 columns=['curah_hujan_tukka', 'curah_hujan_batangtoru', 'hum_tukka', 'hum_batangtoru'])
        
        # Eksekusi Prediksi
        prediction_numeric = model.predict(input_data)
        prediction_label = encoder.inverse_transform(prediction_numeric)[0]
    except Exception as e:
        print(f"❌ Error AI/Model: {e}")
        prediction_label = "ERROR"

    # --- STEP 3: SIMPAN KE SUPABASE (STRUKTUR TABEL BARU) ---
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # ID dan created_at akan terisi otomatis oleh Supabase
        query = """
            INSERT INTO histori_harian (rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, prediksi)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (rt, rb, rht, rhb, prediction_label))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Database Updated: {prediction_label}")
    except Exception as e:
        print(f"❌ Database Error: {e}")

    # --- STEP 4: NOTIFIKASI TELEGRAM (KHUSUS STATUS TINGGI) ---
    if prediction_label.upper() == "TINGGI":
        msg = (
            f"⚠️ *PERINGATAN DINI BANJIR TAPTENG*\n\n"
            f"Status: *TINGGI*\n"
            f"Hujan Tukka: {rt} mm\n"
            f"Hujan Batang Toru: {rb} mm\n"
            f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Warga di sekitar bantaran sungai harap segera waspada!"
        )
        url_tele = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url_tele, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run_worker()
