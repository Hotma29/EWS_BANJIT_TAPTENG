import os
import requests
import psycopg2
import joblib
import pandas as pd
from datetime import datetime

# Ambil Secret dari GitHub Actions
API_KEY = os.getenv("OPENWEATHER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Koordinat Lokasi
LOCS = {
    "Tukka": {"lat": 1.7371, "lon": 98.8681},
    "Batang Toru": {"lat": 1.5035, "lon": 99.0667}
}

def get_weather_data(lat, lon):
    """Mengambil data cuaca dengan penanganan error yang lebih kuat"""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        res = response.json()
        
        if response.status_code == 200:
            # Mengambil curah hujan (jika ada, jika tidak 0.0)
            rain = res.get('rain', {}).get('1h', 0.0)
            # Mengambil kelembapan dengan aman
            humidity = res.get('main', {}).get('humidity', 0)
            return rain, humidity
        else:
            print(f"⚠️ API Error ({response.status_code}): {res.get('message', 'Unknown error')}")
            return 0.0, 0
    except Exception as e:
        print(f"❌ Koneksi Error: {e}")
        return 0.0, 0

def send_telegram(message):
    """Mengirim notifikasi ke Telegram Channel"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"⚠️ Gagal kirim Telegram: {e}")

def run_worker():
    # 1. Ambil data cuaca terbaru
    rt, rht = get_weather_data(LOCS["Tukka"]["lat"], LOCS["Tukka"]["lon"])
    rb, rhb = get_weather_data(LOCS["Batang Toru"]["lat"], LOCS["Batang Toru"]["lon"])
    
    print(f"Data Terambil: Tukka={rt}mm, Batang Toru={rb}mm")

    # 2. Load Model AI
    try:
        model = joblib.load('model_banjir_tapteng_final.pkl')
        encoder = joblib.load('label_encoder_final.pkl')
    except Exception as e:
        print(f"❌ Gagal load model: {e}")
        return

    # 3. Prediksi
    # Sesuaikan urutan kolom dengan training model Abang
    input_data = pd.DataFrame([[rt, rb, rht, rhb]], 
                             columns=['curah_hujan_tukka', 'curah_hujan_batangtoru', 'hum_tukka', 'hum_batangtoru'])
    
    prediction_numeric = model.predict(input_data)
    prediction_label = encoder.inverse_transform(prediction_numeric)[0]

    # 4. Simpan ke Supabase
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        query = """
            INSERT INTO histori_harian (tanggal, curah_hujan_tukka, curah_hujan_batangtoru, prediksi)
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(query, (datetime.now(), rt, rb, prediction_label))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Data berhasil disimpan ke Supabase")
    except Exception as e:
        print(f"❌ Gagal simpan ke DB: {e}")

    # 5. Kirim Notifikasi jika Bahaya
    if prediction_label.upper() == "TINGGI":
        msg = (
            f"⚠️ *PERINGATAN DINI BANJIR TAPTENG*\n\n"
            f"Status: *TINGGI*\n"
            f"Curah Hujan Tukka: {rt} mm\n"
            f"Curah Hujan Batang Toru: {rb} mm\n"
            f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Harap waspada bagi warga di sekitar aliran sungai!"
        )
        send_telegram(msg)

if __name__ == "__main__":
    run_worker()
