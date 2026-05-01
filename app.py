import streamlit as st
import pandas as pd
import psycopg2
import joblib
import requests
from datetime import datetime

# --- CONFIG DASHBOARD ---
st.set_page_config(page_title="EWS Banjir Tapteng", page_icon="🌊", layout="wide")

# --- LOAD SECRETS (Streamlit Cloud) ---
DB_URL = st.secrets["SUPABASE_DB_URL"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
CHAT_ID = st.secrets["TELEGRAM_CHANNEL_ID"]

# --- FUNGSI DATABASE ---
def get_data():
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT created_at, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, prediksi FROM histori_harian ORDER BY created_at DESC LIMIT 50"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Gagal terhubung ke Database Supabase: {e}")
        return pd.DataFrame()

# --- FUNGSI TELEGRAM ---
def send_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# --- UI DASHBOARD ---
st.title("🌊 Dashboard EWS Banjir Tapanuli Tengah")
st.markdown("Sistem Monitoring dan Prediksi Banjir berbasis Machine Learning.")

tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Lab Simulasi Sidang"])

# --- TAB 1: MONITORING REAL-TIME ---
with tab1:
    df_data = get_data()
    
    if not df_data.empty:
        # Metrik Utama (Data Terakhir)
        latest = df_data.iloc[0]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Curah Hujan Tukka", f"{latest['rain_tuk']} mm")
        with col2:
            st.metric("Curah Hujan Batang Toru", f"{latest['rain_btr']} mm")
        with col3:
            status_color = "🔴" if latest['prediksi'] == "TINGGI" else "🟢"
            st.metric("Status Prediksi", f"{status_color} {latest['prediksi']}")

        st.subheader("Riwayat Data Terakhir")
        st.dataframe(df_data, use_container_width=True)
        
        # Grafik Sederhana
        st.subheader("Tren Curah Hujan")
        st.line_chart(df_data.set_index('created_at')[['rain_tuk', 'rain_btr']])
    else:
        st.warning("Belum ada data di database. Pastikan worker.py sudah berjalan di GitHub Actions.")

# --- TAB 2: LAB SIMULASI SIDANG ---
with tab2:
    st.header("🧪 Simulasi Input Model AI")
    st.info("Gunakan bagian ini untuk mendemonstrasikan cara kerja model AI di depan dosen penguji.")

    col_a, col_b = st.columns(2)
    with col_a:
        in_tukka = st.number_input("Input Curah Hujan Tukka (mm)", min_value=0.0, step=0.1)
        in_hum_tukka = st.number_input("Input Kelembapan Tukka (%)", min_value=0.0, max_value=100.0, value=80.0)
    with col_b:
        in_btoru = st.number_input("Input Curah Hujan Batang Toru (mm)", min_value=0.0, step=0.1)
        in_hum_btoru = st.number_input("Input Kelembapan Batang Toru (%)", min_value=0.0, max_value=100.0, value=80.0)

    if st.button("Jalankan Prediksi Simulasi"):
        try:
            # Load Model
            model = joblib.load('model_banjir_tapteng_final.pkl')
            encoder = joblib.load('label_encoder_final.pkl')
            
            # Predict
            features = pd.DataFrame([[in_tukka, in_btoru, in_hum_tukka, in_hum_btoru]], 
                                   columns=['curah_hujan_tukka', 'curah_hujan_batangtoru', 'hum_tukka', 'hum_batangtoru'])
            res_num = model.predict(features)
            res_label = encoder.inverse_transform(res_num)[0]
            
            # Tampilkan Hasil
            if res_label == "TINGGI":
                st.error(f"### HASIL PREDIKSI: {res_label}")
                # Kirim ke Telegram jika status TINGGI
                msg = (
                    f"🚨 *SIMULASI BANJIR TAPTENG*\n"
                    f"Status: *{res_label}*\n"
                    f"Tukka: {in_tukka}mm | B.Toru: {in_btoru}mm\n"
                    f"Notifikasi ini dikirim otomatis oleh sistem simulasi."
                )
                send_telegram(msg)
                st.success("✅ Notifikasi peringatan telah dikirim ke Telegram!")
            else:
                st.success(f"### HASIL PREDIKSI: {res_label}")
                st.info("Status AMAN/RENDAH. Tidak ada notifikasi yang dikirim.")
                
        except Exception as e:
            st.error(f"Gagal menjalankan simulasi: {e}")
            st.info("Pastikan file .pkl sudah di-upload ke GitHub.")
