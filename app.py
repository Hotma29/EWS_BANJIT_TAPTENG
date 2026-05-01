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

# --- FUNGSI DATABASE (Hanya untuk Tab Monitoring) ---
def get_data():
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT created_at, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, prediksi FROM histori_harian ORDER BY created_at DESC LIMIT 50"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Gagal terhubung ke Database: {e}")
        return pd.DataFrame()

# --- FUNGSI TELEGRAM ---
def send_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# --- UI UTAMA ---
st.title("🌊 Dashboard EWS Banjir Tapanuli Tengah")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Lab Simulasi Sidang"])

# --- TAB 1: MONITORING (Menampilkan data dari Database) ---
with tab1:
    df_data = get_data()
    if not df_data.empty:
        latest = df_data.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Hujan Tukka", f"{latest['rain_tuk']} mm")
        with c2: st.metric("Hujan B. Toru", f"{latest['rain_btr']} mm")
        with c3: st.metric("RH Terakhir", f"{latest['rh_tuk_avg']} %")
        with c4: 
            st.metric("Status Prediksi", latest['prediksi'])
        
        st.subheader("Log Data Terakhir")
        st.dataframe(df_data, use_container_width=True)
    else:
        st.info("Menunggu data masuk dari worker GitHub...")

# --- TAB 2: SIMULASI (INPUT MANUAL & TIDAK MASUK DATABASE) ---
with tab2:
    st.header("🧪 Simulasi Input Model AI (Demo Mode)")
    st.write("Input data di bawah ini hanya untuk pengujian model. Data **TIDAK** akan disimpan ke database.")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Data Hujan")
            sim_rain_tukka = st.number_input("Hujan Tukka (mm)", min_value=0.0, step=1.0)
            sim_rain_btoru = st.number_input("Hujan Batang Toru (mm)", min_value=0.0, step=1.0)
        
        with col2:
            st.subheader("📍 Data Kelembapan (rH)")
            sim_rh_tukka = st.number_input("rH Tukka (%)", min_value=0.0, max_value=100.0, value=80.0)
            sim_rh_btoru = st.number_input("rH Batang Toru (%)", min_value=0.0, max_value=100.0, value=80.0)

        with col3:
            st.subheader("📍 Faktor Akumulasi")
            # Ini input manual tambahan yang Abang minta
            sim_akumulasi_3hr = st.number_input("Akumulasi Hujan 3 Hari (mm)", min_value=0.0, step=1.0, help="Total hujan dalam 3 hari terakhir")

    # Tombol Prediksi
    if st.button("🚀 Jalankan Simulasi & Kirim Notifikasi"):
        try:
            # 1. Load Model
            model = joblib.load('model_banjir_tapteng_final.pkl')
            encoder = joblib.load('label_encoder_final.pkl')

            # 2. Siapkan Data Input
            # Catatan: Pastikan urutan & jumlah kolom ini sama dengan saat Abang melatih model (training).
            # Jika model Abang tidak pakai kolom 'akumulasi', hapus dari list columns di bawah.
            data_input = pd.DataFrame([[sim_rain_tukka, sim_rain_btoru, sim_rh_tukka, sim_rh_btoru]], 
                                     columns=['curah_hujan_tukka', 'curah_hujan_batangtoru', 'hum_tukka', 'hum_batangtoru'])
            
            # 3. Eksekusi Prediksi AI
            res_num = model.predict(data_input)
            res_label = encoder.inverse_transform(res_num)[0]

            # 4. Tampilkan Hasil di Layar
            st.markdown("---")
            if res_label == "TINGGI":
                st.error(f"### HASIL SIMULASI: {res_label}")
                # Kirim ke Telegram
                pesan = (
                    f"⚠️ *DEMO SIMULASI TINGGI*\n"
                    f"Hujan Hulu: {sim_rain_tukka + sim_rain_btoru} mm\n"
                    f"Akumulasi 3 Hari: {sim_akumulasi_3hr} mm\n"
                    f"Status: *SIAGA BANJIR*"
                )
                send_telegram(pesan)
                st.success("✅ Notifikasi simulasi berhasil dikirim ke Telegram!")
            else:
                st.success(f"### HASIL SIMULASI: {res_label}")
                st.info("Status Aman. Notifikasi Telegram tidak dikirim.")
            
            st.caption("Peringatan: Data di atas hanya simulasi dan tidak disimpan di tabel histori_harian.")

        except Exception as e:
            st.error(f"Gagal melakukan simulasi: {e}")
