import streamlit as st
import pandas as pd
import psycopg2
import joblib
import requests
from datetime import datetime

# --- 1. CONFIG DASHBOARD ---
st.set_page_config(page_title="EWS Banjir Tapteng", page_icon="🌊", layout="wide")

# --- 2. LOAD SECRETS (PASTIKAN SUDAH DIISI DI STREAMLIT CLOUD) ---
# Gunakan URI dari Session Pooler (port 6543) agar stabil
DB_URL = st.secrets["SUPABASE_DB_URL"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
CHAT_ID = st.secrets["TELEGRAM_CHANNEL_ID"]

# --- 3. FUNGSI DATABASE (UNTUK MONITORING) ---
def get_data():
    try:
        conn = psycopg2.connect(DB_URL)
        # Menarik 50 data terbaru dari tabel histori_harian
        query = """
        SELECT created_at, rain_tuk, rain_btr, rh_tuk_avg, rh_btr_avg, prediksi 
        FROM histori_harian 
        ORDER BY created_at DESC LIMIT 50
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Gagal terhubung ke Database: {e}")
        return pd.DataFrame()

# --- 4. FUNGSI NOTIFIKASI TELEGRAM ---
def send_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.warning(f"Gagal mengirim Telegram: {e}")

# --- 5. UI UTAMA ---
st.title("🌊 Dashboard EWS Banjir Tapanuli Tengah")
st.markdown("Sistem Peringatan Dini Berbasis Machine Learning untuk Wilayah Hulu.")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Lab Simulasi Sidang"])

# --- TAB 1: MONITORING REAL-TIME ---
with tab1:
    df_data = get_data()
    if not df_data.empty:
        latest = df_data.iloc[0]
        
        # Metrik Utama
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Hujan Tukka (Hulu)", f"{latest['rain_tuk']} mm")
        with m2: st.metric("Hujan B. Toru (Hulu)", f"{latest['rain_btr']} mm")
        with m3: 
            avg_rh_now = (latest['rh_tuk_avg'] + latest['rh_btr_avg']) / 2
            st.metric("Rata-rata rH", f"{avg_rh_now:.1f} %")
        with m4:
            warna = "🔴" if latest['prediksi'] == "TINGGI" else "🟢"
            st.metric("Status Sistem", f"{warna} {latest['prediksi']}")
        
        st.subheader("Data Log Historis (Worker GitHub)")
        st.dataframe(df_data, use_container_width=True)
    else:
        st.info("Menunggu data masuk dari GitHub Actions. Silakan klik 'Run Workflow' di GitHub.")

# --- TAB 2: LAB SIMULASI SIDANG (KONSISTEN DATASET) ---
with tab2:
    st.header("🧪 Simulasi Prediksi Model AI")
    st.write("Input data secara manual untuk mendemonstrasikan logika prediksi model tanpa menyimpan ke database.")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Hujan Saat Ini")
            s_rain_tuk = st.number_input("Hujan Tukka (mm)", min_value=0.0, step=0.1, value=100.0)
            s_rain_btr = st.number_input("Hujan B. Toru (mm)", min_value=0.0, step=0.1, value=5.0)
        
        with col2:
            st.subheader("📍 Kelembapan (rH)")
            s_rh_tuk = st.number_input("rH Tukka (%)", min_value=0.0, max_value=100.0, value=90.0)
            s_rh_btr = st.number_input("rH B. Toru (%)", min_value=0.0, max_value=100.0, value=85.0)

        with col3:
            st.subheader("📍 Akumulasi 3 Hari")
            s_rain3_tuk = st.number_input("Akumulasi Tukka (mm)", min_value=0.0, step=0.1, value=30.0)
            s_rain3_btr = st.number_input("Akumulasi B. Toru (mm)", min_value=0.0, step=0.1, value=15.0)

    # --- PROSES FEATURE ENGINEERING (Sesuai Logika Dataset Latih) ---
    max_rain_t = max(s_rain_tuk, s_rain_btr)
    max_rain3 = max(s_rain3_tuk, s_rain3_btr)
    max_rh = max(s_rh_tuk, s_rh_btr)
    rh_avg_sim = (s_rh_tuk + s_rh_btr) / 2

    # Tampilkan Ringkasan untuk Dosen
    st.markdown("### 📊 Parameter Hasil Hitung (Features)")
    k1, k2, k3, k4 = st.columns(4)
    k1.info(f"**MAX_RAIN_T:** {max_rain_t} mm")
    k2.info(f"**MAX_RAIN3:** {max_rain3} mm")
    k3.info(f"**MAX_RH:** {max_rh} %")
    k4.info(f"**RH_AVG:** {rh_avg_sim:.1f} %")

    if st.button("🚀 Jalankan Prediksi AI & Demo Notifikasi"):
        try:
            # Load Model & Encoder
            model = joblib.load('model_banjir_tapteng_final.pkl')
            encoder = joblib.load('label_encoder_final.pkl')

            # Menyiapkan DataFrame dengan urutan dan nama kolom yang PERSIS dengan data latih
            # Berdasarkan Error: MAX_RAIN3, MAX_RAIN_T, MAX_RH, RAIN3_BTR, RAIN3_TUKKA
            input_df = pd.DataFrame([[
                max_rain3, 
                max_rain_t, 
                max_rh, 
                s_rain3_btr, 
                s_rain3_tuk
            ]], columns=['MAX_RAIN3', 'MAX_RAIN_T', 'MAX_RH', 'RAIN3_BTR', 'RAIN3_TUKKA'])
            
            # Prediksi
            res_num = model.predict(input_df)
            res_label = encoder.inverse_transform(res_num)[0]

            st.markdown("---")
            if res_label == "TINGGI":
                st.error(f"## HASIL PREDIKSI: {res_label} (BAHAYA)")
                # Demo Notifikasi Telegram
                msg = (
                    f"🚨 *SIMULASI EWS TAPTENG*\n"
                    f"Status: *{res_label}*\n"
                    f"Max Hujan: {max_rain_t} mm\n"
                    f"Akumulasi 3H: {max_rain3} mm\n"
                    f"RH Avg: {rh_avg_sim:.1f} %"
                )
                send_telegram(msg)
                st.success("✅ Pesan peringatan simulasi telah dikirim ke Telegram!")
            else:
                st.success(f"## HASIL PREDIKSI: {res_label} (AMAN)")
                st.info("Kondisi di bawah ambang batas bahaya. Notifikasi tidak dikirim.")
            
            st.caption("Catatan: Data simulasi ini tidak disimpan ke database Supabase.")

        except Exception as e:
            st.error(f"Gagal menjalankan simulasi: {e}")
            st.info("Tips: Pastikan file .pkl sudah di-upload ke repository GitHub.")
