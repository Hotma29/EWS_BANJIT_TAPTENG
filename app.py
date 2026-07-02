import streamlit as st
import plotly.graph_objects as go
import psycopg2
import os
import requests
import joblib
import numpy as np
import pandas as pd

# --- 1. KONFIGURASI ---
DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .status-box {
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_smart_model():
    model = joblib.load('model_banjirrrr.pkl')
    le = joblib.load('label_encoderrrr.pkl')
    return model, le

# --- 3. FUNGSI KIRIM TELEGRAM ---
def send_telegram_simulation(status, station, rain, rain3, rh, conf, logika, is_instan):
    try:
        emoji = "🚨" if status == "TINGGI" else "⚠️"
        if status == "SEDANG":
            pesan_himbauan = ("*STATUS: WASPADA (SEDANG)*\n" + 
                             ("Terdeteksi hujan mendadak (instan) lebat di hulu. Waspadai debit air." if is_instan 
                              else "Terdeteksi peningkatan akumulasi air. Kondisi tanah mungkin sudah jenuh."))
        else:
            pesan_himbauan = ("*🚨 STATUS: BAHAYA (TINGGI) 🚨*\n" + 
                             ("Hujan badai mendadak! Risiko BANJIR BANDANG KILAT tinggi." if is_instan 
                              else "Akumulasi curah hujan ekstrem! Kapasitas sungai berpotensi meluap masif. Segera evakuasi!"))

        text = (f"🧪 *[MODE SIMULASI LABORATORIUM]*\n{emoji} *EWS BANJIR TAPTENG: {status}*\n\n"
                f"📍 *Titik:* {station}\n🌧️ *Hujan Hari Ini:* {rain:.1f} mm\n🌊 *Hujan Akumulasi (3 Hari):* {rain3:.1f} mm\n"
                f"💧 *Kelembapan:* {rh}%\n⚙️ *Trigger:* {logika}\n\n📢 *Info:* {pesan_himbauan}")
        
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        st.error(f"Gagal mengirim notifikasi Telegram: {e}")

# --- 4. MAIN DASHBOARD ---
st.title("🌊 Sistem Peringatan Dini Potensi Banjir Bandang (Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        # Query mengambil data 3 hari terakhir untuk kalkulasi RAIN3
        query = """
            SELECT tanggal, rain_tuk, rain_sbbn, rain_tuk_latest, rh_tuk_latest, 
                   rain_sbbn_latest, rh_sbbn_latest, created_at, prediksi
            FROM histori_harian ORDER BY tanggal DESC LIMIT 3
        """
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            
            # Hitung Akumulasi 3 Hari dari data di database
            rain3_tuk = df_db['rain_tuk'].sum()
            rain3_sbbn = df_db['rain_sbbn'].sum()
            
            status = latest['prediksi']
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; opacity: 0.9;">STATUS RISIKO BANJIR SAAT INI:</p>
                    <h1 style="font-size: 4.5rem; margin: 0; letter-spacing: 3px;">{status}</h1>
                    <p>Pembaruan Terakhir: {latest['created_at']} WIB</p>
                </div>
            """, unsafe_allow_html=True)

            # Dashboard Hulu Tukka (4 Kolom)
            st.subheader("📍 Pemantauan Hulu Tukka")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Hujan Total", f"{latest['rain_tuk']:.1f} mm")
            with c2: st.metric("Hujan (1 Jam)", f"{latest['rain_tuk_latest']:.1f} mm")
            with c3: st.metric("RH Terakhir", f"{latest['rh_tuk_latest']:.0f} %")
            with c4: st.metric("Akumulasi (3 Hari)", f"{rain3_tuk:.1f} mm")

            # Dashboard Hulu Sibabangun (4 Kolom)
            st.subheader("📍 Pemantauan Hulu Sibabangun")
            c5, c6, c7, c8 = st.columns(4)
            with c5: st.metric("Hujan Total", f"{latest['rain_sbbn']:.1f} mm")
            with c6: st.metric("Hujan (1 Jam)", f"{latest['rain_sbbn_latest']:.1f} mm")
            with c7: st.metric("RH Terakhir", f"{latest['rh_sbbn_latest']:.0f} %")
            with c8: st.metric("Akumulasi (3 Hari)", f"{rain3_sbbn:.1f} mm")

            st.markdown("---")
            # ... (Lanjutkan dengan kode Plotly chart yang sudah ada)
        else:
            st.warning("Belum ada data histori harian di Database.")
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

# ... (Lanjutkan dengan Tab 2 - Mode Simulasi Hybrid seperti semula)
