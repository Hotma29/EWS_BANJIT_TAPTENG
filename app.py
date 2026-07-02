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

# Custom CSS untuk tampilan lebih presisi dan kompak
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .status-box { padding: 20px; border-radius: 10px; color: white; text-align: center; font-weight: bold; margin-bottom: 20px; }
    div[data-testid="stMetric"] { background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
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
        msg = (f"🧪 *[MODE SIMULASI]*\n{emoji} *EWS BANJIR TAPTENG: {status}*\n\n"
               f"📍 *Titik:* {station}\n🌧️ *Hujan Hari Ini:* {rain:.1f} mm\n🌊 *Hujan Akumulasi (3 Hari):* {rain3:.1f} mm\n"
               f"💧 *Kelembapan:* {rh}%\n⚙️ *Trigger:* {logika}")
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                     params={"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        st.error(f"Gagal notif: {e}")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Demo)", use_container_width=True):
        st.info("Fitur Live API aktif.")

# --- 5. MAIN DASHBOARD ---
st.title("🌊 Sistem Peringatan Dini Banjir Tapteng")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT tanggal, prediksi, rain_tuk, rain_tuk_latest, rh_tuk_latest, rain_sbbn, rain_sbbn_latest, rh_sbbn_latest, created_at FROM histori_harian ORDER BY tanggal DESC LIMIT 3"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            rain3_tuk = df_db['rain_tuk'].sum()
            rain3_sbbn = df_db['rain_sbbn'].sum()
            
            bg = "#1b5e20" if latest['prediksi'] == "RENDAH" else "#e65100" if latest['prediksi'] == "SEDANG" else "#b71c1c"
            st.markdown(f'<div class="status-box" style="background-color: {bg};"><h2>STATUS: {latest["prediksi"]}</h2><p>Terakhir: {latest["created_at"]}</p></div>', unsafe_allow_html=True)

            st.subheader("📍 Hulu Tukka")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hujan Harian", f"{latest['rain_tuk']:.1f}mm")
            c2.metric("Hujan Instan", f"{latest['rain_tuk_latest']:.1f}mm")
            c3.metric("RH", f"{latest['rh_tuk_latest']:.0f}%")
            c4.metric("Acc 3 Hari", f"{rain3_tuk:.1f}mm")

            st.subheader("📍 Hulu Sibabangun")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Hujan Harian", f"{latest['rain_sbbn']:.1f}mm")
            c6.metric("Hujan Instan", f"{latest['rain_sbbn_latest']:.1f}mm")
            c7.metric("RH", f"{latest['rh_sbbn_latest']:.0f}%")
            c8.metric("Acc 3 Hari", f"{rain3_sbbn:.1f}mm")

            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure([go.Bar(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Tukka'), go.Bar(x=df_plot['tanggal'], y=df_plot['rain_sbbn'], name='Sibabangun')])
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error Database: {e}")

with tab2:
    st.header("🧪 Mode Simulasi Hybrid")
    col_a, col_b = st.columns(2)
    with col_a:
        s_instan1 = st.number_input("Hujan Instan Tukka (mm)", 0.0, 100.0, 0.0)
        s1 = st.number_input("Hujan Harian Tukka (mm)", 0.0, 300.0, 10.0)
        s2 = st.number_input("Acc 3 Hari Tukka (mm)", 0.0, 500.0, 20.0)
        s3 = st.slider("RH Tukka (%)", 0, 100, 80)
    with col_b:
        s_instan2 = st.number_input("Hujan Instan Sibabangun (mm)", 0.0, 100.0, 0.0)
        s4 = st.number_input("Hujan Harian Sibabangun (mm)", 0.0, 300.0, 5.0)
        s5 = st.number_input("Acc 3 Hari Sibabangun (mm)", 0.0, 500.0, 10.0)
        s6 = st.slider("RH Sibabangun (%)", 0, 100, 75)

    if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
        model, le = load_smart_model()
        skor_tukka, skor_sibabangun = max(s1, s2), max(s4, s5)
        rep = "Hulu Tukka" if skor_tukka >= skor_sibabangun else "Hulu Sibabangun"
        r_rep, r3_rep, rh_rep = (s1, s2, s3) if skor_tukka >= skor_sibabangun else (s4, s5, s6)
        
        if s_instan1 >= 10.0 or s_instan2 >= 10.0:
            status, logika = "TINGGI", "Fail-Safe: Hujan Instan >= 10mm"
        elif s_instan1 >= 5.0 or s_instan2 >= 5.0:
            status, logika = "SEDANG", "Fail-Safe: Hujan Instan >= 5mm"
        else:
            prob = model.predict_proba(pd.DataFrame([[r_rep, r3_rep, rh_rep]], columns=['RAIN', 'RAIN3', 'RH']))[0]
            status = "TINGGI" if prob[2] >= 0.3 else "SEDANG" if prob[1] >= 0.4 else "RENDAH"
            logika = "Analisis Random Forest"

        st.markdown(f'<div style="background-color: {"#b71c1c" if status=="TINGGI" else "#e65100" if status=="SEDANG" else "#1b5e20"}; padding: 20px; border-radius: 10px; color: white; text-align: center;"><h1>{status}</h1></div>', unsafe_allow_html=True)
        send_telegram_simulation(status, rep, r_rep, r3_rep, rh_rep, 0, logika, s_instan1>=5.0 or s_instan2>=5.0)
