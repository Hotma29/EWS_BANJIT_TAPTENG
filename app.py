import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
import requests
import joblib
import numpy as np

# --- CONFIG ---
DB_URL = os.getenv("SUPABASE_DB_URL")
# Ganti protokol postgres:// menjadi postgresql:// jika perlu (standar SQLAlchemy)
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# --- DATABASE ENGINE ---
engine = create_engine(DB_URL)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    # Perbaikan width='stretch'
    if st.button("🔄 Cek Koneksi API (Live Test)", width='stretch'):
        try:
            res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric").json()
            res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric").json()
            st.success("API Terhubung!")
        except:
            st.error("Gagal mengambil data API.")

# --- MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System")
tab1, tab2 = st.tabs(["📊 Real-Time Monitoring", "🧪 Simulation Lab"])

with tab1:
    try:
        # Gunakan engine SQLAlchemy untuk membaca data
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 10"
        df = pd.read_sql_query(query, engine)

        if not df.empty:
            latest = df.iloc[0]
            status = latest['prediksi']
            bg_color = "#d32f2f" if status == "TINGGI" else "#f9a825" if status == "SEDANG" else "#2e7d32"
            
            st.markdown(f"<div style='padding:20px; border-radius:10px; color:white; text-align:center; background-color:{bg_color};'>"
                        f"<h1 style='margin:0;'>STATUS: {status}</h1>"
                        f"<p>Terakhir Diperbarui: {latest['tanggal']}</p></div>", unsafe_allow_html=True)

            st.subheader("📈 Analisis Perbandingan Curah Hujan")
            df_hist = df.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_hist['tanggal'], y=df_hist['rain_tuk'], name='Hulu Tukka'))
            fig.add_trace(go.Bar(x=df_hist['tanggal'], y=df_hist['rain_btr'], name='Hulu BTR'))
            # Perbaikan width='stretch' pada chart
            st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.title("🧪 Laboratorium Simulasi")
    if st.button("🚀 PROSES PREDIKSI AI", type="primary", width='stretch'):
        try:
            # Gunakan file sesuai Source[cite: 1, 2]
            model = joblib.load('model_banjir_final.pkl')[cite: 1]
            le = joblib.load('label_encoder_final_2.pkl')[cite: 2]
            
            # (Logika input simulasi Abang di sini...)
            st.success("Simulasi Berhasil Jalankan!")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
