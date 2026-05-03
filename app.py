import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import os
import requests
import joblib
import numpy as np
from datetime import datetime

# --- CONFIG ---
DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .status-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    if st.button("🔄 Cek Koneksi API (Live Test)", use_container_width=True):
        try:
            res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric").json()
            res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric").json()
            
            st.success("API Terhubung!")
            col1, col2 = st.columns(2)
            col1.metric("Tukka", f"{res_t['main']['temp']}°C", f"{res_t['main']['humidity']}% RH")
            col2.metric("BTR", f"{res_b['main']['temp']}°C", f"{res_b['main']['humidity']}% RH")
        except:
            st.error("Gagal mengambil data API.")

# --- MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System")
st.markdown(f"**Wilayah Monitoring:** Tapanuli Tengah (Hulu Tukka & Batang Toru)")

tab1, tab2 = st.tabs(["📊 Real-Time Monitoring", "🧪 Simulation Lab"])

# --- TAB 1: MONITORING ---
with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        # Ambil data lebih banyak untuk grafik histori
        df = pd.read_sql_query("SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 10", conn)
        conn.close()

        if not df.empty:
            latest = df.iloc[0]
            
            # 1. TAMPILAN STATUS (Besar & Berwarna)
            status = latest['prediksi']
            bg_color = "#d32f2f" if status == "TINGGI" else "#f9a825" if status == "SEDANG" else "#2e7d32"
            
            st.markdown(f"""
                <div class="status-card" style="background-color: {bg_color};">
                    <h1 style='margin:0;'>STATUS: {status}</h1>
                    <p style='margin:0;'>Terakhir Diperbarui: {latest['tanggal']}</p>
                </div>
            """, unsafe_allow_html=True)

            # 2. METRIK DETAIL PER HULU
            st.subheader("📍 Kondisi Detail Per Hulu")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hujan Tukka (1h)", f"{latest['rain_tuk']} mm")
            m2.metric("RH Tukka", f"{latest['rh_tuk_avg']:.1f} %")
            m3.metric("Hujan BTR (1h)", f"{latest['rain_btr']} mm")
            m4.metric("RH BTR", f"{latest['rh_btr_avg']:.1f} %")

            # 3. GRAFIK PERBANDINGAN HULU (Plotly)
            st.markdown("---")
            st.subheader("📈 Analisis Perbandingan Curah Hujan")
            
            # Reformat data untuk bar chart berdampingan
            df_hist = df.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_hist['tanggal'], y=df_hist['rain_tuk'], name='Hulu Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_hist['tanggal'], y=df_hist['rain_btr'], name='Hulu Batang Toru', marker_color='#ef5350'))

            fig.update_layout(
                barmode='group', 
                title="Curah Hujan Harian: Tukka vs Batang Toru",
                xaxis_title="Tanggal",
                yaxis_title="Curah Hujan (mm)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

# --- TAB 2: SIMULASI ---
with tab2:
    st.title("🧪 Laboratorium Simulasi Random Forest")
    st.info("Inputkan parameter di bawah ini untuk melihat bagaimana AI memproses data.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🏠 Hulu Tukka")
        sim_rt = st.number_input("Hujan Saat Ini (mm)", 0.0, 200.0, 5.0, key="s1")
        sim_at = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 30.0, key="s2")
        sim_rht = st.slider("Kelembapan Tukka (%)", 0, 100, 85, key="s3")
    with col_b:
        st.subheader("⛰️ Hulu Batang Toru")
        sim_rb = st.number_input("Hujan Saat Ini (mm) ", 0.0, 200.0, 2.0, key="s4")
        sim_ab = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 40.0, key="s5")
        sim_rhb = st.slider("Kelembapan BTR (%)", 0, 100, 70, key="s6")

    if st.button("🚀 PROSES PREDIKSI AI", type="primary", use_container_width=True):
        try:
            # 1. Load Model Terbaru
            model = joblib.load('model_banjir_final.pkl')[cite: 1]
            le = joblib.load('label_encoder_final.pkl')[cite: 2]

            # 2. Hitung Fitur MAX Otomatis (Engineering)
            s_rain_max = max(sim_rt, sim_rb)
            s_acc_max = max(sim_at, sim_ab)
            s_rh_max = max(sim_rht, sim_rhb)

            # 3. Susun 9 Fitur sesuai urutan model
            # Tukka(3), BTR(3), Max(3)
            input_df = pd.DataFrame([{
                'RAIN_TUKKA': sim_rt, 'RAIN3_TUKKA': sim_at, 'RH_TUKKA': sim_rht,
                'RAIN_BTR': sim_rb, 'RAIN3_BTR': sim_ab, 'RHBTR': sim_rhb, # Nama kolom tanpa underscore
                'RAIN_MAX': s_rain_max, 'RAIN3_MAX': s_acc_max, 'RH_MAX': s_rh_max
            }])

            # 4. Eksekusi Prediksi
            res_num = model.predict(input_df)
            res_text = le.inverse_transform(res_num)[0][cite: 2]

            # 5. Tampilan Hasil
            st.success(f"### Hasil Analisis AI: **{res_text}**")
            
            # Penjelasan Logika (Sangat berguna saat sidang)
            with st.expander("ℹ️ Lihat Logika Fitur Engineering"):
                st.write(f"Model menggunakan nilai tertinggi (MAX) sebagai sinyal bahaya utama:")
                st.write(f"- **Max Rainfall:** {s_rain_max} mm")
                st.write(f"- **Max Accumulation:** {s_acc_max} mm")
                st.write(f"- **Max Humidity:** {s_rh_max} %")

        except Exception as e:
            st.error(f"Gagal memproses AI: {e}. Pastikan file .pkl sudah di-upload ke GitHub.")
