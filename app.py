import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import os
import requests
import joblib
import numpy as np

DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide")

# --- SIDEBAR: TOMBOL LIVE TEST ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Live Test)"):
        # Hanya monitoring, TIDAK simpan ke DB
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric").json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric").json()
        
        st.success("Koneksi API Berhasil!")
        st.write(f"**Tukka:** {res_t.get('rain',{}).get('1h',0)}mm | {res_t['main']['humidity']}%")
        st.write(f"**BTR:** {res_b.get('rain',{}).get('1h',0)}mm | {res_b['main']['humidity']}%")
        st.caption("Data ini hanya tampilan sementara (tidak masuk database).")

# --- MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["📊 Real-Time Monitoring", "🧪 Mode Simulasi (Demo AI)"])

# --- TAB 1: MONITORING ---
with tab1:
    st.title("🌊 Monitoring Banjir Tapanuli Tengah")
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql_query("SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7", conn)
        conn.close()

        if not df.empty:
            now = df.iloc[0]
            c1, c2, c3 = st.columns(3)
            status_color = "red" if now['prediksi'] == "TINGGI" else "green"
            c1.markdown(f"### Status Terkini: :{status_color}[{now['prediksi']}]")
            c2.metric("Total Hujan Hari Ini", f"{max(now['rain_tuk'], now['rain_btr'])} mm")
            c3.metric("Data Masuk", f"{int(now['entry_count'])} Jam")

            # BAR CHART: Hari Ini vs Hari Sebelumnya
            st.subheader("📈 Akumulasi Curah Hujan Harian")
            df_plot = df.head(3).copy()
            df_plot['Hujan_Max'] = df_plot.apply(lambda x: max(x['rain_tuk'], x['rain_btr']), axis=1)
            df_plot = df_plot.sort_values('tanggal')
            
            fig = px.bar(df_plot, x='tanggal', y='Hujan_Max', 
                         title="Perbandingan Total Hujan (mm)", text_auto=True,
                         color='Hujan_Max', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Gagal memuat data monitoring.")

# --- TAB 2: SIMULASI (UNTUK DEMO SIDANG) ---
with tab2:
    st.title("🧪 Laboratorium Simulasi Random Forest")
    st.info("Gunakan mode ini untuk mendemonstrasikan cara kerja algoritma ke dosen penguji.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Input Hulu Tukka**")
        sim_rt = st.number_input("Hujan Jam Ini (mm)", 0.0, 200.0, 5.0, key="a1")
        sim_at = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 120.0, key="a2")
        sim_rht = st.slider("Kelembapan (%)", 0, 100, 85, key="a3")
    with col_b:
        st.markdown("**Input Hulu Batang Toru**")
        sim_rb = st.number_input("Hujan Jam Ini (mm)", 0.0, 200.0, 2.0, key="b1")
        sim_ab = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 40.0, key="b2")
        sim_rhb = st.slider("Kelembapan (%)", 0, 100, 70, key="b3")

    if st.button("🚀 JALANKAN PREDIKSI SIMULASI", type="primary", use_container_width=True):
        # Hitung parameter Max untuk simulasi
        s_rain_max = max(sim_rt, sim_rb)
        s_acc_max = max(sim_at, sim_ab)
        s_rh_max = max(sim_rht, sim_rhb)
        
        # 1. Cek Guardrail (Logika AND Abang)
        if (s_rain_max >= 50) and s_rh_max >= 90:
            final_res = "TINGGI (Trigger: Guardrail Ekstrem)"
        else:
            # 2. Panggil AI
            try:
                model = joblib.load('model_banjir_tapteng_final.pkl')
                le = joblib.load('label_encoder_final.pkl')
                # Susun 10 Fitur
                s_feat = np.array([[sim_rt, sim_at, sim_rht, sim_rb, sim_ab, sim_rhb, 
                                    s_rh_max, s_rain_max, s_acc_max, s_rh_max]])
                final_res = le.inverse_transform(model.predict(s_feat))[0] + " (Berdasarkan Random Forest)"
            except:
                final_res = "Gagal memuat model .pkl"
        
        st.subheader(f"Hasil Prediksi: {final_res}")
