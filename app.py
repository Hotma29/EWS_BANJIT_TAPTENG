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
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS untuk UI yang lebih profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    .status-box {
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stMetric"] label {
        color: #4a4a4a !important; 
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL AI ---
@st.cache_resource
def load_smart_model():
    # Load model Random Forest Murni dengan 5 Fitur
    model = joblib.load('random_forest_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

# --- 3. FUNGSI KIRIM TELEGRAM (Murni AI) ---
def send_telegram_simulation(status, station, features_dict):
    try:
        emoji = "🚨" if status == "TINGGI" else "⚠️"
        
        if status == "TINGGI":
            pesan_himbauan = "PERINGATAN DARURAT: Klasifikasi AI menunjukkan pola cuaca ekstrem. Potensi luapan sungai sangat tinggi. Segera siagakan tim mitigasi!"
        elif status == "SEDANG":
            pesan_himbauan = "WASPADA: Klasifikasi AI menunjukkan pola cuaca memburuk. Pantau pergerakan debit air secara berkala."
        else:
            pesan_himbauan = "AMAN: Pola cuaca normal menurut klasifikasi AI."

        text = (
            f"🧪 *[MODE SIMULASI LABORATORIUM]*\n"
            f"{emoji} *EWS BANJIR TAPTENG: {status}* {emoji}\n\n"
            f"📍 *Titik Pantau Acuan:* {station}\n"
            f"🌧️ *Hujan (1 Jam):* {features_dict['CH']} mm\n"
            f"🌊 *Akumulasi (3 Hari):* {features_dict['CH3']} mm\n"
            f"💧 *Kelembapan Udara:* {features_dict['RH']}%\n"
            f"🌡️ *Suhu Udara:* {features_dict['T2M']}°C\n"
            f"💨 *Kecepatan Angin:* {features_dict['WS10M']} m/s\n\n"
            f"🤖 *Keputusan:* Murni Prediksi Random Forest\n"
            f"📢 *Info:* {pesan_himbauan}\n\n"
            f"⚠️ _Pesan ini simulasi otomatis dari Dashboard EWS._"
        )
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"Gagal mengirim notifikasi Telegram: {e}")

# --- 4. FUNGSI HELPER API OPEN-METEO (Live Demo Sidebar) ---
def fetch_api_only():
    try:
        url_t = "https://api.open-meteo.com/v1/forecast?latitude=1.699608&longitude=98.910028&current=precipitation,relative_humidity_2m,temperature_2m,wind_speed_10m&timezone=Asia/Jakarta"
        url_s = "https://api.open-meteo.com/v1/forecast?latitude=1.541647&longitude=98.993431&current=precipitation,relative_humidity_2m,temperature_2m,wind_speed_10m&timezone=Asia/Jakarta"
        
        res_t = requests.get(url_t, timeout=10).json()
        res_s = requests.get(url_s, timeout=10).json()
        return res_t['current'], res_s['current']
    except:
        return None, None

# --- 5. SIDEBAR KONTROL ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rs = fetch_api_only()
        if rt and rs:
            st.success("Koneksi Open-Meteo Berhasil!")
            
            st.markdown("### 📍 Hulu Tukka")
            st.write(f"🌧️ **Hujan (1 Jam):** {rt['precipitation']} mm")
            st.write(f"💧 **Kelembapan:** {rt['relative_humidity_2m']}%")
            st.write(f"🌡️ **Suhu Udara:** {rt['temperature_2m']}°C")
            st.write(f"💨 **Kecepatan Angin:** {rt['wind_speed_10m']} m/s")
            st.divider()
            
            st.markdown("### 📍 Hulu Sibabangun")
            st.write(f"🌧️ **Hujan (1 Jam):** {rs['precipitation']} mm")
            st.write(f"💧 **Kelembapan:** {rs['relative_humidity_2m']}%")
            st.write(f"🌡️ **Suhu Udara:** {rs['temperature_2m']}°C")
            st.write(f"💨 **Kecepatan Angin:** {rs['wind_speed_10m']} m/s")
            
            st.caption("ℹ️ Data simulasi API ini tidak disimpan ke database.")

# --- 6. MAIN DASHBOARD ---
st.title("🌊 Sistem Peringatan Dini Potensi Banjir Bandang (AI Murni)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = """
            SELECT tanggal, created_at, prediksi, 
                   ch_tuk, ch_tuk_latest, ch3_tuk, rh_tuk, t2m_tuk, ws10m_tuk,
                   ch_sbbn, ch_sbbn_latest, ch3_sbbn, rh_sbbn, t2m_sbbn, ws10m_sbbn
            FROM histori_harian ORDER BY tanggal DESC, created_at DESC LIMIT 7
        """
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            
            status = latest['prediksi']
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; opacity: 0.9;">STATUS RISIKO BANJIR SAAT INI (PREDIKSI RANDOM FOREST):</p>
                    <h1 style="font-size: 4.5rem; margin: 0; letter-spacing: 3px;">{status}</h1>
                    <p>Pembaruan Terakhir: {latest['created_at']} WIB</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            
            col_t, col_s = st.columns(2)
            
            with col_t:
                st.subheader("📍 Pemantauan Hulu Tukka")
                t1, t2, t3 = st.columns(3)
                t1.metric("Hujan 1 Jam Terakhir", f"{latest['ch_tuk_latest']} mm")
                t2.metric("Total Hujan Hari Ini", f"{latest['ch_tuk']} mm")
                t3.metric("Akumulasi 3 Hari", f"{latest['ch3_tuk']} mm")
                
                t4, t5, t6 = st.columns(3)
                t4.metric("Kelembapan (RH)", f"{latest['rh_tuk']} %")
                t5.metric("Suhu Udara (T2M)", f"{latest['t2m_tuk']} °C")
                t6.metric("Angin (WS10M)", f"{latest['ws10m_tuk']} m/s")

            with col_s:
                st.subheader("📍 Pemantauan Hulu Sibabangun")
                s1, s2, s3 = st.columns(3)
                s1.metric("Hujan 1 Jam Terakhir", f"{latest['ch_sbbn_latest']} mm")
                s2.metric("Total Hujan Hari Ini", f"{latest['ch_sbbn']} mm")
                s3.metric("Akumulasi 3 Hari", f"{latest['ch3_sbbn']} mm")
                
                s4, s5, s6 = st.columns(3)
                s4.metric("Kelembapan (RH)", f"{latest['rh_sbbn']} %")
                s5.metric("Suhu Udara (T2M)", f"{latest['t2m_sbbn']} °C")
                s6.metric("Angin (WS10M)", f"{latest['ws10m_sbbn']} m/s")

            st.markdown("---")
            st.subheader("📊 Perbandingan Hujan Harian (7 Hari Terakhir)")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['ch_tuk'], name='Total Hulu Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['ch_sbbn'], name='Total Hulu Sibabangun', marker_color='#ef5350'))
            fig.update_layout(barmode='group', template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Belum ada data histori harian di Database dengan struktur terbaru.")
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.header("🧪 Laboratorium AI Murni (Simulasi 5 Parameter)")
    st.write("Masukkan angka secara manual untuk melihat bagaimana model Random Forest mengklasifikasikan status banjir.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Input Hulu Tukka")
        t_chl = st.number_input("Hujan 1 Jam / Instan (mm)", 0.0, 100.0, 0.0, key="t_chl")
        t_ch = st.number_input("Hujan Hari Ini (mm)", 0.0, 300.0, 10.0, key="t_ch")
        t_ch3 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="t_ch3")
        t_rh = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="t_rh")
        t_t2m = st.number_input("Suhu Udara / T2M (°C)", 10.0, 45.0, 26.0, key="t_t2m")
        t_ws = st.number_input("Kecepatan Angin / WS10M (m/s)", 0.0, 30.0, 1.5, key="t_ws")
        
    with col_b:
        st.markdown("### 📍 Input Hulu Sibabangun")
        s_chl = st.number_input("Hujan 1 Jam / Instan (mm)", 0.0, 100.0, 0.0, key="s_chl")
        s_ch = st.number_input("Hujan Hari Ini (mm) ", 0.0, 300.0, 5.0, key="s_ch")
        s_ch3 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="s_ch3")
        s_rh = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="s_rh")
        s_t2m = st.number_input("Suhu Udara / T2M (°C) ", 10.0, 45.0, 28.0, key="s_t2m")
        s_ws = st.number_input("Kecepatan Angin / WS10M (m/s) ", 0.0, 30.0, 2.0, key="s_ws")

    if st.button("🚀 Jalankan Inferensi AI", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()
            
            # 1. CARI LOKASI TERPARAH (REPRESENTATIF) Murni berdasarkan Hujan (Max Harian vs CH3)
            skor_tukka = max(t_ch, t_ch3)
            skor_sibabangun = max(s_ch, s_ch3)
            
            if skor_tukka >= skor_sibabangun:
                rep_station = "Hulu Tukka"
                # Fitur CH diambil dari Hujan 1 Jam (Instan) sesuai logika di worker.py
                features_dict = {'CH': t_chl, 'CH3': t_ch3, 'RH': t_rh, 'T2M': t_t2m, 'WS10M': t_ws}
            else:
                rep_station = "Hulu Sibabangun"
                features_dict = {'CH': s_chl, 'CH3': s_ch3, 'RH': s_rh, 'T2M': s_t2m, 'WS10M': s_ws}
            
            # 2. LOGIKA KEPUTUSAN MURNI AI (Tanpa If-Else Cuaca)
            features_list = ['CH', 'CH3', 'RH', 'T2M', 'WS10M']
            input_df = pd.DataFrame([features_dict], columns=features_list)
            
            prediksi_kelas = model.predict(input_df)[0]
            status_sim = le.inverse_transform([int(prediksi_kelas)])[0].upper()
            
            if status_sim == "TINGGI":
                pesan_mitigasi = "⚠️ BAHAYA:  Model AI mengklasifikasikan kondisi meteorologii bahaya tinggi."
            elif status_sim == "SEDANG":
                pesan_mitigasi = "👀 WASPADA: Model AI mendeteksi cuaca yang mengarah ke potensi meluapnya air."
            else:
                pesan_mitigasi = "✅ AMAN: Model AI mengklasifikasikan kondisi meteorologi dalam batas normal."

            st.markdown("---")
            st.info(f"🔍 **Analisis Spasial:** Parameter Representatif (REP) diekstrak dari **{rep_station}** karena intensitas akumulasi hujannya lebih parah.")
            
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 30px; border-radius: 20px; text-align: center; color: white;">
                    <h1 style="font-size: 5rem; margin:10px 0;">{status_sim}</h1>
                    <p style="font-size: 1.5rem; font-weight: 500;">{pesan_mitigasi}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- NOTIFIKASI TELEGRAM ---
            if status_sim in ["SEDANG", "TINGGI"]:
                send_telegram_simulation(status_sim, rep_station, features_dict)
                st.toast("🚨 Notifikasi Bahaya Simulasi berhasil dikirim ke Telegram!", icon="🚨")

        except Exception as e:
            st.error(f"Gagal memproses analisis AI: {e}")
