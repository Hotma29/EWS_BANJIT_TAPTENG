import streamlit as st
import plotly.graph_objects as go
import psycopg2
import os
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- 1. KONFIGURASI ---
DB_URL = os.getenv("SUPABASE_DB_URL")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS
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
    model = joblib.load('random_forest_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

# --- 3. FUNGSI KIRIM TELEGRAM (Simulasi Profesional) ---
def send_telegram_simulation(status, station,features_dict):
    try:
        if status == "TINGGI":
            pesan_himbauan = "BAHAYA: Mohon segera lakukan langkah antisipasi dan evakuasi jika diperlukan!"
        elif status == "SEDANG":
            pesan_himbauan = "WASPADA: Pantau terus kondisi cuaca secara berkala!"

        waktu_simulasi = datetime.utcnow() + timedelta(hours=7)
        waktu_lengkap = waktu_simulasi.strftime('%Y-%m-%d %H:%M:%S')

        text = (
            "*INFORMASI POTENSI BANJIR - KAB. TAPANULI TENGAH*\n"
            "[MODE SIMULASI LABORATORIUM]\n"
            "--------------------------------------------------\n\n"
            f"*Status Prediksi  :* {status}\n"
            f"*Titik Pantauan   :* {station}\n\n"
            "*Data Meteorologi:*\n"
            f"- Curah Hujan (Harian)  : {features_dict['CH']} mm\n"
            f"- Akumulasi Hujan (CH3) : {features_dict['CH3']} mm\n"
            f"- Kelembapan Udara (RH) : {features_dict['RH']} %\n"
            f"- Suhu Udara (T2M)      : {features_dict['T2M']} °C\n"
            f"- Kecepatan Angin       : {features_dict['WS10M']} m/s\n\n"
            f"*Instruksi Mitigasi:*\n{pesan_himbauan}\n\n"
            f"*Waktu Simulasi:* {waktu_lengkap} WIB\n"
            "--------------------------------------------------\n"
            "_Pesan ini dikirim melalui Dashboard Simulasi EWS._"
        )
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"Gagal mengirim notifikasi Telegram: {e}")

# --- 4. FUNGSI HELPER API OPEN-METEO (Live Demo Sidang) ---
def fetch_api_proof():
    try:
        # Menggunakan koordinat Hulu Tukka sebagai sampel pembuktian
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.699608&longitude=98.910028&current=precipitation,relative_humidity_2m,temperature_2m,wind_speed_10m&timezone=Asia/Jakarta"
        
        res = requests.get(url, timeout=10)
        
        # Trik Rahasia: Menghitung waktu respons asli dari server API dalam satuan milidetik (ms)
        response_time = int(res.elapsed.total_seconds() * 1000)
        
        return res.json()['current'], response_time
    except:
        return None, None

# --- 5. SIDEBAR KONTROL ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol API")
    st.caption("Gunakan panel ini saat sidang untuk validasi aliran data.")
    
    if st.button("📡 Uji Koneksi Open-Meteo", type="primary", use_container_width=True):
        rt, resp_time = fetch_api_proof()
        
        if rt:
            waktu_sekarang = (datetime.utcnow() + timedelta(hours=7)).strftime('%d-%m-%Y %H:%M')
            
            st.markdown("### Status API")
            st.markdown("───────────────")
            st.success("✓ Open-Meteo Connected")
            
            st.write("**Response Time :**")
            st.code(f"{resp_time} ms")
            
            st.write("**Last Request :**")
            st.code(f"{waktu_sekarang} WIB")
            
            st.write("**Latitude :**")
            st.code("1.699608")
            
            st.write("**Longitude :**")
            st.code("98.910028")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("### Current API Response")
            st.markdown("───────────────")
            
            st.write("**precipitation :**")
            st.code(f"{rt['precipitation']} mm")
            
            st.write("**RH :**")
            st.code(f"{rt['relative_humidity_2m']} %")
            
            st.write("**Temperature :**")
            st.code(f"{rt['temperature_2m']} °C")
            
            st.write("**Wind :**")
            st.code(f"{rt['wind_speed_10m']} m/s")
            
        else:
            st.error("❌ Gagal terhubung ke server Open-Meteo.")



# --- 6. MAIN DASHBOARD ---
st.title("🌊 Dashboard Monitoring Potensi Banjir Kabupaten Tapanuli Tengah")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Mode Simulasi"])

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
        df_db['created_at'] = pd.to_datetime(df_db['created_at']).dt.strftime('%d-%m-%Y %H:%M:%S')

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
            
            col_t, col_spacer, col_s = st.columns([4.5, 1, 4.5])
            
            with col_t:
                st.markdown("<h3 style='text-align: center; color: #4fc3f7;'>📍 Hulu Tukka</h3>", unsafe_allow_html=True)
                st.write("") 
                
                t1, t2, t3 = st.columns(3)
                t1.metric("Hujan Saat Ini (1 Jam Terakhir)", f"{latest['ch_tuk_latest']} mm")
                t2.metric("Curah Hujan Harian (CH)", f"{latest['ch_tuk']} mm")
                t3.metric("Akumulasi Curah Hujan 3 hari (CH3)", f"{latest['ch3_tuk']} mm")
                
                st.write("") 
                t4, t5, t6 = st.columns(3)
                t4.metric("Kelembapan (RH)", f"{latest['rh_tuk']} %")
                t5.metric("Suhu Udara (T2M)", f"{latest['t2m_tuk']} °C")
                t6.metric("Angin (WS10M)", f"{latest['ws10m_tuk']} m/s")

            with col_spacer:
                st.empty()

            with col_s:
                st.markdown("<h3 style='text-align: center; color: #ff8a65;'>📍 Hulu Sibabangun</h3>", unsafe_allow_html=True)
                st.write("") 
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Hujan Saat Ini (1 Jam Terakhir)", f"{latest['ch_sbbn_latest']} mm")
                s2.metric("Curah Hujan Harian (CH)", f"{latest['ch_sbbn']} mm")
                s3.metric("Akumulasi Curah Hujan 3 hari (CH3)", f"{latest['ch3_sbbn']} mm")
                
                st.write("") 
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
    st.header("🧪 Mode Simulasi (Simulasi 5 Parameter)")
    st.write("Masukkan angka secara manual untuk melihat bagaimana model Random Forest mengklasifikasikan status banjir.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Input Hulu Tukka")
        ch_tuk = st.number_input("Hujan Hari Ini (mm)", 0.0, 300.0, 10.0, key="sim_ch_tuk")
        ch3_tuk = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="sim_ch3_tuk")
        rh_tuk = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim_rh_tuk")
        t2m_tuk = st.number_input("Suhu Udara / T2M (°C)", 10.0, 45.0, 26.0, key="sim_t2m_tuk")
        ws10m_tuk = st.number_input("Kecepatan Angin / WS10M (m/s)", 0.0, 30.0, 1.5, key="sim_ws_tuk")
        
    with col_b:
        st.markdown("### 📍 Input Hulu Sibabangun")
        ch_sbbn = st.number_input("Hujan Hari Ini (mm) ", 0.0, 300.0, 5.0, key="sim_ch_sbbn")
        ch3_sbbn = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="sim_ch3_sbbn")
        rh_sbbn = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim_rh_sbbn")
        t2m_sbbn = st.number_input("Suhu Udara / T2M (°C) ", 10.0, 45.0, 28.0, key="sim_t2m_sbbn")
        ws10m_sbbn = st.number_input("Kecepatan Angin / WS10M (m/s) ", 0.0, 30.0, 2.0, key="sim_ws_sbbn")

    if st.button("🚀 Jalankan", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()
            
            # 1. CARI LOKASI TERPARAH (REPRESENTATIF)
            skor_tukka = max(ch_tuk, ch3_tuk)
            skor_sibabangun = max(ch_sbbn, ch3_sbbn)
            
            if skor_tukka >= skor_sibabangun:
                rep_station = "Hulu Tukka"
                features_dict = {'CH': ch_tuk, 'CH3': ch3_tuk, 'RH': rh_tuk, 'T2M': t2m_tuk, 'WS10M': ws10m_tuk}
            else:
                rep_station = "Hulu Sibabangun"
                features_dict = {'CH': ch_sbbn, 'CH3': ch3_sbbn, 'RH': rh_sbbn, 'T2M': t2m_sbbn, 'WS10M': ws10m_sbbn}
            
            # 2. LOGIKA KEPUTUSAN MURNI AI
            features_list = ['CH', 'CH3', 'RH', 'T2M', 'WS10M']
            input_df = pd.DataFrame([features_dict], columns=features_list)
            
            prediksi_kelas = model.predict(input_df)[0]
            status_sim = le.inverse_transform([int(prediksi_kelas)])[0].upper()
            
            if status_sim == "TINGGI":
                pesan_mitigasi = "⚠️ BAHAYA: Bahaya banjir tinggi. Segera evakuasi!"
            elif status_sim == "SEDANG":
                pesan_mitigasi = "👀 WASPADA: Pantau terus kondisi cuaca secara berkala."
            else:
                pesan_mitigasi = "✅ AMAN: Kondisi cuaca dalam batas normal."

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
                send_telegram_simulation(status_sim, rep_station, chl_rep, features_dict)
                st.toast("🚨 Notifikasi Bahaya Simulasi berhasil dikirim ke Telegram!", icon="🚨")

        except Exception as e:
            st.error(f"Gagal memproses analisis: {e}")
