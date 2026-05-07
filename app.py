import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import os
import requests
import joblib
import numpy as np

# --- 1. KONFIGURASI ---
DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .status-box {
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_smart_model():
    model = joblib.load('model_banjir_mine.pkl')
    le = joblib.load('label_encoder_mine.pkl')
    return model, le

# --- 3. FUNGSI HELPER API (Live Demo) ---
def fetch_api_only():
    try:
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric", timeout=10).json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric", timeout=10).json()
        return res_t, res_b
    except:
        return None, None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    st.info("Sistem Peringatan Dini Banjir berbasis AI Random Forest.")
    if st.button("🔄 Cek Koneksi API (Live)", use_container_width=True):
        rt, rb = fetch_api_only()
        if rt and rb:
            st.success("API Terhubung!")
            st.write(f"**Tukka:** {rt['weather'][0]['description']}")
            st.write(f"**B. Toru:** {rb['weather'][0]['description']}")
        else:
            st.error("API Gagal Terhubung.")

# --- 5. MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System (Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulation)"])

with tab1:
    try:
        # Koneksi Database
        conn = psycopg2.connect(DB_URL)
        # Pastikan kolom latest dipanggil di sini
        query = """
            SELECT 
                tanggal, created_at, prediksi,
                rain_tuk, rain_tuk_latest, rh_tuk_latest,
                rain_btr, rain_btr_latest, rh_btr_latest
            FROM histori_harian 
            ORDER BY tanggal DESC, created_at DESC LIMIT 7
        """
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            status = latest['prediksi']
            
            # Pengaturan Warna Status
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; margin-bottom: 5px; opacity: 0.9;">STATUS RISIKO BANJIR SAAT INI:</p>
                    <h1 style="font-size: 4rem; margin: 0; letter-spacing: 2px;">{status}</h1>
                    <p style="margin-top: 15px; font-weight: normal;">Pembaruan Terakhir: {latest['created_at']}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- MONITORING HULU TUKKA ---
            st.subheader("📍 Pemantauan Hulu Tukka")
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.metric("Total Hujan Hari Ini", f"{latest['rain_tuk']:.2f} mm", help="Akumulasi hujan sejak jam 00:00 WIB")
            with c2: 
                st.metric("Hujan (Terakhir)", f"{latest['rain_tuk_latest']:.2f} mm", delta_color="inverse", help="Volume hujan 1 jam terakhir")
            with c3: 
                st.metric("Kelembapan (RH)", f"{latest['rh_tuk_latest']:.1f} %", help="Data kelembapan terbaru dari sensor")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- MONITORING HULU BATANG TORU ---
            st.subheader("📍 Pemantauan Hulu Batang Toru")
            c4, c5, c6 = st.columns(3)
            with c4: 
                st.metric("Total Hujan Hari Ini", f"{latest['rain_btr']:.2f} mm")
            with c5: 
                st.metric("Hujan (Terakhir)", f"{latest['rain_btr_latest']:.2f} mm")
            with c6: 
                st.metric("Kelembapan (RH)", f"{latest['rh_btr_latest']:.1f} %")

            st.markdown("---")
            
            # --- GRAFIK TREN ---
            st.subheader("📈 Tren Akumulasi Hujan (7 Hari Terakhir)")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Hulu Tukka', line=dict(color='#1976d2', width=4), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='Hulu Batang Toru', line=dict(color='#ef5350', width=4), mode='lines+markers'))
            fig.update_layout(hovermode="x unified", template="plotly_white", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Data belum tersedia di database. Jalankan worker terlebih dahulu.")

    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.header("🧪 Simulasi Prediksi AI (Random Forest)")
    st.write("Gunakan fitur ini untuk mensimulasikan kondisi ekstrem guna menguji ketangguhan model AI.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Kondisi Hulu Tukka")
        s1 = st.number_input("Hujan Hari Ini (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="sim2")
        s3 = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim3")
        
    with col_b:
        st.markdown("### 📍 Kondisi Hulu Batang Toru")
        s4 = st.number_input("Hujan Hari Ini (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="sim5")
        s6 = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim6")

    if st.button("🚀 Jalankan Analisis AI", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()

            # Feature Engineering Logika Bang Hotma
            skor_tukka = max(s1, s2)
            skor_btr = max(s4, s5)
            
            if skor_tukka >= skor_btr:
                rain_max, rain3_max, rh_max = s1, s2, s3
            else:
                rain_max, rain3_max, rh_max = s4, s5, s6
            
            rata_rh = (s3 + s6) / 2

            features = ['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX']
            input_vals = [[s1, s2, s3, s4, s5, s6, rata_rh, rain_max, rain3_max, rh_max]]
            input_df = pd.DataFrame(input_vals, columns=features)

            pred_numeric = model.predict(input_df)
            status_sim = le.inverse_transform(pred_numeric)[0]
            
            prob = model.predict_proba(input_df)
            conf = np.max(prob) * 100

            st.markdown("---")
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 30px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <h2 style="margin:0; opacity:0.8;">HASIL PREDIKSI MODEL</h2>
                    <h1 style="font-size: 5rem; margin:10px 0;">{status_sim}</h1>
                    <p style="font-size: 1.5rem; font-weight: bold;">Tingkat Keyakinan AI: {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
