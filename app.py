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
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL (Update Nama File) ---
@st.cache_resource
def load_smart_model():
    # Pastikan file ini ada di folder yang sama di GitHub
    model = joblib.load('model_ews_flood.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

# --- 3. FUNGSI HELPER API ---
def fetch_api_only():
    try:
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric").json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric").json()
        return res_t, res_b
    except:
        return None, None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rb = fetch_api_only()
        if rt and rb:
            st.success("Koneksi API Berhasil!")
            st.write("**Hulu Tukka:**")
            st.write(f"- Hujan: {rt.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- Kelembapan: {rt['main']['humidity']}%")
            st.write("**Hulu Batang Toru:**")
            st.write(f"- Hujan: {rb.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- Kelembapan: {rb['main']['humidity']}%")
            st.caption("Info: Data demo ini tidak masuk ke database.")

# --- 5. MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System (Zone Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (10 Fitur)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            status = latest['prediksi']
            
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; margin-bottom: 5px;">STATUS SISTEM SAAT INI:</p>
                    <h1 style="font-size: 3.5rem; margin: 0;">{status}</h1>
                    <p style="margin-top: 10px;">Tanggal Data: {latest['tanggal']} | Update: {latest['created_at']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Hujan Hulu Tukka", f"{latest['rain_tuk']:.2f} mm")
            with c2: st.metric("RH Hulu Tukka ", f"{latest['rh_tuk_avg']:.1f}%")
            with c3: st.metric("Hujan Hulu Batang Toru", f"{latest['rain_btr']:.2f} mm")
            with c4: st.metric("RH Hulu Batang Toru", f"{latest['rh_btr_avg']:.1f}%")

            st.subheader("📈 Tren Curah Hujan 7 Hari")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Hulu Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='Hulu Batang toru', marker_color='#ef5350'))
            fig.update_layout(barmode='group', height=350)
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Gagal terhubung ke database. Pastikan DB_URL sudah benar.")

with tab2:
    st.header("🧪 Laboratorium Simulasi AI")
    st.info("Simulasi ini menggunakan model Random Forest 10 Fitur dengan Integritas Spasial.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Lokasi 1: Hulu Tukka")
        s1 = st.number_input("Hujan Harian (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="sim2")
        s3 = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim3")
        
    with col_b:
        st.markdown("### 📍 Lokasi 2: Hulu Batang Toru")
        s4 = st.number_input("Hujan Harian (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="sim5")
        s6 = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim6")

    if st.button("🚀 Jalankan Prediksi AI", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()

            # --- FEATURE ENGINEERING (Sesuai Logika Training) ---
            # 1. Hitung Skor masing-masing stasiun
            skor_tukka = max(s1, s2)
            skor_btr = max(s4, s5)
            
            # 2. Seleksi 1 Paket Data Representative (Fitur 8, 9, 10)
            if skor_tukka >= skor_btr:
                rain_max, rain3_max, rh_max = s1, s2, s3
            else:
                rain_max, rain3_max, rh_max = s4, s5, s6
            
            # 3. Hitung Rata-rata RH (Fitur 7)
            rata_rh = (s3 + s6) / 2

            # 4. Susun 10 Fitur dalam urutan yang tepat
            features = [
                'RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 
                'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 
                'RATA-RATA_RH', 
                'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX'
            ]
            
            input_vals = [[s1, s2, s3, s4, s5, s6, rata_rh, rain_max, rain3_max, rh_max]]
            input_df = pd.DataFrame(input_vals, columns=features)

            # 5. Prediksi
            pred_numeric = model.predict(input_df)
            status_sim = le.inverse_transform(pred_numeric)[0]
            
            prob = model.predict_proba(input_df)
            conf = np.max(prob) * 100

            st.markdown("---")
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 25px; border-radius: 15px; text-align: center; color: white;">
                    <h3>HASIL ANALISIS RANDOM FOREST</h3>
                    <h1 style="font-size: 4rem; margin:0;">{status_sim}</h1>
                    <p style="font-size: 1.2rem;">Confidence Level: {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Info Tambahan untuk Bahan Sidang
            with st.expander("🔍 Lihat Detail Input AI"):
                st.write(f"**Stasiun Representative:** {'Tukka' if skor_tukka >= skor_btr else 'Batang Toru'}")
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Error Model: {e}. Pastikan file model_ews_flood.pkl dan label_encoder.pkl sudah di-upload.")
