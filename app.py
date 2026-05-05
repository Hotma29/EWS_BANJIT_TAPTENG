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

# --- 2. FUNGSI LOAD MODEL (Cached) ---
@st.cache_resource
def load_smart_model():
    # Menggunakan nama file terbaru Abang
    model = joblib.load('model_ews_mytapteng.pkl')
    le = joblib.load('label_encoder_ews_mytapteng.pkl')
    return model, le

# --- 3. FUNGSI HELPER API ---
def fetch_api_only():
    """Fungsi khusus demo API untuk Dosen"""
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
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Simulasi AI (10 Fitur)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            latest = df_db.iloc[0]
            status = latest['prediksi']
            
            # Warna status dinamis
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; margin-bottom: 5px;">STATUS SISTEM SAAT INI:</p>
                    <h1 style="font-size: 3.5rem; margin: 0;">{status}</h1>
                    <p style="margin-top: 10px;">Tanggal Data: {latest['tanggal']} | WIB</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Metrik Utama
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Hujan Tukka", f"{latest['rain_tuk']} mm")
            with c2: st.metric("RH Tukka Avg", f"{latest['rh_tuk_avg']:.1f}%")
            with c3: st.metric("Hujan BTR", f"{latest['rain_btr']} mm")
            with c4: st.metric("RH BTR Avg", f"{latest['rh_btr_avg']:.1f}%")

            # Chart Riwayat
            st.subheader("📈 Tren Curah Hujan 7 Hari")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='BTR', marker_color='#ef5350'))
            
            fig.update_layout(barmode='group', height=350, margin=dict(l=20, r=20, t=30, b=20),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Koneksi Database Terputus.")

with tab2:
    st.header("🧪 Laboratorium Simulasi AI")
    st.info("Fitur ini mensimulasikan 'Otak' AI yang baru dilatih dengan logic RH.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Lokasi 1: Tukka")
        s1 = st.number_input("Curah Hujan (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 50.0, key="sim2")
        s3 = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim3")
        
    with col_b:
        st.markdown("### 📍 Lokasi 2: BTR")
        s4 = st.number_input("Curah Hujan (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 30.0, key="sim5")
        s6 = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim6")

    if st.button("🚀 Prediksi dengan Random Forest", type="primary", use_container_width=True):
        try:
            # 1. Load Model terbaru
            model, le = load_smart_model()

            # 2. FEATURE ENGINEERING (Wajib 10 Fitur sesuai urutan)
            rain_max = max(s1, s4)
            rain3_max = max(s2, s5)
            rh_max = max(s3, s6)
            rata_rh = (s3 + s6) / 2 # Fitur ke-7 yang baru kita latih

            # Susun data dalam DataFrame agar urutan kolom PASTI BENAR
            # Urutan ini harus sama dengan saat training (model.feature_names_in_)
            features = [
                'RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 
                'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 
                'RATA-RATA_RH', 
                'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX'
            ]
            
            input_vals = [[s1, s2, s3, s4, s5, s6, rata_rh, rain_max, rain3_max, rh_max]]
            input_df = pd.DataFrame(input_vals, columns=features)

            # 3. Jalankan Prediksi
            pred_numeric = model.predict(input_df)
            status_sim = le.inverse_transform(pred_numeric)[0]
            
            # Ambil probabilitas/keyakinan
            prob = model.predict_proba(input_df)
            conf = np.max(prob) * 100

            # 4. Tampilkan Hasil
            st.markdown("---")
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h3>HASIL PREDIKSI AI</h3>
                    <h1 style="margin:0;">{status_sim}</h1>
                    <p>Tingkat Keyakinan Model: {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Edukasi Dosen
            if status_sim == "RENDAH" and (s2 > 100 or s5 > 100) and rh_max < 85:
                st.info("💡 **Analisis AI:** Meskipun Akumulasi Hujan tinggi (>100mm), AI tetap memprediksi RENDAH karena Kelembapan (RH) di bawah 85%. Logika ini sesuai dengan pola banjir di Tapteng.")

        except Exception as e:
            st.error(f"Error Model: {e}")
