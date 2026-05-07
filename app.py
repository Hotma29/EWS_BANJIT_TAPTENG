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

# Custom CSS untuk tampilan Dashboard yang Profesional
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
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-bottom: 4px solid #1976d2;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_smart_model():
    # Memuat model Random Forest dan Label Encoder
    model = joblib.load('model_banjir_mine.pkl')
    le = joblib.load('label_encoder_mine.pkl')
    return model, le

# --- 3. FUNGSI HELPER API (Untuk Live Demo di Sidebar) ---
def fetch_api_only():
    try:
        # Menarik data mentah dari OpenWeather untuk validasi sistem
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric", timeout=10).json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric", timeout=10).json()
        return res_t, res_b
    except:
        return None, None

# --- 4. SIDEBAR (PANEL KONTROL & LIVE DEMO) ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    st.info("Fitur ini digunakan untuk membuktikan sistem dapat terhubung ke stasiun cuaca secara real-time.")
    
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rb = fetch_api_only()
        if rt and rb:
            st.success("Koneksi API Berhasil!")
            
            # Tampilan Live Data Hulu Tukka
            st.markdown("### 📍 Hulu Tukka")
            st.write(f"- **Kondisi:** {rt['weather'][0]['description'].capitalize()}")
            st.write(f"- **Hujan (1 Jam):** {rt.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rt['main']['humidity']}%")
            
            st.divider()
            
            # Tampilan Live Data Hulu Batang Toru
            st.markdown("### 📍 Hulu Batang Toru")
            st.write(f"- **Kondisi:** {rb['weather'][0]['description'].capitalize()}")
            st.write(f"- **Hujan (1 Jam):** {rb.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rb['main']['humidity']}%")
            
            st.caption("⚠️ Data Sidebar ini hanya untuk pengecekan dan tidak disimpan ke database.")
        else:
            st.error("Gagal terhubung ke API. Cek koneksi atau API Key.")

# --- 5. MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System (Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

# --- TAB 1: MONITORING REAL-TIME ---
with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        # Memanggil kolom harian (Daily) dan kolom terbaru (Latest)
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
            
            # Menentukan Warna Status Berdasarkan Hasil AI
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; margin-bottom: 5px; opacity: 0.9;">STATUS RISIKO BANJIR SAAT INI:</p>
                    <h1 style="font-size: 4.5rem; margin: 0; letter-spacing: 3px;">{status}</h1>
                    <p style="margin-top: 15px; font-weight: normal;">Pembaruan Terakhir: {latest['created_at']} WIB</p>
                </div>
            """, unsafe_allow_html=True)

            # --- METRIK HULU TUKKA ---
            st.subheader("📍 Pemantauan Hulu Tukka")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Rain Harian (Daily)", f"{latest['rain_tuk']:.2f} mm", help="Total akumulasi hujan dari jam 00:00")
            with c2: st.metric("Rain Terakhir (1h)", f"{latest['rain_tuk_latest']:.2f} mm", help="Volume hujan pada tarikan terakhir")
            with c3: st.metric("RH Terakhir", f"{latest['rh_tuk_latest']:.1f} %", help="Kelembapan udara saat ini")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- METRIK HULU BATANG TORU ---
            st.subheader("📍 Pemantauan Hulu Batang Toru")
            c4, c5, c6 = st.columns(3)
            with c4: st.metric("Rain Harian (Daily) ", f"{latest['rain_btr']:.2f} mm")
            with c5: st.metric("Rain Terakhir (1h) ", f"{latest['rain_btr_latest']:.2f} mm")
            with c6: st.metric("RH Terakhir ", f"{latest['rh_btr_latest']:.1f} %")

            st.markdown("---")
            
            # --- GRAFIK TREN (DAILY RAIN) ---
            st.subheader("📈 Tren Akumulasi Curah Hujan (7 Hari Terakhir)")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Hulu Tukka', line=dict(color='#1976d2', width=4), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='Hulu Batang Toru', line=dict(color='#ef5350', width=4), mode='lines+markers'))
            fig.update_layout(template="plotly_white", height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Data belum tersedia. Silakan jalankan sistem worker terlebih dahulu.")

    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

# --- TAB 2: LABORATORIUM AI ---
with tab2:
    st.header("🧪 Simulasi Analisis AI (Random Forest)")
    st.write("Gunakan fitur ini untuk mensimulasikan kondisi ekstrem hulu guna melihat respon model AI.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Input Hulu Tukka")
        s1 = st.number_input("Hujan Hari Ini (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="sim2")
        s3 = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim3")
        
    with col_b:
        st.markdown("### 📍 Input Hulu Batang Toru")
        s4 = st.number_input("Hujan Hari Ini (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="sim5")
        s6 = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim6")

    if st.button("🚀 Jalankan Analisis AI", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()

            # Logika Integritas Spasial (Seleksi Fitur 8, 9, 10)
            skor_tukka = max(s1, s2)
            skor_btr = max(s4, s5)
            
            if skor_tukka >= skor_btr:
                rep_station = "Hulu Tukka"
                rain_max, rain3_max, rh_max = s1, s2, s3
            else:
                rep_station = "Hulu Batang Toru"
                rain_max, rain3_max, rh_max = s4, s5, s6
            
            rata_rh = (s3 + s6) / 2

            # Menyiapkan Input 10 Fitur
            features = ['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX']
            input_df = pd.DataFrame([[s1, s2, s3, s4, s5, s6, rata_rh, rain_max, rain3_max, rh_max]], columns=features)
            
            # Eksekusi Prediksi
            pred_numeric = model.predict(input_df)
            status_sim = le.inverse_transform(pred_numeric)[0]
            conf = np.max(model.predict_proba(input_df)) * 100

            st.markdown("---")
            
            # Hasil Analisis Integritas Spasial
            st.info(f"🔍 **Analisis Spasial:** Representasi Hulu bahaya saat ini adalah **{rep_station}**.")
            
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 30px; border-radius: 20px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <h2 style="margin:0; opacity:0.8;">HASIL ANALISIS MODEL RANDOM FOREST</h2>
                    <h1 style="font-size: 5.5rem; margin:10px 0;">{status_sim}</h1>
                    <p style="font-size: 1.5rem; font-weight: bold;">Tingkat Keyakinan (Confidence): {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memproses model AI: {e}")
