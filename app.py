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
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# Custom CSS
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
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_smart_model():
    model = joblib.load('model_banjir_mine.pkl')
    le = joblib.load('label_encoder_mine.pkl')
    return model, le

# --- 3. FUNGSI KIRIM TELEGRAM (KHUSUS SIMULASI) ---
def send_telegram_simulation(status, station, rain, rh, conf):
    try:
        text = (
            f"🧪 *[MODE SIMULASI LABORATORIUM]*\n"
            f"🚨 *STATUS: {status}* 🚨\n\n"
            f"Lokasi Representatif: {station}\n"
            f"Input Hujan: {rain} mm\n"
            f"Input Kelembapan: {rh} %\n"
            f"Confidence AI: {conf:.2f}%\n\n"
            f"⚠️ _Pesan ini dikirim otomatis melalui fitur simulasi dashboard._"
        )
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"Gagal mengirim notifikasi Telegram: {e}")

# --- 4. FUNGSI HELPER API (Live Demo Sidebar) ---
def fetch_api_only():
    try:
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric", timeout=10).json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric", timeout=10).json()
        return res_t, res_b
    except:
        return None, None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rb = fetch_api_only()
        if rt and rb:
            st.success("Koneksi API Berhasil!")
            st.markdown("### 📍 Hulu Tukka")
            st.write(f"- **Hujan (1 Jam):** {rt.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rt['main']['humidity']}%")
            st.divider()
            st.markdown("### 📍 Hulu Batang Toru")
            st.write(f"- **Hujan (1 Jam):** {rb.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rb['main']['humidity']}%")
            st.caption("ℹ️ Data ini tidak disimpan ke database.")

# --- 6. MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System (Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = """
            SELECT tanggal, created_at, prediksi, rain_tuk, rain_tuk_latest, rh_tuk_latest,
                   rain_btr, rain_btr_latest, rh_btr_latest
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
                    <p style="font-size: 1.2rem; opacity: 0.9;">STATUS RISIKO BANJIR SAAT INI:</p>
                    <h1 style="font-size: 4.5rem; margin: 0; letter-spacing: 3px;">{status}</h1>
                    <p>Pembaruan Terakhir: {latest['created_at']} WIB</p>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("📍 Pemantauan Hulu Tukka")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Rain Harian", f"{latest['rain_tuk']:.2f} mm")
            with c2: st.metric("Rain Terakhir", f"{latest['rain_tuk_latest']:.2f} mm")
            with c3: st.metric("RH Terakhir", f"{latest['rh_tuk_latest']:.1f} %")

            st.subheader("📍 Pemantauan Hulu Batang Toru")
            c4, c5, c6 = st.columns(3)
            with c4: st.metric("Rain Harian ", f"{latest['rain_btr']:.2f} mm")
            with c5: st.metric("Rain Terakhir ", f"{latest['rain_btr_latest']:.2f} mm")
            with c6: st.metric("RH Terakhir ", f"{latest['rh_btr_latest']:.1f} %")

            st.markdown("---")
            st.subheader("📈 Tren Akumulasi Curah Hujan (7 Hari)")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Tukka', line=dict(color='#1976d2', width=4), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='Batang Toru', line=dict(color='#ef5350', width=4), mode='lines+markers'))
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.header("🧪 Simulasi Analisis AI (Random Forest)")
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
            skor_tukka = max(s1, s2)
            skor_btr = max(s4, s5)
            
            if skor_tukka >= skor_btr:
                rep_station, rain_max, rain3_max, rh_max = "Hulu Tukka", s1, s2, s3
            else:
                rep_station, rain_max, rain3_max, rh_max = "Hulu Batang Toru", s4, s5, s6
            
            rata_rh = (s3 + s6) / 2
            features = ['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'RAIN_MAX', 'RAIN3_MAX', 'RH_MAX']
            input_df = pd.DataFrame([[s1, s2, s3, s4, s5, s6, rata_rh, rain_max, rain3_max, rh_max]], columns=features)
            
            pred = model.predict(input_df)
            status_sim = le.inverse_transform(pred)[0]
            conf = np.max(model.predict_proba(input_df)) * 100

            st.markdown("---")
            st.info(f"🔍 **Analisis Spasial:** Representasi hulu bahaya adalah **{rep_station}**.")
            
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 30px; border-radius: 20px; text-align: center; color: white;">
                    <h1 style="font-size: 5rem; margin:10px 0;">{status_sim}</h1>
                    <p style="font-size: 1.5rem;">Confidence Level: {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # --- LOGIKA NOTIFIKASI TELEGRAM JIKA TINGGI ---
            if status_sim == "TINGGI":
                send_telegram_simulation(status_sim, rep_station, rain_max, rh_max, conf)
                st.toast("🚨 Notifikasi Bahaya telah dikirim ke Telegram!", icon="🚨")

        except Exception as e:
            st.error(f"Gagal memproses model AI: {e}")
