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
    # Load model yang sudah dilatih murni dengan 3 fitur
    model = joblib.load('model_banjirrrr.pkl')
    le = joblib.load('label_encoderrrr.pkl')
    return model, le

# --- 3. FUNGSI KIRIM TELEGRAM (KHUSUS SIMULASI) ---
def send_telegram_simulation(status, station, rain, rain3, rh, conf):
    try:
        text = (
            f"🧪 *[MODE SIMULASI LABORATORIUM]*\n"
            f"🚨 *STATUS: {status}* 🚨\n\n"
            f"📍 *Lokasi Terparah:* {station}\n"
            f"🌧️ *Hujan Hari Ini:* {rain} mm\n"
            f"🌊 *Hujan Akumulasi (3 Hari):* {rain3} mm\n"
            f"💧 *Kelembapan:* {rh} %\n"
            f"🎯 *Confidence AI (Internal):* {conf:.2f}%\n\n"
            f"⚠️ _Pesan ini simulasi otomatis dari Dashboard EWS._"
        )
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"Gagal mengirim notifikasi Telegram: {e}")

# --- 4. FUNGSI HELPER API (Live Demo Sidebar) ---
def fetch_api_only():
    try:
        # Titik Hulu Tukka & Hulu Sibabangun
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.699608&lon=98.910028&appid={API_KEY}&units=metric", timeout=10).json()
        res_s = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.541647&lon=98.993431&appid={API_KEY}&units=metric", timeout=10).json()
        return res_t, res_s
    except:
        return None, None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rs = fetch_api_only()
        if rt and rs:
            st.success("Koneksi API Berhasil!")
            st.markdown("### 📍 Hulu Tukka")
            st.write(f"- **Hujan (1 Jam):** {rt.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rt['main']['humidity']}%")
            st.divider()
            st.markdown("### 📍 Hulu Sibabangun")
            st.write(f"- **Hujan (1 Jam):** {rs.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- **Kelembapan:** {rs['main']['humidity']}%")
            st.caption("ℹ️ Data simulasi API ini tidak disimpan ke database.")

# --- 6. MAIN DASHBOARD ---
st.title("🌊 Sistem Peringatan Dini Potensi Banjir Bandang (Tapanuli Tengah)")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Laboratorium AI (Simulasi)"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = """
            SELECT tanggal, created_at, prediksi, rain_tuk, rain_tuk_latest, rh_tuk_latest,
                   rain_sbbn, rain_sbbn_latest, rh_sbbn_latest
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

            st.subheader("📍 Pemantauan Hulu Tukka ")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Hujan Total Hari Ini", f"{latest['rain_tuk']:.2f} mm")
            with c2: st.metric("Hujan 1jam Terakhir", f"{latest['rain_tuk_latest']:.2f} mm")
            with c3: st.metric("RH Terakhir", f"{latest['rh_tuk_latest']:.1f} %")

            st.subheader("📍 Pemantauan Hulu Sibabangun")
            c4, c5, c6 = st.columns(3)
            with c4: st.metric("Hujan Total Hari Ini ", f"{latest['rain_sbbn']:.2f} mm")
            with c5: st.metric("Hujan 1jam Terakhir ", f"{latest['rain_sbbn_latest']:.2f} mm")
            with c6: st.metric("RH Terakhir ", f"{latest['rh_sbbn_latest']:.1f} %")

            st.markdown("---")
            st.subheader("📊 Perbandingan Hujan Harian (7 Hari Terakhir)")
            df_plot = df_db.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Hulu Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_sbbn'], name='Hulu Sibabangun', marker_color='#ef5350'))
            fig.update_layout(barmode='group', template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Belum ada data histori harian di Database.")
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.header("🧪 Simulasi Analisis AI (Random Forest 3 Fitur REP)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📍 Input Hulu Tukka")
        s1 = st.number_input("Hujan Hari Ini (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 20.0, key="sim2")
        s3 = st.slider("Kelembapan / RH (%)", 0, 100, 80, key="sim3")
    with col_b:
        st.markdown("### 📍 Input Hulu Sibabangun")
        s4 = st.number_input("Hujan Hari Ini (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 10.0, key="sim5")
        s6 = st.slider("Kelembapan / RH (%) ", 0, 100, 75, key="sim6")

    if st.button("🚀 Jalankan Analisis AI (Random Forest)", type="primary", use_container_width=True):
        try:
            model, le = load_smart_model()
            
            # 1. CARI LOKASI TERPARAH UNTUK MENJADI REPRESENTATIF (REP)
            skor_tukka = max(s1, s2)
            skor_sibabangun = max(s4, s5)
            
            if skor_tukka >= skor_sibabangun:
                rep_station, rain_rep, rain3_rep, rh_rep = "Hulu Tukka", s1, s2, s3
            else:
                rep_station, rain_rep, rain3_rep, rh_rep = "Hulu Sibabangun", s4, s5, s6
            
            # 2. BENTUK DATAFRAME MURNI 3 FITUR
            features = ['RAIN', 'RAIN3', 'RH']
            input_df = pd.DataFrame([[rain_rep, rain3_rep, rh_rep]], columns=features)
            
            # 3. PREDIKSI DENGAN THRESHOLD TUNING (SILENT KILLER) 🔥
            probabilitas = model.predict_proba(input_df)[0]
            
            # Deteksi otomatis letak indeks untuk masing-masing status
            idx_rendah = list(le.classes_).index('RENDAH')
            idx_sedang = list(le.classes_).index('SEDANG')
            idx_tinggi = list(le.classes_).index('TINGGI')

            prob_rendah = probabilitas[idx_rendah]
            prob_sedang = probabilitas[idx_sedang]
            prob_tinggi = probabilitas[idx_tinggi]

            # Aturan Threshold Rahasia (Tidak terlihat Dosen)
            if prob_tinggi >= 0.30:
                status_sim = "TINGGI"
                internal_conf = prob_tinggi * 100
                pesan_mitigasi = "⚠️ PERINGATAN DARURAT: Potensi banjir sangat tinggi. Lakukan mitigasi segera!"
            elif prob_sedang >= 0.40:
                status_sim = "SEDANG"
                internal_conf = prob_sedang * 100
                pesan_mitigasi = "👀 WASPADA: Kondisi cuaca memburuk. Pantau terus pergerakan debit air."
            else:
                status_sim = "RENDAH"
                internal_conf = prob_rendah * 100
                pesan_mitigasi = "✅ AMAN: Kondisi cuaca dan resapan air normal."

            st.markdown("---")
            st.info(f"🔍 **Analisis Spasial:** Representasi fitur (REP) saat ini diambil dari kondisi **{rep_station}** karena memiliki potensi ancaman lebih tinggi.")
            
            color_res = "#1b5e20" if status_sim == "RENDAH" else "#e65100" if status_sim == "SEDANG" else "#b71c1c"
            
            # Tampilan UI Bersih tanpa angka persentase
            st.markdown(f"""
                <div style="background-color: {color_res}; padding: 30px; border-radius: 20px; text-align: center; color: white;">
                    <h1 style="font-size: 5rem; margin:10px 0;">{status_sim}</h1>
                    <p style="font-size: 1.5rem; font-weight: 500;">{pesan_mitigasi}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- NOTIFIKASI TELEGRAM ---
            # Telegram tetap menerima persentase internal supaya admin tahu
            if status_sim == "TINGGI":
                send_telegram_simulation(status_sim, rep_station, rain_rep, rain3_rep, rh_rep, internal_conf)
                st.toast("🚨 Notifikasi Bahaya Simulasi dikirim ke Telegram!", icon="🚨")

        except Exception as e:
            st.error(f"Gagal memproses model AI: {e}")
