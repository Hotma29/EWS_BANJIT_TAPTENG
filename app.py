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

# --- 2. FUNGSI HELPER ---
def fetch_api_only():
    """Fungsi khusus unjuk ke dosen (tidak simpan ke DB)"""
    try:
        # Koordinat Tukka & BTR
        res_t = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.72&lon=98.92&appid={API_KEY}&units=metric").json()
        res_b = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=1.55&lon=99.10&appid={API_KEY}&units=metric").json()
        return res_t, res_b
    except:
        return None, None

# --- 3. SIDEBAR (UNTUK DEMO) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4064/4064233.png", width=80)
    st.title("⚙️ Panel Kontrol")
    st.info("Tombol di bawah untuk menunjukkan ke Dosen bahwa sistem bisa menarik data real-time dari API.")
    
    if st.button("🔄 Tarik Data API (Live Demo)", use_container_width=True):
        rt, rb = fetch_api_only()
        if rt and rb:
            st.success("Berhasil terkoneksi API!")
            st.write("**Hulu Tukka:**")
            st.write(f"- Hujan: {rt.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- Kelembapan: {rt['main']['humidity']}%")
            st.write("**Hulu Batang Toru:**")
            st.write(f"- Hujan: {rb.get('rain',{}).get('1h', 0.0)} mm")
            st.write(f"- Kelembapan: {rb['main']['humidity']}%")
            st.caption("Catatan: Data ini tidak dimasukkan ke database.")

# --- 4. MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Simulasi AI"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            latest = df.iloc[0]
            status = latest['prediksi']
            
            # Warna status
            bg_color = "#1b5e20" if status == "RENDAH" else "#e65100" if status == "SEDANG" else "#b71c1c"
            
            # Status Card
            st.markdown(f"""
                <div class="status-box" style="background-color: {bg_color};">
                    <p style="font-size: 1.2rem; margin-bottom: 5px;">STATUS SISTEM SAAT INI:</p>
                    <h1 style="font-size: 3.5rem; margin: 0;">{status}</h1>
                    <p style="margin-top: 10px;">Update Terakhir: {latest['tanggal']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Metrik Utama
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Hujan Tukka (1j)", f"{latest['rain_tuk']} mm")
            with c2: st.metric("RH Tukka", f"{latest['rh_tuk_avg']:.1f}%")
            with c3: st.metric("Hujan BTR (1j)", f"{latest['rain_btr']} mm")
            with c4: st.metric("RH BTR", f"{latest['rh_btr_avg']:.1f}%")

            # Chart Perbandingan (Dikecilkan ukurannya)
            st.subheader("📈 Riwayat Curah Hujan 7 Hari Terakhir")
            df_plot = df.sort_values('tanggal')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_tuk'], name='Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df_plot['tanggal'], y=df_plot['rain_btr'], name='BTR', marker_color='#ef5350'))
            
            fig.update_layout(
                barmode='group',
                height=350, # Ukuran chart dikecilkan
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memuat data dari database.")

with tab2:
    st.header("🧪 Laboratorium Simulasi")
    st.write("Gunakan bagian ini untuk mensimulasikan nilai input ke model Random Forest.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hulu Tukka**")
        s1 = st.number_input("Hujan Per Jam (mm)", 0.0, 300.0, 10.0, key="sim1")
        s2 = st.number_input("Akumulasi 3 Hari (mm)", 0.0, 500.0, 50.0, key="sim2")
        s3 = st.slider("Kelembapan Tukka (%)", 0, 100, 80, key="sim3")
    with col_b:
        st.markdown("**Hulu Batang Toru**")
        s4 = st.number_input("Hujan Per Jam (mm) ", 0.0, 300.0, 5.0, key="sim4")
        s5 = st.number_input("Akumulasi 3 Hari (mm) ", 0.0, 500.0, 30.0, key="sim5")
        s6 = st.slider("Kelembapan BTR (%)", 0, 100, 75, key="sim6")

    if st.button("🚀 Jalankan Prediksi AI", type="primary", use_container_width=True):
        try:
            # 1. Load Model (PASTIKAN NAMA FILE BENAR DI GITHUB)
            model = joblib.load('model_ews.pkl')
            le = joblib.load('label_encoder_ews.pkl')

            # 2. Hitung Fitur MAX Otomatis
            input_df = pd.DataFrame([{
                'RAIN_TUKKA': s1, 'RAIN3_TUKKA': s2, 'RH_TUKKA': s3,
                'RAIN_BTR': s4, 'RAIN3_BTR': s5, 'RHBTR': s6,
                'RAIN_MAX': max(s1, s4), 
                'RAIN3_MAX': max(s2, s5), 
                'RH_MAX': max(s3, s6)
            }])

            # 3. Prediksi
            res = model.predict(input_df)
            status_sim = le.inverse_transform(res)[0]
            
            st.success(f"### Hasil Analisis AI: **{status_sim}**")
            
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
