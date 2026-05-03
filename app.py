import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
import requests
import joblib
import numpy as np

# --- CONFIG ---
DB_URL = os.getenv("SUPABASE_DB_URL")
# Perbaikan otomatis untuk standar SQLAlchemy
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

# --- DATABASE ENGINE ---
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

engine = get_engine()

# --- MAIN DASHBOARD ---
st.title("🌊 Smart Flood Early Warning System")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Simulasi AI"])

with tab1:
    try:
        # Gunakan engine SQLAlchemy (Fix Warning Pandas)
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 10"
        df = pd.read_sql_query(query, engine)

        if not df.empty:
            latest = df.iloc[0]
            status = latest['prediksi']
            
            # Tampilan Status Card
            bg_color = "#d32f2f" if status == "TINGGI" else "#f9a825" if status == "SEDANG" else "#2e7d32"
            st.markdown(f"""
                <div style='padding:20px; border-radius:10px; color:white; text-align:center; background-color:{bg_color};'>
                    <h1 style='margin:0;'>STATUS: {status}</h1>
                    <p>Update Terakhir: {latest['tanggal']}</p>
                </div>
            """, unsafe_allow_html=True)

            # Detail Per Hulu
            st.subheader("📍 Kondisi Detail Hulu")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hujan Tukka", f"{latest['rain_tuk']} mm")
            c2.metric("RH Tukka", f"{latest['rh_tuk_avg']:.1f}%")
            c3.metric("Hujan BTR", f"{latest['rain_btr']} mm")
            c4.metric("RH BTR", f"{latest['rh_btr_avg']:.1f}%")

            # Grafik Plotly (Fix use_container_width -> width='stretch')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['tanggal'], y=df['rain_tuk'], name='Tukka'))
            fig.add_trace(go.Bar(x=df['tanggal'], y=df['rain_btr'], name='BTR'))
            st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")

with tab2:
    st.title("🧪 Simulasi Prediksi AI")
    col_a, col_b = st.columns(2)
    with col_a:
        s1 = st.number_input("Hujan Tukka (mm)", 0.0, 300.0, 10.0)
        s2 = st.number_input("Akumulasi 3hr Tukka (mm)", 0.0, 500.0, 50.0)
        s3 = st.slider("RH Tukka (%)", 0, 100, 80)
    with col_b:
        s4 = st.number_input("Hujan BTR (mm)", 0.0, 300.0, 5.0)
        s5 = st.number_input("Akumulasi 3hr BTR (mm)", 0.0, 500.0, 30.0)
        s6 = st.slider("RH BTR (%)", 0, 100, 75)

    if st.button("🚀 PREDIKSI SEKARANG", width='stretch', type="primary"):
        try:
            # Load Model & Encoder[cite: 1, 2]
            model = joblib.load('model_banjir_final.pkl')[cite: 1]
            le = joblib.load('label_encoder_final.pkl')[cite: 2]

            # Feature Engineering
            # Harus DataFrame dengan nama kolom persis saat training!
            input_df = pd.DataFrame([{
                'RAIN_TUKKA': s1, 'RAIN3_TUKKA': s2, 'RH_TUKKA': s3,
                'RAIN_BTR': s4, 'RAIN3_BTR': s5, 'RHBTR': s6,
                'RAIN_MAX': max(s1, s4), 'RAIN3_MAX': max(s2, s5), 'RH_MAX': max(s3, s6)
            }])

            # Prediksi
            res = model.predict(input_df)
            status_sim = le.inverse_transform(res)[0][cite: 2]
            st.success(f"### Hasil Prediksi AI: {status_sim}")
        except Exception as e:
            st.error(f"Error AI: {e}")
