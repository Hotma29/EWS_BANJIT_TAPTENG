import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
import joblib
import numpy as np

# --- CONFIG ---
DB_URL = os.getenv("SUPABASE_DB_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide", page_icon="🌊")

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

engine = get_engine()

# --- DASHBOARD UI ---
st.title("🌊 Smart EWS Banjir Tapanuli Tengah")
tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Simulasi AI"])

with tab1:
    try:
        query = "SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7"
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            latest = df.iloc[0]
            st.subheader(f"Update Terakhir: {latest['tanggal']}")
            
            # Status Card
            status = latest['prediksi']
            color = "#d32f2f" if status == "TINGGI" else "#f9a825" if status == "SEDANG" else "#2e7d32"
            st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>STATUS: {status}</h1></div>", unsafe_allow_html=True)
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hujan Tukka", f"{latest['rain_tuk']} mm")
            c2.metric("RH Tukka", f"{latest['rh_tuk_avg']:.1f}%")
            c3.metric("Hujan BTR", f"{latest['rain_btr']} mm")
            c4.metric("RH BTR", f"{latest['rh_btr_avg']:.1f}%")

            # Chart Perbandingan
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['tanggal'], y=df['rain_tuk'], name='Tukka', marker_color='#1976d2'))
            fig.add_trace(go.Bar(x=df['tanggal'], y=df['rain_btr'], name='BTR', marker_color='#ef5350'))
            fig.update_layout(title="Perbandingan Curah Hujan (mm)", barmode='group')
            st.plotly_chart(fig, width='stretch')
            
    except Exception as e:
        st.error(f"Koneksi Database Bermasalah: {e}")

with tab2:
    st.header("🧪 Laboratorium Simulasi")
    col_a, col_b = st.columns(2)
    with col_a:
        s1 = st.number_input("Hujan Tukka (mm)", 0.0, 300.0, 10.0)
        s2 = st.number_input("Akumulasi 3hr Tukka (mm)", 0.0, 500.0, 50.0)
        s3 = st.slider("RH Tukka (%)", 0, 100, 80)
    with col_b:
        s4 = st.number_input("Hujan BTR (mm)", 0.0, 300.0, 5.0)
        s5 = st.number_input("Akumulasi 3hr BTR (mm)", 0.0, 500.0, 30.0)
        s6 = st.slider("RH BTR (%)", 0, 100, 75)

    if st.button("Jalankan Prediksi AI", type="primary", width='stretch'):
        try:
            model = joblib.load('model_ews.pkl')[cite: 4]
            le = joblib.load('label_encoder_ews.pkl')[cite: 3]
            
            sim_df = pd.DataFrame([{
                'RAIN_TUKKA': s1, 'RAIN3_TUKKA': s2, 'RH_TUKKA': s3,
                'RAIN_BTR': s4, 'RAIN3_BTR': s5, 'RHBTR': s6,
                'RAIN_MAX': max(s1, s4), 'RAIN3_MAX': max(s2, s5), 'RH_MAX': max(s3, s6)
            }])
            
            res = model.predict(sim_df)
            status_sim = le.inverse_transform(res)[0][cite: 3]
            st.success(f"### Hasil Prediksi AI: {status_sim}")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
