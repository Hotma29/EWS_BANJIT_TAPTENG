import streamlit as st
import pandas as pd
import psycopg2
import joblib
import numpy as np
import os
import plotly.express as px

DB_URL = os.getenv("SUPABASE_DB_URL")

st.set_page_config(page_title="EWS BANJIR TAPTENG", layout="wide")
st.title("🌊 Dashboard EWS Banjir Tapanuli Tengah")

tab1, tab2 = st.tabs(["📈 Monitoring Real-Time", "🧪 Lab Simulasi Sidang"])

with tab1:
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql_query("SELECT * FROM histori_harian ORDER BY tanggal DESC LIMIT 7", conn)
        conn.close()
        
        if not df.empty:
            st.subheader("Tren Curah Hujan 7 Hari Terakhir")
            fig = px.line(df, x='tanggal', y=['rain_tuk', 'rain_btr'], markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)
        else:
            st.info("Menunggu data otomatis ditarik dari Cloud...")
    except:
        st.error("Gagal terhubung ke Database Supabase.")

with tab2:
    st.header("Simulasi Model Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        s_rt = st.number_input("Hujan Tukka (mm)", 0.0, 500.0, 10.0)
        s_at = st.number_input("Acc 3 Hari Tukka (mm)", 0.0, 1000.0, 30.0)
        s_rht = st.slider("RH Tukka (%)", 0, 100, 80)
    with col2:
        s_rb = st.number_input("Hujan BTR (mm)", 0.0, 500.0, 5.0)
        s_ab = st.number_input("Acc 3 Hari BTR (mm)", 0.0, 1000.0, 15.0)
        s_rhb = st.slider("RH BTR (%)", 0, 100, 75)
    
    if st.button("Jalankan Prediksi Simulasi"):
        model = joblib.load('model_banjir_tapteng_final.pkl')
        le = joblib.load('label_encoder_final.pkl')
        
        r_max = max(s_rt, s_rb)
        rh_max = max(s_rht, s_rhb)
        
        # Logika Pengaman Ekstrem
        if r_max >= 50 and rh_max >= 90:
            res = "TINGGI"
            st.error(f"HASIL: {res} (Picu Pengaman Ekstrem)")
        else:
            feat = np.array([[s_rt, s_at, s_rht, s_rb, s_ab, s_rhb, rh_max, r_max, max(s_at, s_ab), rh_max]])
            res = le.inverse_transform(model.predict(feat))[0]
            st.success(f"HASIL PREDIKSI MODEL: {res}")