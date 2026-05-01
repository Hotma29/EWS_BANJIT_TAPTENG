import streamlit as st
import pandas as pd
import psycopg2
import joblib
import requests

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="EWS Banjir Tapteng", page_icon="🌊", layout="wide")
DB_URL = st.secrets["SUPABASE_DB_URL"] # Gunakan Port 6543 (Pooler)
TELEGRAM_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
CHAT_ID = st.secrets["TELEGRAM_CHANNEL_ID"]

# --- 2. FUNGSI DATABASE ---
def get_data():
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT * FROM histori_harian ORDER BY created_at DESC LIMIT 50"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Koneksi Database Gagal: {e}")
        return pd.DataFrame()

def send_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"})

# --- 3. UI DASHBOARD ---
st.title("🌊 Dashboard EWS Banjir Tapanuli Tengah")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Monitoring Real-Time", "🧪 Lab Simulasi Sidang"])

# --- TAB 1: MONITORING ---
with tab1:
    df_data = get_data()
    if not df_data.empty:
        latest = df_data.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Hujan Tukka (1.72, 98.92)", f"{latest['rain_tuk']} mm")
        with c2: st.metric("Hujan B. Toru (1.55, 99.08)", f"{latest['rain_btr']} mm")
        with c3: 
            avg_rh = (latest['rh_tuk_avg'] + latest['rh_btr_avg']) / 2
            st.metric("RH Rata-rata", f"{avg_rh:.1f} %")
        with c4:
            warna = "🔴" if latest['prediksi'] == "TINGGI" else "🟢"
            st.metric("Status Sistem", f"{warna} {latest['prediksi']}")
        st.subheader("Log Riwayat Data Real-Time")
        st.dataframe(df_data, use_container_width=True)
    else:
        st.info("Menunggu data masuk dari GitHub Actions...")

# --- TAB 2: SIMULASI (LOGIKA HYBRID & KONSISTENSI DATASET) ---
with tab2:
    st.header("🧪 Simulasi Prediksi (Aturan Pakar + AI)")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            s_rain_tuk = st.number_input("Hujan Tukka (mm)", value=10.0)
            s_rain_btr = st.number_input("Hujan B. Toru (mm)", value=5.0)
        with col2:
            s_rh_tuk = st.number_input("rH Tukka (%)", value=80.0)
            s_rh_btr = st.number_input("rH B. Toru (%)", value=75.0)
        with col3:
            s_rain3_tuk = st.number_input("Akumulasi 3H Tukka (mm)", value=30.0)
            s_rain3_btr = st.number_input("Akumulasi 3H B. Toru (mm)", value=25.0)

    # Feature Engineering Otomatis
    max_rain_t = max(s_rain_tuk, s_rain_btr)
    max_rain3 = max(s_rain3_tuk, s_rain3_btr)
    max_rh = max(s_rh_tuk, s_rh_btr)
    rh_avg = (s_rh_tuk + s_rh_btr) / 2

    if st.button("🚀 Jalankan Prediksi Simulasi"):
        try:
            # A. LOGIKA OVERRIDE (50/90 RULE)
            if max_rain_t >= 50 and max_rh >= 90:
                res_label = "TINGGI"
                metode = "Override (Kondisi Ekstrem)"
            else:
                # B. LOGIKA RANDOM FOREST
                model = joblib.load('model_banjir_tapteng_final.pkl')
                encoder = joblib.load('label_encoder_final.pkl')
                
                # Mapping 10 Kolom sesuai dataset Abang
                input_df = pd.DataFrame([[
                    s_rain_tuk, s_rain3_tuk, s_rh_tuk, s_rain_btr, 
                    s_rain3_btr, s_rh_btr, rh_avg, max_rain_t, max_rain3, max_rh
                ]], columns=['RAIN_TUKKA', 'RAIN3_TUKKA', 'RH_TUKKA', 'RAIN_BTR', 
                            'RAIN3_BTR', 'RHBTR', 'RATA-RATA_RH', 'MAX_RAIN_T', 
                            'MAX_RAIN3', 'MAX_RH'])
                
                res_num = model.predict(input_df)
                res_label = encoder.inverse_transform(res_num)[0]
                metode = "Random Forest AI"

            st.divider()
            st.subheader(f"Hasil: {res_label}")
            st.caption(f"Metode: {metode}")
            if res_label == "TINGGI":
                st.error("⚠️ STATUS SIAGA BANJIR!")
                send_telegram(f"🚨 SIMULASI: TINGGI!\nMetode: {metode}\nHujan: {max_rain_t}mm\nRH: {max_rh}%")
            else:
                st.success("✅ Kondisi Aman")
        except Exception as e:
            st.error(f"Error: {e}")
