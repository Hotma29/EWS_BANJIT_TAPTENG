import streamlit as st
import pandas as pd
import psycopg2
import joblib
import requests
import pytz # Untuk konversi waktu ke WIB

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="EWS Banjir Tapteng", page_icon="🌊", layout="wide")
DB_URL = st.secrets["SUPABASE_DB_URL"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
CHAT_ID = st.secrets["TELEGRAM_CHANNEL_ID"]

# --- 2. FUNGSI DATABASE & KONVERSI WAKTU ---
def get_data():
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT * FROM histori_harian ORDER BY created_at DESC LIMIT 50"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # KONVERSI JAM KE WIB
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('Asia/Jakarta')
            df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
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

# --- TAB 1: MONITORING (DIPERBARUI) ---
with tab1:
    df_data = get_data()
    if not df_data.empty:
        latest = df_data.iloc[0]
        
        # Hitung Faktor MAX Secara Real-Time
        max_rain_now = max(float(latest['rain_tuk']), float(latest['rain_btr']))
        max_rh_now = max(float(latest['rh_tuk_avg']), float(latest['rh_btr_avg']))
        
        # METRIK UTAMA (Faktor Banjir)
        st.subheader("⚠️ Parameter Kritis Saat Ini (Faktor Max)")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Max Curah Hujan (Hulu)", f"{max_rain_now} mm")
        with m2: st.metric("Max Kelembapan (rH)", f"{max_rh_now} %")
        with m3:
            warna = "🔴" if latest['prediksi'] == "TINGGI" else "🟢"
            st.metric("Status Sistem", f"{warna} {latest['prediksi']}")

        st.markdown("---")
        st.subheader("📍 Detail Per Lokasi")
        c1, c2, c3 = st.columns(3)
        with c1: st.write(f"**Tukka (1.72, 98.92):** {latest['rain_tuk']} mm")
        with c2: st.write(f"**B. Toru (1.55, 99.08):** {latest['rain_btr']} mm")
        with c3: st.write(f"**Update Terakhir (WIB):** {latest['created_at']}")
        
        st.subheader("Log Riwayat Data (Waktu WIB)")
        st.dataframe(df_data, use_container_width=True)
    else:
        st.info("Menunggu data masuk dari GitHub Actions...")

# --- TAB 2: SIMULASI (TETAP SAMA) ---
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

    if st.button("🚀 Jalankan Prediksi"):
        try:
            if max_rain_t >= 50 and max_rh >= 90:
                res_label = "TINGGI"
                metode = "Override (Kondisi Ekstrem)"
            else:
                model = joblib.load('model_banjir_tapteng_final.pkl')
                encoder = joblib.load('label_encoder_final.pkl')
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
