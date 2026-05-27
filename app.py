import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# TAHAP 1: KONFIGURASI HALAMAN & DESAIN UI (HEADER)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; text-align: center; font-family: 'Helvetica Neue', sans-serif; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #2980b9; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #3498db; }
    .safe-box { padding: 20px; border-radius: 10px; background-color: #d4edda; color: #155724; text-align: center; border: 2px solid #c3e6cb; margin-top: 20px; }
    .danger-box { padding: 20px; border-radius: 10px; background-color: #f8d7da; color: #721c24; text-align: center; border: 2px solid #f5c6cb; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 AI Customer Churn Predictor")
st.markdown('<p class="subtitle">Sistem Deteksi Dini Pelanggan yang Berisiko Meninggalkan Perusahaan</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------
# SIDEBAR: INFORMASI FITUR
# ---------------------------------------------------------------------
st.sidebar.header("ℹ️ Informasi Fitur VIP")
st.sidebar.write("Model AI dilatih khusus menggunakan 9 indikator paling krusial (Feature Importance):")
st.sidebar.markdown("""
1. **Satisfaction Score**: Tingkat kepuasan (1-5).
2. **Total Spent**: Total uang dibelanjakan.
3. **Support Tickets**: Frekuensi komplain.
4. **Subscription Type**: Paket langganan (Annual/Monthly).
5. **Is Premium User**: Status pengguna premium (Ya/Tidak).
6. **Discount Used**: Sering memakai diskon? (Ya/Tidak).
7. **Delivery Delay Days**: Hari keterlambatan layanan.
8. **NPS Score**: Net Promoter Score (0-10).
9. **Purchase Frequency**: Keaktifan transaksi (3 bln terakhir).
""")
st.sidebar.info("Gunakan formulir di tengah halaman untuk melakukan prediksi.")

# ---------------------------------------------------------------------
# TAHAP 2: MEMUAT FILE .PKL (OTAK MACHINE LEARNING)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("model_churn_terbaik.pkl")
    encoder = joblib.load("encoder_churn.pkl")
    return model, encoder

try:
    model, encoder = load_models()
    le_sub = encoder['subscription_type']
except Exception as e:
    st.error(f"⚠️ Gagal memuat model Machine Learning! Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# TAHAP 3: FORM INPUT FITUR (TENGAH HALAMAN)
# ---------------------------------------------------------------------
st.header("📊 Masukkan Data Pelanggan")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        support_tickets = st.number_input("Jumlah Tiket Bantuan", min_value=0, max_value=50, value=2)
        is_premium_user = st.selectbox("Status Premium?", ["Ya", "Tidak"])
        
    with col2:
        total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, max_value=100000.0, value=500.0)
        subscription_type = st.selectbox("Tipe Langganan", le_sub.classes_)
        discount_used = st.selectbox("Menggunakan Diskon?", ["Ya", "Tidak"])
        
    with col3:
        delivery_delay_days = st.number_input("Keterlambatan (Hari)", min_value=0, max_value=30, value=1)
        nps_score = st.slider("NPS Score (0-10)", 0, 10, 7)
        last_3_month_purchase_freq = st.number_input("Frekuensi Belanja (3 Bln)", min_value=0, max_value=100, value=5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔮 Analisis Risiko Churn")

# ---------------------------------------------------------------------
# TAHAP 4: PROSES PREDIKSI & TAMPILAN HASIL
# ---------------------------------------------------------------------
if submit_button:
    is_premium_val = 1 if is_premium_user == "Ya" else 0
    discount_val = 1 if discount_used == "Ya" else 0
    sub_encoded = le_sub.transform([subscription_type])[0]
    
    # Input data disusun menjadi DataFrame persis seperti saat model dilatih (X_train_vip)
    input_data = pd.DataFrame({
        'satisfaction_score': [satisfaction_score],
        'total_spent': [total_spent],
        'support_tickets': [support_tickets],
        'subscription_type': [sub_encoded],
        'is_premium_user': [is_premium_val],
        'discount_used': [discount_val],
        'delivery_delay_days': [delivery_delay_days],
        'nps_score': [nps_score],
        'last_3_month_purchase_freq': [last_3_month_purchase_freq]
    })
    
    try:
        with st.spinner("AI sedang menganalisis pola pelanggan..."):
            prediksi = model.predict(input_data)[0]
            probabilitas = model.predict_proba(input_data)[0]
            prob_churn = probabilitas[1] * 100
        
        st.subheader("📝 Hasil Analisis AI")
        
        if prediksi == 1:
            st.markdown(f"""
                <div class="danger-box">
                    <h2>⚠️ RISIKO TINGGI (CHURN)</h2>
                    <p>AI memprediksi pelanggan ini <b>AKAN MENINGGALKAN</b> layanan dengan tingkat keyakinan <b>{prob_churn:.1f}%</b>.</p>
                </div>
            """, unsafe_allow_html=True)
            st.write("---")
            st.error("💡 **Rekomendasi Aksi Bisnis:** Segera hubungi pelanggan ini secara personal dan berikan diskon retensi/perpanjangan.")
            
        else:
            st.markdown(f"""
                <div class="safe-box">
                    <h2>✅ AMAN (ACTIVE)</h2>
                    <p>Pelanggan ini diprediksi <b>SETIA</b> dengan tingkat keyakinan <b>{(100 - prob_churn):.1f}%</b>.</p>
                </div>
            """, unsafe_allow_html=True)
            st.write("---")
            st.success("💡 **Rekomendasi Aksi Bisnis:** Tawarkan produk premium (Upselling) atau masukkan ke Loyalty Program.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.error("TIPS: Pastikan Anda telah menimpa model_churn_terbaik.pkl dengan model yang dilatih HANYA menggunakan 9 fitur VIP.")
