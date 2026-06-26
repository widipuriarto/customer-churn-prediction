import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# TAHAP 1: KONFIGURASI HALAMAN & DESAIN UI
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Premium
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    h1 { color: #1e3799; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 800;}
    h2 { color: #2f3542; padding-bottom: 10px;}
    .subtitle { text-align: center; color: #57606f; font-size: 1.3rem; margin-bottom: 2rem; font-style: italic;}
    .stButton>button { width: 100%; font-weight: bold; background-color: #1e3799; color: white; border-radius: 10px; padding: 10px; font-size: 1.1rem; border: none; transition: 0.3s;}
    .stButton>button:hover { background-color: #4a69bd; box-shadow: 0px 4px 15px rgba(0,0,0,0.2); transform: scale(1.02); }
    .safe-box { padding: 25px; border-radius: 12px; background-color: #E8F5E9; color: #2E7D32; text-align: center; border: 1px solid #A5D6A7; margin-top: 20px;}
    .danger-box { padding: 25px; border-radius: 12px; background-color: #FFEBEE; color: #C62828; text-align: center; border: 1px solid #EF9A9A; margin-top: 20px;}
    .info-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #1e3799;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Customer Churn Predictor with Decision Tree")
st.markdown('<p class="subtitle">Decision Tree Model after Hyperparameter Tuning</p>', unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------------------
# TAHAP 2: MEMUAT FILE .PKL (OTAK MACHINE LEARNING)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("model_churn_terbaik.pkl")
    return model

try:
    model = load_models()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model Machine Learning! Error: {e}")
    st.stop()

# Membuat Sistem Tab agar Profesional
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Interaktif", "📖 Dokumentasi Proyek", "📊 Wawasan Data & Fitur"])

# =====================================================================
# TAB 1: PREDIKSI INTERAKTIF (FORM UTAMA)
# =====================================================================
with tab1:
    st.markdown('<div class="info-box">Silakan masukkan data pelanggan terbaru ke dalam formulir di bawah ini. AI akan menganalisis 9 metrik VIP Decision Tree untuk menentukan probabilitas pelanggan kabur (Churn).</div>', unsafe_allow_html=True)
    
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            satisfaction_score = st.slider("⭐ Satisfaction Score (1-5)", 1, 5, 3, help="1 = Sangat Kecewa, 5 = Sangat Puas")
            days_since_signup = st.number_input("📅 Usia Akun (Hari Sejak Daftar)", min_value=0, max_value=5000, value=365)
            email_open_rate = st.slider("📧 Email Open Rate (0.0 - 1.0)", 0.0, 1.0, 0.5, step=0.05)
            
        with col2:
            total_spent = st.number_input("💰 Total Pengeluaran ($)", min_value=0.0, max_value=100000.0, value=500.0)
            marketing_spend_per_user = st.number_input("📢 Biaya Marketing per User ($)", min_value=0.0, max_value=5000.0, value=20.0)
            days_since_last_purchase = st.number_input("🛒 Hari Sejak Belanja Terakhir", min_value=0, max_value=365, value=30)
            
        with col3:
            support_tickets = st.number_input("🎫 Jumlah Tiket Bantuan (Komplain)", min_value=0, max_value=50, value=2)
            lifetime_value = st.number_input("💎 Lifetime Value (CLV $)", min_value=0.0, max_value=100000.0, value=1000.0)
            avg_order_value = st.number_input("💵 Rata-rata Nilai Order ($)", min_value=0.0, max_value=10000.0, value=50.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🔮 Mulai Analisis Risiko Churn")

    if submit_button:
        # Menyusun DataFrame sesuai urutan asli dari model (feature_names_in_)
        input_data = pd.DataFrame({
            'satisfaction_score': [satisfaction_score],
            'total_spent': [total_spent],
            'support_tickets': [support_tickets],
            'days_since_signup': [days_since_signup],
            'marketing_spend_per_user': [marketing_spend_per_user],
            'lifetime_value': [lifetime_value],
            'email_open_rate': [email_open_rate],
            'days_since_last_purchase': [days_since_last_purchase],
            'avg_order_value': [avg_order_value]
        })
        
        try:
            with st.spinner("🧠 AI Decision Tree sedang menyusuri percabangan pohon logika..."):
                prediksi = model.predict(input_data)[0]
                
                if hasattr(model, "predict_proba"):
                    probabilitas = model.predict_proba(input_data)[0]
                    prob_churn = probabilitas[1] * 100
                    prob_text = f" <b>{prob_churn:.1f}%</b>"
                else:
                    prob_text = ""
            
            # --- Layout Terpusat yang Lebih Kecil ---
            col_space_kiri, col_utama, col_space_kanan = st.columns([1, 4, 1])
            
            with col_utama:
                st.markdown("<h3 style='text-align: center;'>Hasil Prediksi</h3><br>", unsafe_allow_html=True)
                
                if prediksi == 1:
                    st.markdown(f"""
                        <div class="danger-box" style="margin-top: 0; padding: 15px;">
                            <h2 style="color: #C62828; font-size: 32px; margin-bottom: 5px;">CHURN</h2>
                            <p style="font-size: 1.1rem; margin: 0;">Probabilitas:{prob_text}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="safe-box" style="margin-top: 0; padding: 15px;">
                            <h2 style="color: #2E7D32; font-size: 32px; margin-bottom: 5px;">TIDAK CHURN</h2>
                            <p style="font-size: 1.1rem; margin: 0;">Probabilitas:{prob_text}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")

# =====================================================================
# TAB 2: DESKRIPSI PROYEK & PIPELINE MODEL
# =====================================================================
with tab2:
    st.markdown("## 📖 Latar Belakang Proyek")
    st.write("""
    Dalam industri modern, **biaya untuk mendapatkan pelanggan baru (Customer Acquisition Cost) bisa 5 hingga 25 kali lipat lebih mahal daripada mempertahankan pelanggan yang sudah ada**. 
    Proyek ini dibangun untuk menyelesaikan masalah tersebut dengan menggunakan kecerdasan buatan (Machine Learning) guna mendeteksi tanda-tanda awal pelanggan yang berencana untuk pergi (*Customer Churn*).
    """)
    
    st.markdown("## ⚙️ Arsitektur Sistem AI (Pipeline)")
    st.markdown("""
    Aplikasi web ini ditenagai oleh model Machine Learning yang telah melewati proses riset saintifik yang sangat ketat di *Jupyter Notebook*, meliputi:
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**1. Data Preprocessing & Cleansing**")
        st.write("Penghapusan data *outlier*, pengisian nilai kosong (missing values), dan *Label Encoding* untuk mengubah data teks menjadi matriks numerik.")
        
        st.info("**2. Feature Selection (Ekstraksi Fitur VIP)**")
        st.write("Dari puluhan variabel awal (53 kolom setelah *encoding*), kami menggunakan analisis *Decision Tree Feature Importance* untuk membuang variabel sampah (*noise*) dan mempertahankan **9 Metrik VIP** paling berpengaruh. Ini membuat model lebih ringan, cepat, dan mencegah *overfitting*.")
        
    with col_b:
        st.info("**3. Hyperparameter Tuning**")
        st.write("Model tidak hanya dilatih menggunakan parameter bawaan. Kami menggunakan teknik `RandomizedSearchCV` untuk mencari konfigurasi parameter terbaik (Tuning) secara matematis demi akurasi maksimal.")
        
        st.info("**4. Model Sang Juara (Tuned Decision Tree)**")
        st.write("Berbeda dengan asumsi umum, model yang terpilih sebagai juara dalam eksperimen ini justru adalah *Decision Tree* tunggal yang telah di-tuning secara presisi. Model ini berhasil mencapai metrik **Recall luar biasa (99.3%)** dan **F1-Score tertinggi (0.67)**, mengalahkan algoritma *Ensemble* kompleks. Pohon keputusan yang ramping ini sangat tajam dalam mendeteksi hampir *semua* pelanggan yang berniat pergi (sensitivitas super tinggi).")

# =====================================================================
# TAB 3: WAWASAN DATA & VISUALISASI FITUR
# =====================================================================
with tab3:
    st.markdown("## 📊 Anatomi 9 Fitur VIP")
    st.write("Berikut adalah penjelasan mendalam tentang 9 fitur terpilih yang terbukti secara statistik paling memengaruhi keputusan pelanggan untuk churn berdasarkan analisa logika *Decision Tree*.")
    
    # Ekstraksi Bobot Kepentingan Fitur secara DINAMIS dari model langsung!
    fitur_names = [
        'Satisfaction Score', 'Total Spent', 'Support Tickets', 
        'Days Since Signup', 'Marketing Spend per User', 'Lifetime Value', 
        'Email Open Rate', 'Days Since Last Purchase', 'Avg Order Value'
    ]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_ * 100
    else:
        importances = [11.1] * 9

    feature_importance_data = pd.DataFrame({
        'Fitur': fitur_names,
        'Dampak terhadap Churn (%)': importances
    })
    
    # Mengurutkan dari dampak paling mematikan (terbesar) ke terkecil
    feature_importance_data = feature_importance_data.sort_values('Dampak terhadap Churn (%)', ascending=False)
    
    # Render grafik menggunakan Matplotlib & Seaborn
    st.markdown("### 🔥 Grafik Pengaruh Fitur (Feature Importance)")
    
    # Menampilkan kode ekstraksi untuk pamer ke dosen
    with st.expander("💻 Lihat Source Code Ekstraksi Dinamis"):
        st.code("""
# Mengekstrak atribut feature_importances_ bawaan dari Model Decision Tree
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_ * 100
    
feature_importance_data = pd.DataFrame({
    'Fitur': fitur_names,
    'Dampak terhadap Churn (%)': importances
})
        """, language="python")

    # Menampilkan Data Asli (Tabel) agar persentase terlihat jelas
    st.write("**Tabel Detail Persentase Kontribusi Setiap Fitur:**")
    st.dataframe(feature_importance_data.style.format({'Dampak terhadap Churn (%)': '{:.2f}%'}))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x='Dampak terhadap Churn (%)', 
        y='Fitur', 
        hue='Fitur',
        legend=False,
        data=feature_importance_data, 
        palette='rocket',
        ax=ax
    )
    ax.set_title('Tingkat Pengaruh Setiap Fitur Terhadap Keputusan Churn', fontweight='bold', fontsize=14)
    ax.set_xlabel('Bobot Kepentingan (%)', fontsize=12)
    ax.set_ylabel('')
    
    # Menambahkan teks persentase di batang
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}%", 
                    (p.get_width() + 0.5, p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10)
    
    st.pyplot(fig)
    
    st.write("---")
    st.markdown("### 📚 Kamus Fitur (Data Dictionary)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Satisfaction Score**: Indikator psikologis utama pelanggan.
        - **Total Spent**: Semakin besar uang yang dihabiskan, biasanya pelanggan semakin merasa 'terikat' (Sunk-cost fallacy).
        - **Support Tickets**: Tingginya frekuensi komplain menandakan gesekan (*friction*) yang sering berujung pada kekecewaan.
        - **Days Since Signup**: Menunjukkan usia/loyalitas historis akun tersebut.
        - **Marketing Spend per User**: Seberapa banyak uang promosi yang 'dibakar' untuk mempertahankan atau mengakuisisi pelanggan ini.
        """)
    with col2:
        st.markdown("""
        - **Lifetime Value (CLV)**: Proyeksi nilai finansial pelanggan di masa depan berdasarkan tren belanjanya.
        - **Email Open Rate**: Menandakan apakah pelanggan tersebut mengabaikan (0.0) atau rutin membaca (1.0) komunikasi *brand*.
        - **Days Since Last Purchase**: Jeda hari yang semakin panjang menandakan menurunnya minat belanja (*dormancy*).
        - **Avg Order Value**: Rata-rata nominal per keranjang belanja. Menunjukkan daya beli dan kebiasaan transaksi.
        """)
