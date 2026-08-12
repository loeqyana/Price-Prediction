import streamlit as st
import joblib
import pandas as pd

# 1. Memuat model Pipeline yang sudah disimpan
# Model ini mencakup preprocessor (scaling & encoding) dan XGB Regression
model = joblib.load('XGB_submit.pkl')

# 3. session state
if "job" not in st.session_state:
    st.session_state.job = "Mahasiswa"

if "budget" not in st.session_state:
    st.session_state.budget = 10000000

# 2. Judul dan Deskripsi Aplikasi
st.title("Laptop Price Predictor")
st.write("""
Aplikasi ini memprediksi **Harga Laptop** berdasarkan spesifikasi teknis 
menggunakan model XGBoost Regression yang telah dioptimalkan.
""")

st.header("👤 Profil Pengguna")

colA, colB = st.columns(2)

with colA:

    job = st.selectbox(
        "Pekerjaan",
        [
            "Mahasiswa",
            "Programmer",
            "Data Analyst",
            "Designer",
            "Video Editor",
            "Gamer",
            "Pekerja Kantoran",
            "Lainnya"
        ],
        key="job"
    )

with colB:

    budget = st.number_input(
        "Budget (Rp)",
        min_value=1000000,
        max_value=100000000,
        step=500000,
        key="budget"
    )

st.divider()

st.header("Masukkan Spesifikasi Laptop")

# 3. Membuat Input Form untuk 13 Fitur Terpilih
col1, col2 = st.columns(2)

with col1:
    # Fitur Kategorikal (Berdasarkan analisis EDA) 
    company = st.selectbox("Merek (Company)", [
        'HP', 'Toshiba',
        'Acer', 'Asus', 
        'Dell', 'Lenovo', 
        'Apple', 'Other'
        ])
    if company == "Apple":
        typename_options = ["Ultrabook"]
    else:
        typename_options = [
            "Ultrabook", "Notebook",
            "Netbook", "Gaming",
            "2 in 1 Convertible", "Workstation"
        ]
    typename = st.selectbox("Tipe Laptop", typename_options)
    cpu_brand = st.selectbox("Cpu Brand", ['Intel', 'AMD'])
    if cpu_brand == "Intel":
        cpu_family_options = [
            "Core i7", "Core i5", 
            "Core i3", "Core M", 
            "Celeron Dual Core", "Pentium Quad Core", 
            "Xeon" 
        ]
    elif cpu_brand == "AMD" :
        cpu_family_options = [ 
            "Ryzen", "A-Series", 
            "E-Series", "FX-Series"
        ]
    cpu_family = st.selectbox("Keluarga CPU", cpu_family_options)
    if company == "Apple":
        opsys_options = ["macOS"]
    else:
        opsys_options = [
            "Windows", "Linux",
            "No OS", "Android",
            "Chrome OS"
        ]
    opsys = st.selectbox("Sistem Operasi", opsys_options)
    gpu = st.selectbox("GPU Brand", 
                    ['Intel', 'AMD', 'Nvidia'])
    if gpu == "Intel":
        gpu_options = [
            "HD Graphics", "HD Graphics 400",
            "HD Graphics 405", "HD Graphics 500",
            "HD Graphics 505", "HD Graphics 510",
            "HD Graphics 515", "HD Graphics 520",
            "HD Graphics 530", "HD Graphics 540",
            "HD Graphics 615", "HD Graphics 620",
            "HD Graphics 630", "HD Graphics 5300",
            "HD Graphics 6000", "Iris Graphics 540",
            "Iris Plus Graphics 640", "Iris Plus Graphics 650",
            "UHD Graphics 620", "UHD Graphics 620"
        ]
    elif gpu == "Nvidia":
        gpu_options = [
            "GeForce 920", "GeForce 920M",
            "GeForce 920MX", "GeForce 930M",
            "GeForce GT 940MX", "GeForce GTX 1050",
            "GeForce GTX 1050 Ti", "GeForce GTX 1050M",
            "GeForce GTX 1060", "GeForce GTX 1070",
            "GeForce GTX 1080", "GeForce GTX 930MX",
            "GeForce GTX 940M", "GeForce GTX 940MX",
            "GeForce GTX 950M", "GeForce GTX 960",
            "GeForce GTX 960M", "GeForce GTX 965M",
            "GeForce GTX 970M", "GeForce GTX 980",
            "GeForce GTX 980M", "GeForce MX130",
            "GeForce MX150", "GTX 980 SLI"
            "Quadro M620", "Quadro M100M",
            "Quadro 3000M", "Quadro M1200",
            "Quadro M2000M", "Quadro M2200",
            "Quadro M2200M", "Quadro M3000M",
            "Quadro M5200M", "Quadro M620M",
        ]
    else:   # AMD
        gpu_options = [
            "FirePro W4190M", "FirePro W5130M", 
            "FirePro W6150M", "R17M-M1-70", 
            "Radeon 530", "Radeon Pro 455", 
            "Radeon Pro 555", "Radeon Pro 560",  
            "Radeon R5 520", "Radeon R5 M315", 
            "Radeon R5 M330", "Radeon R5 M420", 
            "Radeon R5 M420X", "Radeon R5 M430", 
            "Radeon R7 M365X", "Radeon R7 M440", 
            "Radeon R7 M445", "Radeon R7 M460", 
            "Radeon R7 M465", "Radeon RX 550", 
            "Radeon RX 580"
        ]
    gpu_model = st.selectbox("GPU Model", gpu_options)

with col2:
    # Fitur Numerik (Disesuaikan dengan preprocessing StandardScaler) [2]
    inches = st.number_input("Inches", min_value=10.0, max_value=18.5, value=15.6, step=0.01)
    ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
    ssd = st.selectbox("Kapasitas SSD (GB)", [0, 128, 256, 512, 1024, 2048])
    weight = st.number_input("Berat Laptop (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.01)
    clockspeed = st.number_input("CPU Clock Speed (GHz)", min_value=0.9, max_value=4.0, value=2.5, step=0.1)
    ppi = st.number_input("PPI (Pixels Per Inch)", value=141.21)
    screentype = st.selectbox("Screen Type", ['IPS Panel', 'IPS Panel Retina Display', 'Standard'])

submit_button = st.button("Estimasi Harga")

# 4. Logika Prediksi
if submit_button:
    # Data harus dalam bentuk DataFrame agar sesuai dengan ColumnTransformer di Pipeline
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'ScreenType': [screentype],
        'Ram': [ram],
        'Gpu': [gpu],
        'OpSys': [opsys],
        'Weight': [weight],
        'Inches': [inches],
        'PPI': [ppi],
        'SSD_GB': [ssd],
        'Cpu_Brand' : [cpu_brand],
        'Cpu_Family': [cpu_family],
        'Cpu_Clockspeed': [clockspeed]
    })
    
    # Prediksi menggunakan pipeline (Otomatis melakukan scaling & encoding)
    prediksi = model.predict(input_data)[0]
    
    margin = 0.05  # Margin 5% 
    batas_bawah = prediksi * (1 - margin)
    batas_atas = prediksi * (1 + margin)

    kurs_eur_idr = 19000
    harga_rupiah_min = batas_bawah * kurs_eur_idr
    harga_rupiah_max = batas_atas * kurs_eur_idr
    harga_tengah = (harga_rupiah_min + harga_rupiah_max) / 2

    st.metric(
        "💶 Estimasi Harga (Euro)",
        f"€ {batas_bawah:,.2f} - € {batas_atas:,.2f}"
    )

    st.metric(
        "🇮🇩 Estimasi Harga (Rupiah)",
        f"Rp {harga_rupiah_min:,.0f} - {harga_rupiah_max:,.0f}"
    )
    
    st.divider()

    st.subheader("📊 Analisis Budget")

    if harga_tengah <= budget:
        st.success(
            f"✅ Laptop ini sesuai dengan budget Anda.\n\n"
            f"Budget: Rp {budget:,.0f}"
        )
    else:

        selisih = harga_tengah - budget

        st.error(
            f"❌ Laptop ini melebihi budget sekitar "
            f"Rp {selisih:,.0f}"
        )
        
    st.divider()

    st.subheader("💡 Rekomendasi")

    if job == "Mahasiswa":

        st.info("""
        Cocok untuk:
        ✔ Microsoft Office
        ✔ Zoom
        ✔ Browser
        ✔ Coding ringan

        Rekomendasi:
        • RAM ≥ 8 GB
        • SSD ≥ 256 GB
        """)

    elif job == "Programmer":

        st.info("""
        Cocok untuk:
        ✔ Visual Studio Code
        ✔ Android Studio
        ✔ Docker

        Rekomendasi:
        • RAM ≥ 16 GB
        • SSD ≥ 512 GB
        • Core i7 / Ryzen 7
        """)

    elif job == "Data Analyst":

        st.info("""
        Cocok untuk:
        ✔ Python
        ✔ SQL
        ✔ Power BI
        ✔ Tableau

        Rekomendasi:
        • RAM ≥ 16 GB
        • SSD ≥ 512 GB
        """)

    elif job == "Designer":

        st.info("""
        Cocok untuk:
        ✔ Photoshop
        ✔ Illustrator

        Rekomendasi:
        • GPU Dedicated
        • RAM ≥ 16 GB
        """)

    elif job == "Video Editor":

        st.info("""
        Cocok untuk:
        ✔ Premiere Pro
        ✔ After Effects

        Rekomendasi:
        • RAM ≥ 32 GB
        • Nvidia GPU
        • SSD ≥ 1 TB
        """)

    elif job == "Gamer":

        st.info("""
        Cocok untuk:
        ✔ Gaming AAA

        Rekomendasi:
        • Nvidia GTX/RTX
        • RAM ≥ 16 GB
        """)
        
    st.divider()

    st.subheader("📋 Evaluasi Spesifikasi")

    if ram >= 16:
        st.success("✅ RAM sudah sangat baik.")
    elif ram >= 8:
        st.warning("⚠ RAM cukup, tetapi 16 GB akan lebih nyaman.")
    else:
        st.error("❌ RAM terlalu kecil.")

    if ssd >= 512:
        st.success("✅ SSD sudah memadai.")
    elif ssd >= 256:
        st.warning("⚠ SSD cukup.")
    else:
        st.error("❌ SSD terlalu kecil.")

    if cpu_family in ["Core i7", "Ryzen", "Xeon"]:
        st.success("✅ Performa CPU tinggi.")
    elif cpu_family in ["Core i5"]:
        st.info("ℹ CPU kelas menengah.")
    else:
        st.warning("⚠ CPU cocok untuk penggunaan ringan.")