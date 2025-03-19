import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import MaxAbsScaler

# Konfigurasi halaman menjadi full screen (wide)
st.set_page_config(page_title="Skripsi", layout="wide")

# CSS untuk tampilan dan style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    
    * {
        font-family: 'Playfair Display', sans-serif;
    }

    /* Heading (h1, h2, h3) warna navy */
    h1, h2, h3 {
        color: navy;
    }

    /* Teks biasa warna navy */
    p, div, span, li {
        color: #2F2F2F;
    }

    .stApp {
        background-color:#ffffff;  
    }

    /* Hilangkan margin dan padding default Streamlit */
    .main .block-container {
        padding: 0;
        margin: 0;
    }

    /* Penyesuaian header dengan logo */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        background-color: #ffffff;
    }

    .logo {
        height: 50px;
    }

    .seminar-text {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }

    /* Class untuk menu navigasi Streamlit */
    .css-1d391kg {  
        font-size: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header dengan logo di kiri dan tulisan "Sidang Skripsi" di kanan
st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        background-color: #ffffff;
    }
    .logo-container img {
        height: 50px;
    }
    .seminar-text {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    </style>

    <div class="header-container">
        <div class="logo-container">
            <!-- Menampilkan gambar logo menggunakan st.image() -->
        </div>
        <span class="seminar-text">Seminar Hasil</span>
    </div>
""", unsafe_allow_html=True)

# Gunakan st.image untuk menampilkan logo dengan path relatif
st.image("images/download (5).png", width=250)  # Pastikan path gambar sesuai

# Navigasi Utama
selected_main = option_menu(
    menu_title=None,
    options=["Cover", "Introduction", "Data Source", "Results", "Conclusion"],
    icons=["book", "list-task", "bar-chart-line", "clipboard-check", "check-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Konten berdasarkan pilihan di menu utama
if selected_main == "Cover":
    # CSS untuk memusatkan judul dan mengatur barisan untuk informasi lainnya
    st.markdown("""
        <style>
        .cover-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 80vh;
            background-color: #f9f9f9;
            padding: 10px;
        }
        .cover-title {
            font-size: 25px;
            font-weight: bold;
            margin-top: 5px;
            margin-bottom: 30px;
            margin-left: 5cm;
            margin-right: 5cm;
        }
        .cover-text {
            font-size: 15px;
            line-height: 1.15;
        }
        .info-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin-bottom: 20px;
        }
        .info-top {
            display: flex;
            justify-content: center;
            width: 100%;
            margin-bottom: 1cm;
        }
        .info-left,
        .info-right {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-left: 3cm;
            margin-right: 3cm;
        }
        .info-item {
            margin-bottom: 10px;
            align-items: left;
        }
        .info-bottom {
            display: flex;
            justify-content: space-between;
            width: 100%;
        }
        .logoo {
            height: 100px;
        }
        </style>
        <div class="cover-container">
            <div class="cover-title">PEMBENTUKAN PORTOFOLIO <i>MEAN-SEMIVARIANCE</i> BERBASIS SELEKSI SAHAM MENGGUNAKAN <i>PARTITIONING AROUND MEDOIDS</i> (PAM) <i>CLUSTERING</i> PADA IDX SKETOR KESEHATAN</div>
            <div class="info-container">
                <div class="info-top">
                    <div class="info-left">
                        <div class="info-item"><strong>Nama:</strong> Safril Ahmadi Sanmas</div>
                        <div class="info-item"><strong>NIM:</strong> B2A021017</div>
                    </div>
                </div>
                <div class="info-bottom">
                    <div class="info-left">
                        <div class="info-item"><strong>Pembimbing 1:</strong> M Al Haris, M.Si </div>
                    </div>
                    <div class="info-right">
                        <div class="info-item"><strong>Pembimbing 2:</strong> Dannu Purwanto, S.T., M.Kom </div>
                    </div>
                </div>
            </div>
            <div class="cover-text">
                <p>PROGRAM STUDI STATISTIKA</p>
                <p>FAKULTAS SAINS DAN TEKNOLOGI PERTANIAN</p>
                <p>UNIVERSITAS MUHAMMADIYAH SEMARANG</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Daftar sub-bagian untuk Introduction
intro_sections = [
    "Latar Belakang",
    "Penelitian Terdahulu",
    "Rumusan Masalah",
    "Tujuan Penelitian",
    "Manfaat Penelitian",
    "Batasan Penelitian",
    "Tinjauan Pustaka"
]
# Daftar sub-bagian untuk Data Source
data_source = [
    "Data Rasio Profitabilitas",
    "Data Harga Penutupan Saham",
    "Data Suku Bungan Bank Indonesia (BI Rate)",
]

# Daftar sub-bagian untuk Introduction
results = [
    "Statistika Deskriptif Rasio Profitabilitas",
    "Pendeteksian Pencilan",
    "Standarisasi Variabel Fundamental",
    "Pengujian Asumsi Klasik",
    "Principal Component Analysis",
    "Perhitungan Jarak",
    "Partitioning Around Medoids",
    "Profilisasi Klaster",
    "Pemilihan Saham Terbaik Tiap Klaster"
]

# Inisialisasi session_state untuk menyimpan index sub-bagian
if "intro_section" not in st.session_state:
    st.session_state["intro_section"] = 0  # Mulai dari sub-bagian pertama

# Fungsi untuk navigasi
def next_section():
    if st.session_state["intro_section"] < len(intro_sections) - 1:
        st.session_state["intro_section"] += 1

def prev_section():
    if st.session_state["intro_section"] > 0:
        st.session_state["intro_section"] -= 1

# Inisialisasi session_state untuk menyimpan index sub-bagian
if "data_source" not in st.session_state:
    st.session_state["data_source"] = 0  # Mulai dari sub-bagian pertama

# Fungsi untuk navigasi
def next_section1():
    if st.session_state["data_source"] < len(data_source) - 1:
        st.session_state["data_source"] += 1

def prev_section1():
    if st.session_state["data_source"] > 0:
        st.session_state["data_source"] -= 1


# Inisialisasi session_state untuk menyimpan index sub-bagian
if "results" not in st.session_state:
    st.session_state["results"] = 0  # Mulai dari sub-bagian pertama

# Fungsi untuk navigasi
def next_section2():
    if st.session_state["results"] < len(results) - 1:
        st.session_state["results"] += 1

def prev_section2():
    if st.session_state["results"] > 0:
        st.session_state["results"] -= 1

# Konten berdasarkan pilihan di menu utama
if selected_main == "Introduction":
    st.header(intro_sections[st.session_state["intro_section"]])

    # Konten berdasarkan sub-bagian
    if st.session_state["intro_section"] == 0:  # Latar Belakang
        st.markdown("""
    <div style='font-size: 20px;'>
        <ol>
            <li><strong>Saham</strong> adalah bukti kepemilikan perusahaan yang memberikan peluang partisipasi dalam pertumbuhan perusahaan melalui investasi.</li>
            <li><strong>Indeks IDX Sektor Kesehatan</strong> mencakup perusahaan farmasi, rumah sakit, dan alat kesehatan, memberikan gambaran kinerja sektor kesehatan di Indonesia sejak diluncurkan pada 2021.</li>
            <li>Dengan alokasi anggaran kesehatan yang meningkat menjadi Rp186,4 triliun pada 2024, sektor ini menunjukkan pertumbuhan signifikan, didorong oleh permintaan layanan kesehatan yang terus meningkat.</li>
            <li>Indeks ini mencatat pertumbuhan 58,8% pada 2024, namun volatilitas tinggi akibat faktor ekonomi global dan risiko geopolitik menjadi tantangan bagi investor.</li>
            <li>Diversifikasi risiko melalui pembentukan portofolio optimal menggunakan analisis <em>clustering</em> untuk meminimalkan risiko dan memaksimalkan <em>return</em>.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)
        
    elif st.session_state["intro_section"] == 1:  # Penelitian Terdahulu
        st.markdown("""
    <div style='font-size: 20px;'>
        <ol>
            <li><strong>Gubu et al. (2021)</strong> <em>Time Series Clustering</em> menggunakan PAM dan DTW untuk mengelompokkan saham berdasarkan pola harga.</li>
            <li><strong>Suyasa et al. (2021)</strong> Pembentukan portofolio menggunakan <em>Mean-Semivariance</em>  tanpa <em>clustering</em>.</li>
            <li><strong>Pangkuwati et al. (2024)</strong> Pembentukan portofolio menggunakan <em>Mean-Semivariance</em> tanpa <em>clustering</em>.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)
        
    elif st.session_state["intro_section"] == 2:  # Rumusan Masalah
        st.markdown("""
    <div style='font-size: 20px;'>
        <ol>
            <li>Bagaimana proses pengelompokan saham pada Indeks IDX Kesehatan berdasarkan variabel fundamental <em>Net Profit Margin</em> (NPM), <em>Return on Equity</em> (ROE), <em>Return on Asset</em> (ROA), dan <em>Earnings Per Share</em> (EPS) menggunakan metode <em>PAM Clustering</em>?</li>
            <li>Bagaimana pembentukan portofolio yang optimal menggunakan metode <em>Mean-Semivariance</em> untuk saham terbaik dari masing-masing klaster?</li>
            <li>Bagaimana perhitungan nilai <em>VaR</em> portofolio saham terbentuk dengan metode <em>Historical Simulation</em>?</li>
            <li>Bagaimana pengukuran kinerja portofolio saham yang terbentuk dengan <em>Indeks Sharpe</em>?</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

    elif st.session_state["intro_section"] == 3:  # Tujuan Penelitian
        st.markdown("""
    <div style='font-size: 20px;'>
        <ol>
            <li>Mengelompokkan saham IDX Kesehatan berdasarkan <em>NPM</em>, <em>ROE</em>, <em>ROA</em>, dan <em>EPS</em> menggunakan <em>PAM Clustering</em>.</li>
            <li>Membentuk portofolio optimal dengan menggunakan model <em>Mean-Semivariance</em>.</li>
            <li>Menentukan kerugian maksimal atau nilai <em>VaR</em> saat berinvestasi pada portofolio saham indeks IDX Sektor Kesehatan.</li>
            <li>Menentukan baik tidaknya kinerja portofolio saham indeks IDX Sektor Kesehatan yang terbentuk berdasarkan <em>Indeks Sharpe</em>.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

    elif st.session_state["intro_section"] == 4:  # Manfaat Penelitian
        st.markdown("""
    <div style='font-size: 20px;'>
        <b>Manfaat Teoritis:</b> Memperkaya literatur dan menjadi referensi tentang penggunaan <em>clustering</em> dalam seleksi saham serta aplikasi <em>Mean-Semivariance</em> dalam pembentukan portofolio.
        <br><br>
        <b>Manfaat Praktis:</b> Memberikan panduan bagi investor dan praktisi pasar modal dalam mengelola portofolio saham sektor kesehatan dengan teknik <em>clustering</em> dan <em>Mean-Semivariance</em> untuk analisis risiko yang lebih akurat.
    </div>
""", unsafe_allow_html=True)


    elif st.session_state["intro_section"] == 5:  # Batasan Penelitian
        st.markdown("""
    <ul style='font-size: 20px;'>
        <li>Data yang digunakan adalah laporan keuangan tahunan perusahaan sektor kesehatan yang terdaftar di BEI pada tahun 2021-2023.</li>
        <li>Fokus clustering hanya pada variabel fundamental: <em>Net Profit Margin</em> (NPM), <em>Return on Equity</em> (ROE), <em>Return on Asset</em> (ROA), dan <em>Earnings Per Share</em> (EPS).</li>
        <li>Pembentukan portofolio menggunakan <em>Mean-Semivariance</em> tanpa mempertimbangkan faktor eksternal seperti kondisi makroekonomi dan kebijakan pemerintah.</li>
    </ul>
""", unsafe_allow_html=True)


    elif st.session_state["intro_section"] == 6:  # Tinjauan Pustaka
         st.header("Saham & IDX Sektor Kesahatan")
         st.markdown("""
    <p style='font-size: 20px;'>
        <strong>Saham</strong> adalah instrumen investasi yang menandakan kepemilikan di suatu perusahaan, memberikan hak atas keuntungan perusahaan, dan dalam beberapa kasus, pemegang saham memiliki hak suara dalam pengambilan keputusan.
    </p>

    <p style='font-size: 20px;'>
        <strong>IDX Sektor Kesehatan</strong> adalah indeks yang mencerminkan kinerja sektor kesehatan di Bursa Efek Indonesia, mencakup perusahaan dalam layanan kesehatan, farmasi, dan penelitian kesehatan.
    </p>
""", unsafe_allow_html=True)

         st.header("Rasio Profitabilitas")

         st.markdown("""
    <ul style='font-size: 20px;'>
        <li><strong>Earnings Per Share (EPS):</strong> Mengukur laba bersih per lembar saham yang beredar. Indikator kinerja operasi dan profitabilitas perusahaan.</li>
        <li><strong>Net Profit Margin (NPM):</strong> Persentase laba bersih terhadap pendapatan. Menunjukkan efisiensi perusahaan dalam mengelola biaya dan pajak.</li>
        <li><strong>Return on Equity (ROE):</strong> Mengukur pengembalian atas ekuitas pemegang saham. Indikator kemampuan perusahaan dalam memanfaatkan modal sendiri untuk menghasilkan laba.</li>
        <li><strong>Return on Assets (ROA):</strong> Mengukur efisiensi penggunaan aset perusahaan untuk menghasilkan laba.</li>
    </ul>
""", unsafe_allow_html=True)


         st.header("PAM _Clustering_")
         st.markdown("""
    <p style='text-align: justify; font-size: 20px;'>
        <strong>PAM Clustering</strong> adalah metode <em>non-hierarchical clustering</em> yang mengatasi kelemahan <em>K-Means</em> terhadap pencilan (<em>outlier</em>) dengan menggunakan <em>medoid</em> sebagai pusat klaster.  
        Berbeda dengan <em>centroid</em> pada <em>K-Means</em>, <em>medoid</em> adalah objek nyata dalam data yang paling mewakili klaster.
    </p>

    <p style='font-size: 20px;'><strong>Keunggulan</strong></p>
    <ul style='font-size: 20px;'>
        <li><strong>Tahan terhadap pencilan:</strong> Lebih robust dibandingkan <em>K-Means</em> dalam menangani pencilan.</li>
        <li><strong>Penggunaan <em>Medoid</em>:</strong> Memungkinkan penafsiran yang lebih mudah karena <em>medoid</em> adalah data aktual.</li>
    </ul>
""", unsafe_allow_html=True)
         
         st.header("Portofolio _Mean-Semivariance_")
         st.markdown("""
    <p style='text-align: justify; font-size: 20px;'>
        <strong><em>Mean-semivariance</em></strong> berfokus pada risiko kerugian yang berada di bawah tolok ukur (<em>benchmark</em>), 
                    berbeda dengan <em>mean-variance</em> yang memperhitungkan semua fluktuasi. Ini lebih relevan bagi investor yang lebih khawatir tentang potensi kerugian daripada keuntungan di atas ekspektasi.
    </p>
                     
    <p style='font-size: 20px;'><strong>Keunggulan</strong></p>
    <ul style='font-size: 20px;'>
        Memberikan gambaran yang lebih realistis tentang risiko <em>downside</em>, yang lebih sesuai dengan perilaku investor dalam pengambilan keputusan investasi.
    </ul>
""", unsafe_allow_html=True)

         st.header("_Value at Risk_ (VaR)")
         st.markdown("""
    <p style='text-align: justify; font-size: 20px;'>
        <strong><em>Value at Risk</em>(VaR)</strong> mengukur potensi kerugian maksimum dalam periode tertentu dengan tingkat kepercayaan tertentu. Metode <em>Historical Simulation</em> 
                     memprediksi kerugian masa depan menggunakan data <em>return</em> historis tanpa asumsi distribusi.
    </p>
                     
    <p style='font-size: 20px;'><strong>Keunggulan</strong></p>
    <ul style='font-size: 20px;'>
        Estimasi realistis dan fleksibel terhadap distribusi <em>return</em> asimetris, namun bergantung pada kualitas dan panjang data historis.
    </ul>
""", unsafe_allow_html=True)
         
         st.header("Indeks _Sharpe_")
         st.markdown("""
    <p style='text-align: justify; font-size: 20px;'>
        <strong>Indeks<em>Sharpe</em></strong> mengukur kinerja portofolio dengan melihat seberapa besar <em>return</em> yang diperoleh relatif terhadap risiko yang diambil. 
                     Indeks <em>Sharpe</em> membantu investor membandingkan kinerja berbagai portofolio atau investasi lain.

    </p>
                     
    <p style='font-size: 20px;'><strong>Interpretasi</strong></p>
    <ul style='font-size: 20px;'>
        Rasio yang lebih tinggi menunjukkan kinerja yang lebih baik dalam mengelola risiko, artinya portofolio memberikan <em>return</em> yang lebih tinggi per unit risiko yang diambil.
    </ul>
""", unsafe_allow_html=True)

    # Tombol navigasi
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state["intro_section"] > 0:
            st.button("Back", on_click=prev_section)

    with col2:
        if st.session_state["intro_section"] < len(intro_sections) - 1:
            st.button("Next", on_click=next_section)

# Menampilkan header berdasarkan pemilihan data source
if selected_main == "Data Source":
    st.header(data_source[st.session_state["data_source"]])

    # Konten berdasarkan sub-bagian
    if st.session_state["data_source"] == 0:  # Data Rasio Profitabilitas
        st.markdown("""
        <p style='text-align: justify; font-size: 20px;'>
            17 perusahaan IDX Sektor Kesehatan (sumber: idx.co.id, Financial Data & Ratio) yang konsisten bergabung di IDX Sektor Kesehatan
            sejak peluncuran Januari 2021 hingga September 2024 dengan Rasio Profitabilitas Positif.
        </p>
        """, unsafe_allow_html=True)

        # Membaca file Excel
        file_path = "D:/SKRIPSI/Data/Rasio Profit.xlsx"
        df = pd.read_excel(file_path)

        # Menampilkan data sebagai tabel
        st.dataframe(df)

    elif st.session_state["data_source"] == 1:  # Data Harga Penutupan Saham
        st.markdown("""
        <p style='text-align: justify; font-size: 20px;'>
        17 saham IDX Sektor Kesehatan, rentang waktu Maret 2023 hingga Desember 2024 dengan IHSG sebagai <em>banchmark</em> (sumber: finance.yahoo.com).
        </p>
        """, unsafe_allow_html=True)
    
            # Daftar saham yang diminta
        stocks = ["BMHS.JK", "HALO.JK", "HEAL.JK", "MIKA.JK", "MTMH.JK",
                  "OMED.JK", "PRAY.JK", "PRDA.JK", "RSGK.JK", "SILO.JK",
                  "SRAJ.JK", "KLBF.JK", "MERK.JK", "PEVE.JK", "SIDO.JK",
                  "SOHO.JK", "TSPC.JK"]
        
        # Menambahkan IHSG sebagai benchmark
        stocks.append("^JKSE")  # ^JKSE adalah simbol untuk IHSG di Yahoo Finance

        # Mengunduh data harga penutupan saham dan IHSG
        data = yf.download(stocks, start="2023-03-01", end="2024-12-31")['Close']

        # Mengubah nama kolom (menghilangkan .JK dan mengganti ^JKSE dengan IHSG)
        data.columns = [col.split(".")[0] if col != "^JKSE" else "IHSG" for col in data.columns]

        # Menampilkan data sebagai tabel
        st.dataframe(data)

    elif st.session_state["data_source"] == 2:  # Data Harga Penutupan Saham
        st.markdown("""
        <p style='text-align: justify; font-size: 20px;'>
        Tingkat suku bunga BI bulanan periode Maret 2023 hingga Desember 2024 (sumber: bi.go.id) digunakan sebagai suku bunga bebas risiko.
        </p>
        """, unsafe_allow_html=True)

    # Membaca file Excel
        file_path1 = "D:/SKRIPSI/Data/BI Rate.xlsx"
        
        df1 = pd.read_excel(file_path1)
        # Menampilkan data sebagai tabel
        st.dataframe(df1)

        # Tombol navigasi
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state["data_source"] > 0:
            st.button("Back", on_click=prev_section1)

    with col2:
        if st.session_state["data_source"] < len(data_source) - 1:
            st.button("Next", on_click=next_section1)

# Menampilkan header berdasarkan pemilihan data source
if selected_main == "Results":
    st.header(results[st.session_state["results"]])

    # Konten berdasarkan sub-bagian
    if st.session_state["results"] == 0:
        st.markdown("""
        <p style='text-align: justify; font-size: 20px;'>
            Rasio profitabilitas digunakan dalam analisis klastering karena mencerminkan kinerja keuangan dan strategi bisnis perusahaan dalam menghasilkan laba.
        </p>
        """, unsafe_allow_html=True)

        # Menentukan file path untuk membaca data
        file_path = "D:/SKRIPSI/Data/Rasio Profit.xlsx"  # Pastikan pathnya benar

        # Membaca data dan menghitung statistik deskriptif
        df = pd.read_excel(file_path)
        stats = df.describe().loc[["count", "mean", "min", "max", "std"]].T

        # Menampilkan tabel statistik deskriptif
        st.dataframe(stats)

        # Menambahkan tombol untuk menampilkan syntax
        if st.button('Syntax'):
            code = """
# Menentukan file path untuk membaca data
file_path = "D:/SKRIPSI/Data/Rasio Profit.xlsx"  # Pastikan pathnya benar

# Membaca file Excel
df = pd.read_excel(file_path)

# Menghitung statistik deskriptif yang diinginkan
stats = df.describe().loc[["count", "mean", "min", "max", "std"]].T

# Menampilkan statistik dalam aplikasi Streamlit
st.dataframe(stats)  # Menampilkan tabel statistik deskriptif
"""
            st.code(code, language="python")  # Menampilkan kode Python dengan format yang jelas

    # Konten berdasarkan sub-bagian untuk visualisasi boxplot dan analisis pencilan
    elif st.session_state["results"] == 1:
        st.markdown("""
        <p style='text-align: justify; font-size: 20px;'>
            Analisis pencilan dalam suatu data merupakan salah satu aspek penting dalam statistik multivariat, karena pencilan dapat mempengaruhi hasil analisis dan mengarah pada kesimpulan yang tidak akurat.
        </p>
        """, unsafe_allow_html=True)

        # Menentukan file path untuk membaca data
        file_path = "D:/SKRIPSI/Data/Rasio Profit.xlsx"  # Pastikan pathnya benar

        # Membaca data dan menghitung statistik deskriptif
        df = pd.read_excel(file_path)

        # Fungsi untuk menghasilkan Boxplot
        def generate_boxplot(ax, df, variable, fill_color, title):
                sns.boxplot(x=df[variable], color=fill_color, fliersize=3, flierprops=dict(markerfacecolor='red', marker='o', markersize=3), ax=ax)
                ax.set_title(title, fontsize=10, fontweight='bold', loc='center')
                ax.set_xlabel('')
                ax.set_ylabel(variable)
                ax.grid(True, linestyle='--', alpha=0.5)

            # Visualisasi Boxplot untuk masing-masing variabel dalam format 2x2
        st.header("Visualisasi Boxplot untuk Outlier")
        fig, axes = plt.subplots(2, 2, figsize=(6, 3))  # Membuat layout 2x2
        generate_boxplot(axes[0, 0], df, 'EPS', '#0073e6', 'Visualisasi Outlier pada EPS')
        generate_boxplot(axes[0, 1], df, 'ROA', '#ff6600', 'Visualisasi Outlier pada ROA')
        generate_boxplot(axes[1, 0], df, 'ROE', '#32cd32', 'Visualisasi Outlier pada ROE')
        generate_boxplot(axes[1, 1], df, 'NPM', '#ff6699', 'Visualisasi Outlier pada NPM')

            # Menyesuaikan jarak antar sub-plot agar tidak saling bertumpukan
        plt.tight_layout()
        st.pyplot(fig)  # Menampilkan plot di Streamlit

        # Menambahkan tombol untuk menampilkan sintaks
        if st.button('Syntax Boxplot'):
            code = """
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk menghasilkan Boxplot
def generate_boxplot(ax, df, variable, fill_color, title):
    sns.boxplot(x=df[variable], color=fill_color, fliersize=3, flierprops=dict(markerfacecolor='red', marker='o', markersize=3), ax=ax)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('')
    ax.set_ylabel(variable)
    ax.grid(True, linestyle='--', alpha=0.5)

# Visualisasi Boxplot untuk masing-masing variabel dalam format 2x2
st.header("Visualisasi Boxplot untuk Outlier")
fig, axes = plt.subplots(2, 2, figsize=(6, 3))  # Membuat layout 2x2
generate_boxplot(axes[0, 0], df, 'EPS', '#0073e6', 'Visualisasi Outlier pada EPS')
generate_boxplot(axes[0, 1], df, 'ROA', '#ff6600', 'Visualisasi Outlier pada ROA')
generate_boxplot(axes[1, 0], df, 'ROE', '#32cd32', 'Visualisasi Outlier pada ROE')
generate_boxplot(axes[1, 1], df, 'NPM', '#ff6699', 'Visualisasi Outlier pada NPM')

# Menyesuaikan jarak antar sub-plot agar tidak saling bertumpukan
plt.tight_layout()
st.pyplot(fig)  # Menampilkan plot di Streamlit
"""
            st.code(code, language="python")  # Menampilkan kode Python dengan format yang jelas

                # Perhitungan Jarak Kuadrat Mahalanobis
        st.header("Perhitungan Jarak Kuadrat Mahalanobis")
        df_numeric = df.select_dtypes(include=[np.number])

        # Hitung mean dan matriks kovarians
        mean_values = np.mean(df_numeric, axis=0)
        cov_matrix = np.cov(df_numeric, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Menghitung jarak kuadrat Mahalanobis untuk setiap titik data
        D2 = np.array([mahalanobis(x, mean_values, inv_cov_matrix) ** 2 for x in df_numeric.values])

        # Menentukan batas nilai kritis dari distribusi Chi-Square untuk deteksi pencilan
        alpha = 0.05
        k = df_numeric.shape[1]  # Jumlah fitur (kolom)
        chi_square_critical = chi2.ppf(1 - alpha, k)

        # Menampilkan hasil jarak Mahalanobis dan chi-square kritis
        st.markdown("### Jarak Kuadrat Mahalanobis untuk setiap data:")
        st.write(D2)

        st.markdown(f"### Batas Chi-Square (alpha={alpha}):")
        st.write(chi_square_critical)

        # Menandai outlier jika jarak Mahalanobis melebihi batas chi-square
        outliers = np.where(D2 > chi_square_critical)[0]

        st.markdown("### Indeks Data yang Terdeteksi sebagai Outlier:")
        st.write(outliers)

        # Menambahkan tombol untuk menampilkan sintaks Mahalanobis
        if st.button('Syntax Mahalanobis'):
            code = """
        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import mahalanobis
        from scipy.stats import chi2

        # Perhitungan Mahalanobis
        mean_values = np.mean(Data, axis=0)
        cov_matrix = np.cov(Data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        D2 = np.array([mahalanobis(x, mean_values, inv_cov_matrix) ** 2 for x in Data.values])

        # Menentukan chi-square kritis
        alpha = 0.05
        k = Data.shape[1]
        chi_square_critical = chi2.ppf(1 - alpha, k)

        # Menampilkan hasil
        print("Jarak Kuadrat Mahalanobis:", D2)
        print("Batas Chi-Square:", chi_square_critical)
        outliers = np.where(D2 > chi_square_critical)[0]
        print("Outliers Indices:", outliers)
        """
            st.code(code, language="python")  # Menampilkan kode Python dengan format yang jelas

         # Konten berdasarkan sub-bagian untuk Standarisasi
    
    # Konten berdasarkan sub-bagian untuk Standarisasi
    elif st.session_state.get("results") == 2:
        st.markdown("""
    <p style='text-align: justify; font-size: 20px;'>
        Standarisasi pada penelitian ini menggunakan <em>Maximum Absolute Scaler</em>.
    </p>
    """, unsafe_allow_html=True)

        # Menentukan file path secara langsung
        file_path = "D:/SKRIPSI/Data/Rasio Profit.xlsx"
        df = pd.read_excel(file_path)
         # Memisahkan kolom non-numerik dan numerik
        df_non_numeric = df.select_dtypes(exclude=[np.number])  # Kolom non-numerik sebagai indeks
        df_numeric = df.select_dtypes(include=[np.number])  # Hanya kolom numerik yang distandarisasi


        # Inisialisasi MaxAbsScaler
        scaler = MaxAbsScaler()

            # Melakukan standarisasi hanya pada kolom numerik
        Data_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

            # Menggunakan kolom non-numerik sebagai indeks
        Data_scaled.index = df_non_numeric.iloc[:, 0]  # Menggunakan kolom pertama non-numerik sebagai indeks
        Data_scaled.index.name = df_non_numeric.columns[0]  # Menjaga nama indeks tetap sesuai

            # Menampilkan hasil standarisasi
        st.markdown("### Hasil Standarisasi Data Rasio Profitabilitas:")
        st.write(Data_scaled)

        
    # Tombol navigasi
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state["results"] > 0:
            st.button("Back", on_click=prev_section2)

    with col2:
        if st.session_state["results"] < len(results) - 1:
            st.button("Next", on_click=next_section2)

    