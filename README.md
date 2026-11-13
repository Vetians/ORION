# SPK Rekomendasi Kegiatan Mahasiswa

Sistem Pendukung Keputusan (SPK) yang merekomendasikan kegiatan mahasiswa di luar perkuliahan berdasarkan profil akademik, komitmen akademik, keterlibatan organisasi, kemampuan komunikasi, keterampilan/prestasi, dan kebutuhan finansial. Metode keputusan yang digunakan adalah **SMARTER (ROC) + SAW (Profile Matching)**, lalu hasilnya dipelajari oleh model **Random Forest** dan disajikan melalui aplikasi web **Streamlit**.

Proyek mata kuliah Data Mining (Kelompok 12), menggunakan metodologi **CRISP-DM**.

> Aplikasi ini merupakan alat bantu pengambilan keputusan berbasis tugas akademik, bukan keputusan mutlak. Keputusan akhir tetap berada pada mahasiswa.

## Daftar Alternatif Kegiatan

1. **A1** – Organisasi & Kepemimpinan
2. **A2** – Volunteer & Sosial
3. **A3** – Unit Kegiatan Mahasiswa (UKM)
4. **A4** – Fokus Kuliah Saja
5. **A5** – Kuliah sambil Kerja Part-Time
6. **A6** – Kuliah sambil Kerja Full-Time
7. **A7** – Ikut Lomba

## Kriteria Keputusan & Pembentukan Indeks

| Kode | Kriteria | Indeks pada Dataset | Komponen Data Mentah | Prioritas SMARTER | Bobot ROC |
|---|---|---|---|---|---|
| **K1** | Performa akademik | `academic_performance_index` | `high_school_gpa`, `sat_score`, `act_score`, `attendance_rate` | 2 | 0,2417 |
| **K2** | Komitmen akademik | `academic_commitment_index` | `high_school_gpa`, `attendance_rate`, `ap_courses` | 6 | 0,0278 |
| **K3** | Keterlibatan organisasi | `organizational_engagement_index` | `extracurricular_count`, `volunteer_hours`, `leadership_positions` | 4 | 0,1028 |
| **K4** | Kemampuan komunikasi | `communication_profile_index` | `essay_score`, `recommendation_score`, `interview_score` | 5 | 0,0611 |
| **K5** | Skill / prestasi | `career_skill_index` | `coding_projects`, `online_certifications` | 1 | 0,4083 |
| **K6** | Kebutuhan finansial | `financial_need_index` | `family_income` $\rightarrow$ `relative_income` (`1 - MinMax`) | 3 | 0,1583 |

Setiap indeks pada **K1–K5** merupakan rata-rata nilai normalisasi Min-Max dari komponen atribut mentahnya. Khusus **K6 (`financial_need_index`)**, pendapatan keluarga ditransformasikan terlebih dahulu menjadi **`relative_income`** lalu dihitung dengan rumus `1 - MinMax(relative_income)`.

## Adaptasi Pendapatan Rupiah melalui `relative_income`

Dataset latih (`genz_college_admission_prediction.csv`) berasal dari konteks mahasiswa Amerika Serikat yang mencatat `family_income` sebagai pendapatan tahunan dalam USD (rentang setelah *winsorizing* IQR: **$2.124** hingga **$164.290,5**, median **$71.701**). Agar aplikasi dapat langsung digunakan oleh mahasiswa Indonesia dengan satuan **Rupiah per bulan (Rp/bulan)** tanpa terdistorsi oleh kurs nominal mata uang, sistem menggunakan fitur **`relative_income`**:

1. **Konsep Rasio Bebas Satuan (`relative_income`)**:
   Pendapatan keluarga dinyatakan sebagai rasio terhadap pendapatan acuan (*reference income*) di negara masing-masing:
   $$\text{relative\_income} = \frac{\text{Pendapatan Keluarga}}{\text{Reference Income}}$$
   * **Pada dataset AS (USD/tahun)**: $\text{Reference Income}_{\text{USD}} = \$83.730\text{/tahun}$, menghasilkan rentang *winsorized* $\text{relative\_income} \in [0,0254,\ 1,9621]$ dengan median $0,8563$.
   * **Pada aplikasi untuk pengguna Indonesia (Rp/bulan)**: Pengguna cukup mengisi pendapatan orang tua/keluarga per bulan dalam Rupiah, lalu dibagi dengan $\text{Reference Income}_{\text{IDR, bulanan}}$ (*default* **Rp 10.000.000/bulan**, dapat disesuaikan di menu pengaturan aplikasi).
2. **Konsistensi Perhitungan Bulanan vs Tahunan**:
   Karena `relative_income` adalah rasio, pembagian dengan pendapatan bulanan menghasilkan nilai yang **sama persis** dengan pembagian pendapatan tahunan:
   $$\frac{\text{Pendapatan Bulanan (Rp)} \times 12}{\text{Acuan Bulanan (Rp)} \times 12} = \frac{\text{Pendapatan Bulanan (Rp)}}{\text{Acuan Bulanan (Rp)}}$$
3. **Normalisasi ke `financial_need_index` (K6)**:
   Nilai `relative_income` pengguna dibatasi (*winsorized*) pada rentang $[0,0254,\ 1,9621]$ lalu dinormalisasi dengan Min-Max dan dibalik:
   $$\text{financial\_need\_index} = 1 - \text{clip}\left(\frac{\text{relative\_income} - 0,0254}{1,9621 - 0,0254},\ 0,\ 1\right)$$
   Dengan acuan *default* Rp 10.000.000/bulan, rentang efektif pendapatan keluarga adalah **Rp 253.672/bulan** (`financial_need_index = 1,000`) hingga **Rp 19.621.462/bulan** (`financial_need_index = 0,000`), dengan titik median setara **Rp 8.563.358/bulan** (`financial_need_index = 0,571`).

## Alur Kerja Sistem

1. **Pra-pemrosesan & Feature Engineering (`Data Transformasi dan Feature Engineering.py`)**: Memeriksa *missing value*, mendeteksi *outlier* dengan IQR, melakukan *winsorizing* pada `family_income`, menghitung `relative_income`, melakukan *One-Hot Encoding* dan normalisasi Min-Max, serta membentuk 6 indeks kriteria.
2. **Pemodelan Keputusan SMARTER + SAW (`modelingv2.py`)**:
   * Menghitung bobot kriteria dengan *Rank Order Centroid* (ROC) berdasarkan urutan prioritas $K_5 > K_1 > K_6 > K_3 > K_4 > K_2$.
   * Mengubah nilai indeks mahasiswa menjadi posisi relatif (persentil $p_j \in [0, 1]$ terhadap populasi dataset).
   * Menghitung tingkat kecocokan tiap alternatif kegiatan $c_{ij} = 1 - |p_j - r_{ij}|$ dan skor akhir SAW $V_i = \sum_{j=1}^{6} w_j c_{ij}$ untuk menghasilkan label rekomendasi utama (`recommendation_activities`).
3. **Pelatihan & Evaluasi Random Forest (`randomforestv2.py` & `evaluasi.py`)**:
   * Melatih `RandomForestClassifier` menggunakan 6 indeks mentah sebagai fitur ($X$) dan `recommendation_activities` sebagai target ($y$) dengan pembagian data 80:20 *stratified* serta *hyperparameter tuning* (`RandomizedSearchCV` 3-*fold* pada 200.000 sampel).
   * Menyimpan model terlatih ke dalam `random_forest_rekomendasi.pkl`.
4. **Aplikasi Web Streamlit (`app.py` & `spk_core.py`)**:
   * Menerima input 14 data mentah mahasiswa (termasuk opsi melewati skor SAT/ACT dan input pendapatan keluarga dalam **Rp/bulan**).
   * Menghitung 6 indeks kriteria menggunakan rumus yang identik dengan tahap *data preparation*, memetakan persentil melalui `spk_params.json`, lalu menampilkan 3 rekomendasi teratas dari Random Forest beserta perbandingan skor SAW langsung dan rincian kecocokan per kriteria.

## Struktur File

| Nama File | Fungsi Utama |
|---|---|
| `Data Transformasi dan Feature Engineering.py` | Pipeline *data preparation* CRISP-DM: *winsorizing* IQR, pembuatan `relative_income`, normalisasi Min-Max/Z-score, dan pembentukan 6 indeks kriteria |
| `modelingv2.py` | Perhitungan bobot SMARTER (ROC), matriks kecocokan, dan perankingan SAW pada seluruh dataset untuk menghasilkan `hasil_spk_smarter_saw2.csv` |
| `randomforestv2.py` | Pelatihan model Random Forest dengan `RandomizedSearchCV` (3-*fold* CV) dan penyimpanan model ke `random_forest_rekomendasi.pkl` |
| `evaluasi.py` | Evaluasi lengkap Random Forest: *classification report* per kelas, *confusion matrix*, parameter final, dan analisis *gap training vs testing* |
| `spk_core.py` | Modul inti SPK: konstanta `REFERENCE_INCOME_USD` & `REFERENCE_INCOME_IDR_MONTHLY`, konversi `relative_income`, perhitungan 6 indeks, bobot ROC, dan matriks kebutuhan SAW |
| `make_params.py` | Menghasilkan `spk_params.json` (rentang Min-Max, batas *winsorizing*, parameter `relative_income`, dan tabel kuantil 1.001 titik) dari dataset |
| `verify_indices.py` | Memverifikasi bahwa perhitungan 6 indeks di `spk_core.py` menghasilkan angka yang **sama persis** (selisih $< 10^{-6}$) dengan pipeline *data preparation* |
| `make_requirements.py` | Menghasilkan `requirements.txt` sesuai versi pustaka Python pada lingkungan pelatihan model |
| `app.py` | Aplikasi web interaktif berbasis Streamlit untuk pengguna akhir (mahasiswa) |
| `spk_params.json` | File parameter statistik dan kuantil hasil eksekusi `make_params.py` |
| `random_forest_rekomendasi.pkl` | Paket model Random Forest terlatih beserta daftar fitur dan *hyperparameter* terbaik |

## Panduan Menjalankan Secara Lokal

1. Pastikan folder kerja memuat seluruh skrip di atas beserta dataset:
   * `genz_college_admission_prediction.csv` (dataset mentah)
   * `hasil_spk_smarter_saw2.csv` (hasil pelabelan SMARTER + SAW dari `modelingv2.py`)
2. Jalankan pembuatan parameter dan verifikasi konsistensi indeks:
   ```bash
   python make_params.py
   python verify_indices.py
   python make_requirements.py
   ```
   Pastikan `verify_indices.py` menampilkan `HASIL: SAMA persis (aman dipakai).`
3. Instal dependensi dan jalankan aplikasi Streamlit:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Panduan Deployment (Streamlit Community Cloud)

1. Pastikan ukuran `random_forest_rekomendasi.pkl` berada di bawah batas GitHub (100 MB), atau gunakan Git LFS / kompresi `joblib.dump(..., compress=3)`.
2. Unggah 5 file berikut ke repositori GitHub (tidak perlu mengunggah file CSV berukuran besar):
   * `app.py`
   * `spk_core.py`
   * `spk_params.json`
   * `random_forest_rekomendasi.pkl`
   * `requirements.txt`
3. Hubungkan repositori ke **Streamlit Community Cloud** (`share.streamlit.io`), pilih `app.py` sebagai *entrypoint*, dan samakan versi Python dengan lingkungan pelatihan model.

## Ringkasan Evaluasi Model Random Forest

Evaluasi dilakukan pada 200.000 baris data uji (pembagian 80% *training* : 20% *testing* secara *stratified* dari 1.000.000 data):

| Metrik Evaluasi | Nilai |
|---|---|
| **Accuracy (Data Uji)** | **98,83%** (`0,9883`) |
| **Best CV F1-Weighted (3-Fold, 200.000 sampel tuning)** | **0,9771** |

Model Random Forest mempelajari aturan pencocokan profil SMARTER + SAW, sehingga metrik di atas mengukur ketepatan model dalam mereplikasi keputusan SPK tanpa harus mengurutkan ulang populasi 1 juta baris setiap kali pengguna baru meminta rekomendasi.

## Keterbatasan & Catatan Metodologis

* **Matriks kebutuhan alternatif & urutan prioritas ROC**: Ditentukan berdasarkan pertimbangan desain penyusun dan arah literatur, sehingga disarankan untuk diuji lebih lanjut melalui analisis sensitivitas bobot dan validasi pakar (dosen wali/kemahasiswaan).
* **Adaptasi konteks Indonesia**: Opsi melewati skor SAT/ACT menghitung `academic_performance_index` dari IPK dan kehadiran saja, sedangkan input pendapatan keluarga dalam Rupiah (Rp/bulan) dipetakan melalui rasio `relative_income` terhadap pendapatan acuan bulanan.
* **Tumpang tindih komponen kriteria**: `high_school_gpa` dan `attendance_rate` digunakan bersama pada K1 (Performa akademik) dan K2 (Komitmen akademik), dengan bobot K2 yang relatif kecil (`0,0278`).

## Anggota Kelompok 12

* **412024003** – Christian Melhan
* **412024004** – Felix Wong
