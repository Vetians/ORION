# SPK Rekomendasi Kegiatan Mahasiswa

Sistem pendukung keputusan (SPK) yang merekomendasikan kegiatan mahasiswa berdasarkan profil
akademik, organisasi, skill, komunikasi, dan kebutuhan finansial. Metode yang dipakai adalah
**SMARTER + SAW**, lalu hasilnya ditiru oleh model **Random Forest** dan disajikan lewat
aplikasi web **Streamlit**.

Proyek mata kuliah Data Mining (Kelompok 12), metodologi CRISP-DM.

> Aplikasi ini alat bantu berbasis tugas akademik, bukan keputusan final.

## Alternatif kegiatan

1. Organisasi & Kepemimpinan
2. Volunteer & Sosial
3. UKM
4. Fokus Kuliah Saja
5. Kuliah sambil Kerja Part-Time
6. Kuliah sambil Kerja Full-Time
7. Ikut Lomba

## Kriteria

| Kode | Kriteria | Indeks di dataset |
|---|---|---|
| K1 | Performa akademik | `academic_performance_index` |
| K2 | Komitmen akademik | `academic_commitment_index` |
| K3 | Keterlibatan organisasi | `organizational_engagement_index` |
| K4 | Kemampuan komunikasi | `communication_profile_index` |
| K5 | Skill / prestasi | `career_skill_index` |
| K6 | Kebutuhan finansial | `financial_need_index` |

Setiap indeks adalah rata-rata nilai Min-Max dari komponen datanya. `financial_need_index`
adalah 1 dikurangi Min-Max pendapatan keluarga yang sudah di-winsorize dengan IQR.

## Cara kerja

1. **Profil kebutuhan kegiatan.** Setiap kegiatan punya tingkat kebutuhan (skala 1-5, lalu dibagi 5)
   terhadap K1-K6. Matriks ini adalah penilaian penyusun berdasarkan arah literatur, bukan angka
   yang dilaporkan sumber mana pun, dan perlu divalidasi.
2. **Bobot SMARTER (ROC).** Urutan prioritas: K5, K1, K6, K3, K4, K2. Bobot dihitung dengan
   Rank Order Centroid (n = 6).
3. **Posisi relatif (persentil).** Nilai indeks mahasiswa diubah menjadi persentil terhadap seluruh
   mahasiswa di dataset, agar skalanya sebanding dengan skala kebutuhan kegiatan.
4. **Kecocokan dan SAW.** `Kecocokan = 1 - |nilai mahasiswa - kebutuhan kegiatan|`, lalu skor akhir
   `V = Σ (bobot × kecocokan)`. Kegiatan dengan skor tertinggi menjadi label rekomendasi.
5. **Random Forest.** Model dilatih untuk memprediksi label SAW dari enam indeks. Persentil tidak
   memengaruhi pohon keputusan, sehingga model memakai indeks mentah sebagai fitur.
6. **Aplikasi.** Pengguna mengisi data mentah, aplikasi menghitung enam indeks dengan rumus yang
   sama dengan data preparation, lalu menampilkan rekomendasi Random Forest beserta penjelasan
   SAW langsung.

## Struktur file

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `spk_core.py` | Rumus indeks, matriks kebutuhan, bobot ROC (dipakai bersama oleh semua skrip) |
| `make_params.py` | Membuat `spk_params.json` dari dataset (rentang Min-Max, batas winsorizing, tabel persentil) |
| `verify_indices.py` | Memastikan perhitungan indeks di aplikasi sama dengan indeks di dataset |
| `make_requirements.py` | Membuat `requirements.txt` dengan versi library sesuai lingkungan training |
| `random_forest_rekomendasi.pkl` | Model Random Forest hasil training |
| `spk_params.json` | Parameter hasil `make_params.py` |
| `requirements.txt` | Dependensi hasil `make_requirements.py` |

## Menjalankan secara lokal

Siapkan satu folder yang berisi semua file di atas, ditambah:

- `genz_college_admission_prediction.csv` (dataset mentah)
- `hasil_spk_smarter_saw2.csv` (hasil SMARTER + SAW dengan kolom indeks)

Lalu jalankan:

```
python make_params.py
python verify_indices.py
python make_requirements.py
pip install -r requirements.txt
streamlit run app.py
```

`verify_indices.py` harus menampilkan "SAMA persis". Jika ada selisih, rumus indeks di aplikasi
berbeda dari pipeline data preparation dan perlu diperiksa sebelum aplikasi dipakai.

## Deployment (Streamlit Community Cloud)

1. Pastikan ukuran `random_forest_rekomendasi.pkl` di bawah 100 MB (batas file GitHub). Jika lebih
   besar, latih ulang dengan parameter yang lebih kecil (misalnya `max_depth` dan
   `min_samples_leaf` lebih ketat) atau simpan dengan `joblib.dump(..., compress=3)`.
2. Unggah ke repo GitHub: `app.py`, `spk_core.py`, `spk_params.json`, `requirements.txt`,
   dan `random_forest_rekomendasi.pkl`. Jangan unggah CSV berukuran besar.
3. Buka share.streamlit.io, pilih repo dan `app.py`, lalu deploy. Samakan versi Python dengan yang
   dipakai saat training.

## Hasil evaluasi Random Forest

Pada data uji 200.000 baris (split 80/20, stratified):

| Metrik | Nilai |
|---|---|
| Accuracy | 0,9883 |
| Best CV F1 (3-fold, sampel tuning 200.000) | 0,9771 |

Model meniru label SMARTER + SAW, sehingga metrik ini menunjukkan seberapa baik model menirukan
rumus tersebut, bukan kualitas rekomendasi itu sendiri. Kualitas rekomendasi bergantung pada
matriks kebutuhan dan urutan prioritas kriteria.

## Keterbatasan

- **Matriks kebutuhan dan urutan ROC** adalah keputusan desain penyusun, bukan hasil data.
  Usulan validasi: analisis sensitivitas urutan ROC dan pemeriksaan beberapa profil contoh oleh
  dosen atau kakak tingkat.
- **Persentil bersifat relatif** terhadap populasi dataset, bukan terhadap mahasiswa di kampus tertentu.
- **Dataset latih** adalah data mahasiswa Amerika. SAT, ACT, AP, dan satuan pendapatan mengikuti
  dataset tersebut. Aplikasi menyediakan opsi tanpa SAT/ACT, yang menghitung indeks performa
  akademik dari IPK dan kehadiran saja dan sedikit menyimpang dari data latih.
- **Tumpang tindih kriteria.** K1 dan K2 sama-sama memuat GPA dan kehadiran.
- **Ukuran model** dapat menjadi kendala pada hosting gratis.
- File `.pkl` hanya boleh dimuat dari sumber yang dipercaya.

## Anggota kelompok

- 412024003 - Christian Melhan
- 412024004 - Felix Wong