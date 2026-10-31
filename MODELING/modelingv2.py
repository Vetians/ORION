import pandas as pd
import numpy as np


# ============================================================
# 1. KONFIGURASI
# ============================================================

INPUT_FILE = "genz_college_transformed.csv"
OUTPUT_FILE = "hasil_spk_smarter_saw2.csv"

# Jumlah rekomendasi utama yang ingin ditampilkan
TOP_N = 3


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOAD DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Jumlah baris : {len(df):,}")
print(f"Jumlah kolom : {len(df.columns):,}")


# ============================================================
# 3. DEFINISI KRITERIA SPK
# ============================================================

criteria = [
    "K1",
    "K2",
    "K3",
    "K4",
    "K5",
    "K6"
]

feature_mapping = {
    "K1": "academic_performance_index",
    "K2": "academic_commitment_index",
    "K3": "organizational_engagement_index",
    "K4": "communication_profile_index",
    "K5": "career_skill_index",
    "K6": "financial_need_index"
}


# ============================================================
# 4. CEK FEATURE YANG DIBUTUHKAN
# ============================================================

required_features = list(feature_mapping.values())

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Feature berikut tidak ditemukan dalam dataset:\n"
        + "\n".join(missing_features)
    )

print("\nFeature yang digunakan:")
for k, feature in feature_mapping.items():
    print(f"{k} -> {feature}")


# ============================================================
# 5. MEMBUAT PROFIL KRITERIA MAHASISWA
# ============================================================

student_features = df[required_features].copy()

student_features.columns = criteria

# Persentil: posisi relatif mahasiswa terhadap seluruh mahasiswa (0-1)
student_pct = student_features.rank(pct=True, method="average")
student_matrix = student_pct.to_numpy(dtype=np.float32)


# ============================================================
# 6. SMARTER - RANKING PRIORITAS KRITERIA
# ============================================================

# Prioritas:
# 1. Skill/Prestasi
# 2. Performa Akademik
# 3. Finansial
# 4. Organisasi
# 5. Komunikasi
# 6. Komitmen Akademik

priority_order = [
    "K5",
    "K1",
    "K6",
    "K3",
    "K4",
    "K2"
]


# ============================================================
# 7. SMARTER ROC WEIGHT
# ============================================================

n = len(priority_order)

roc_weights = {}

for rank, criterion in enumerate(priority_order, start=1):

    weight = sum(
        1 / j
        for j in range(rank, n + 1)
    ) / sum(
        sum(1 / j for j in range(i, n + 1))
        for i in range(1, n + 1)
    )

    roc_weights[criterion] = weight


# Urutkan sesuai K1-K6
weight_array = np.array(
    [roc_weights[c] for c in criteria],
    dtype=np.float32
)


print("\n" + "=" * 70)
print("BOBOT SMARTER")
print("=" * 70)

for criterion in criteria:
    print(
        f"{criterion} : "
        f"{roc_weights[criterion]:.4f}"
    )

print(
    f"\nTotal bobot : "
    f"{sum(roc_weights.values()):.4f}"
)


# ============================================================
# 8. ALTERNATIF KEGIATAN
# ============================================================

alternatives = [
    "Organisasi & Kepemimpinan",
    "Volunteer & Sosial",
    "UKM",
    "Fokus Kuliah Saja",
    "Kuliah sambil Kerja Part-Time",
    "Kuliah sambil Kerja Full-Time",
    "Ikut Lomba"
]


# ============================================================
# 9. REQUIREMENT MATRIX
# ============================================================

# Urutan kolom:
# K1, K2, K3, K4, K5, K6

requirement_matrix = np.array([
    # K1  K2  K3  K4  K5  K6

    [3, 3, 5, 5, 2, 1],   # Organisasi & Kepemimpinan

    [2, 2, 4, 4, 2, 1],   # Volunteer & Sosial

    [2, 2, 4, 3, 3, 1],   # UKM

    [5, 5, 1, 2, 2, 1],   # Fokus Kuliah Saja

    [3, 3, 1, 3, 3, 4],   # Kerja Part-Time

    [2, 2, 1, 4, 5, 5],   # Kerja Full-Time

    [4, 3, 2, 3, 5, 1]    # Ikut Lomba
], dtype=np.float32)


# ============================================================
# 10. NORMALISASI REQUIREMENT
# ============================================================

# Skala requirement 1-5
# diubah menjadi 0-1

requirement_array = requirement_matrix / 5.0


print("\n" + "=" * 70)
print("REQUIREMENT AKTIVITAS")
print("=" * 70)

requirement_df = pd.DataFrame(
    requirement_matrix,
    index=alternatives,
    columns=criteria
)

print(requirement_df)


# ============================================================
# 11. HITUNG COMPATIBILITY
# ============================================================

print("\n" + "=" * 70)
print("MENGHITUNG COMPATIBILITY")
print("=" * 70)

# Formula:
#
# Cij = 1 - |Student_j - Requirement_ij|
#
# Semakin mendekati 1 berarti semakin sesuai.

compatibility = (
    1
    - np.abs(
        student_matrix[:, None, :]
        - requirement_array[None, :, :]
    )
)


# ============================================================
# 12. HITUNG NILAI SAW
# ============================================================

# Formula:
#
# Vi = Σ (wj × Cij)

saw_scores = np.sum(
    compatibility
    * weight_array[None, None, :],
    axis=2
)


# ============================================================
# 13. SIMPAN SKOR SETIAP ALTERNATIF
# ============================================================

for i, alternative in enumerate(alternatives):

    column_name = f"SAW_{alternative}"

    df[column_name] = saw_scores[:, i]


# ============================================================
# 14. MENENTUKAN RANKING
# ============================================================

# Urutkan alternatif dari skor terbesar ke terkecil

ranking_index = np.argsort(
    -saw_scores,
    axis=1
)


# ============================================================
# 15. AMBIL SKOR BERDASARKAN RANKING
# ============================================================

ranked_scores = np.take_along_axis(
    saw_scores,
    ranking_index,
    axis=1
)


# ============================================================
# 16. MEMBUAT recommendation_1 SAMPAI recommendation_7
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT RANKING REKOMENDASI")
print("=" * 70)

for rank in range(7):

    df[f"recommendation_{rank + 1}"] = [
        alternatives[index]
        for index in ranking_index[:, rank]
    ]


# ============================================================
# 17. SIMPAN SCORE RANKING 1-7
# ============================================================

for rank in range(7):

    df[f"recommendation_{rank + 1}_score"] = (
        ranked_scores[:, rank]
    )


# ============================================================
# 18. recommendation_activities
# ============================================================

# recommendation_activities =
# rekomendasi utama / ranking pertama

df["recommendation_activities"] = (
    df["recommendation_1"]
)


# ============================================================
# 19. TAMBAHKAN NILAI K1-K6
# ============================================================

for criterion in criteria:

    df[criterion] = student_features[criterion]


# ============================================================
# 20. TAMPILKAN CONTOH HASIL
# ============================================================

print("\n" + "=" * 70)
print("CONTOH HASIL REKOMENDASI")
print("=" * 70)

display_columns = [
    "recommendation_activities",
    "recommendation_1",
    "recommendation_1_score",
    "recommendation_2",
    "recommendation_2_score",
    "recommendation_3",
    "recommendation_3_score",
    "recommendation_4",
    "recommendation_4_score",
    "recommendation_5",
    "recommendation_5_score",
    "recommendation_6",
    "recommendation_6_score",
    "recommendation_7",
    "recommendation_7_score"
]

print(
    df[display_columns].head(10).to_string(index=False)
)


# ============================================================
# 21. DISTRIBUSI REKOMENDASI UTAMA
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUSI REKOMENDASI UTAMA")
print("=" * 70)

recommendation_distribution = (
    df["recommendation_activities"]
    .value_counts()
)

recommendation_percentage = (
    df["recommendation_activities"]
    .value_counts(normalize=True)
    * 100
)

distribution_df = pd.DataFrame({
    "Jumlah": recommendation_distribution,
    "Persentase": recommendation_percentage
})

print(distribution_df)


# ============================================================
# 22. SIMPAN HASIL
# ============================================================

print("\n" + "=" * 70)
print("MENYIMPAN HASIL")
print("=" * 70)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"File berhasil disimpan:\n"
    f"{OUTPUT_FILE}"
)

print("\nJumlah data akhir:")
print(f"{len(df):,} baris")

print("\nJumlah kolom akhir:")
print(f"{len(df.columns):,} kolom")

print("\nSELESAI.")