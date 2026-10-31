"""Inti perhitungan SPK: dipakai bersama oleh app.py, make_params.py, dan verify_indices.py.

Aturan pembentukan indeks mengikuti dokumen Feature Engineering:
- indeks = rata-rata nilai Min-Max dari komponen mentahnya
- financial_need_index = 1 - Min-Max(family_income yang sudah di-winsorize IQR)
"""
import numpy as np
import pandas as pd

MODEL_FILE = "random_forest_rekomendasi.pkl"
PARAMS_FILE = "spk_params.json"

# ------------------------------------------------------------
# Fitur mentah
# ------------------------------------------------------------
RAW_COLUMNS = [
    "high_school_gpa", "sat_score", "act_score", "attendance_rate",
    "ap_courses", "extracurricular_count", "volunteer_hours",
    "leadership_positions", "coding_projects", "online_certifications",
    "essay_score", "recommendation_score", "interview_score",
]
INCOME_COLUMN = "family_income"
SAT_ACT = ("sat_score", "act_score")

COMPONENTS = {
    "academic_performance_index": ["high_school_gpa", "sat_score", "act_score", "attendance_rate"],
    "academic_commitment_index": ["high_school_gpa", "attendance_rate", "ap_courses"],
    "organizational_engagement_index": ["extracurricular_count", "volunteer_hours", "leadership_positions"],
    "career_skill_index": ["coding_projects", "online_certifications"],
    "communication_profile_index": ["essay_score", "recommendation_score", "interview_score"],
}
INDEX_COLUMNS = list(COMPONENTS) + ["financial_need_index"]

# ------------------------------------------------------------
# Tampilan form: kolom -> (label, bantuan, min_ui, max_ui, step, integer?)
# ------------------------------------------------------------
FIELDS = {
    "high_school_gpa": ("IPK / GPA (skala 4,0)", "Nilai rata-rata akademik.", 0.0, 4.0, 0.01, False),
    "sat_score": ("Skor SAT", "Rentang 400-1600. Kosongkan lewat kotak centang di atas bila tidak ada.", 400, 1600, 10, True),
    "act_score": ("Skor ACT", "Rentang 1-36. Kosongkan lewat kotak centang di atas bila tidak ada.", 1, 36, 1, True),
    "attendance_rate": ("Tingkat kehadiran (%)", "Persentase kehadiran.", 0.0, 100.0, 0.1, False),
    "ap_courses": ("Jumlah mata pelajaran AP / kelas akselerasi", "Jumlah kelas tingkat lanjut yang diikuti.", 0, 50, 1, True),
    "extracurricular_count": ("Jumlah kegiatan ekstrakurikuler/organisasi", "Jumlah kegiatan yang diikuti.", 0, 50, 1, True),
    "volunteer_hours": ("Total jam volunteer", "Total jam kegiatan sukarela.", 0, 5000, 1, True),
    "leadership_positions": ("Jumlah posisi kepemimpinan", "Posisi/peran kepemimpinan yang pernah dipegang.", 0, 50, 1, True),
    "coding_projects": ("Jumlah proyek pemrograman", "Proyek yang pernah dikerjakan.", 0, 50, 1, True),
    "online_certifications": ("Jumlah sertifikasi online", "Sertifikasi yang dimiliki.", 0, 50, 1, True),
    "essay_score": ("Nilai esai (0-100)", "Nilai esai seleksi.", 0.0, 100.0, 0.1, False),
    "recommendation_score": ("Nilai rekomendasi (0-100)", "Nilai surat/penilaian rekomendasi.", 0.0, 100.0, 0.1, False),
    "interview_score": ("Nilai wawancara (0-100)", "Nilai wawancara seleksi.", 0.0, 100.0, 0.1, False),
    "family_income": ("Pendapatan keluarga", "Dalam satuan yang sama dengan dataset (lihat catatan di bawah form).", 0, 10_000_000, 1000, True),
}

GROUPS = {
    "Akademik": ["high_school_gpa", "sat_score", "act_score", "attendance_rate", "ap_courses"],
    "Organisasi & kegiatan": ["extracurricular_count", "volunteer_hours", "leadership_positions"],
    "Skill & karier": ["coding_projects", "online_certifications"],
    "Komunikasi & profil": ["essay_score", "recommendation_score", "interview_score"],
    "Keuangan": ["family_income"],
}

# ------------------------------------------------------------
# Kriteria SPK (urutan K1..K6 sama dengan matriks kebutuhan)
# ------------------------------------------------------------
KEYS = ["K1", "K2", "K3", "K4", "K5", "K6"]
CRITERIA = {
    "K1": ("academic_performance_index", "Performa akademik"),
    "K2": ("academic_commitment_index", "Komitmen akademik"),
    "K3": ("organizational_engagement_index", "Keterlibatan organisasi"),
    "K4": ("communication_profile_index", "Kemampuan komunikasi"),
    "K5": ("career_skill_index", "Skill / prestasi"),
    "K6": ("financial_need_index", "Kebutuhan finansial"),
}

ALTERNATIVES = [
    "Organisasi & Kepemimpinan",
    "Volunteer & Sosial",
    "UKM",
    "Fokus Kuliah Saja",
    "Kuliah sambil Kerja Part-Time",
    "Kuliah sambil Kerja Full-Time",
    "Ikut Lomba",
]

# Baris = alternatif, kolom = K1..K6 (skala 1-5)
REQUIREMENT = np.array([
    [3, 3, 5, 5, 2, 1],
    [2, 2, 4, 4, 2, 1],
    [2, 2, 4, 3, 3, 1],
    [5, 5, 1, 2, 2, 1],
    [3, 3, 1, 3, 3, 4],
    [2, 2, 1, 4, 5, 5],
    [4, 3, 2, 3, 5, 1],
], dtype=float) / 5.0

PRIORITY = ["K5", "K1", "K6", "K3", "K4", "K2"]
_n = len(PRIORITY)
_roc = {c: sum(1 / j for j in range(r, _n + 1)) / _n for r, c in enumerate(PRIORITY, start=1)}
WEIGHTS = np.array([_roc[k] for k in KEYS])

QUANTILE_POINTS = np.linspace(0, 1, 1001)


# ------------------------------------------------------------
# Fungsi perhitungan
# ------------------------------------------------------------
def _minmax(series, lo, hi):
    return ((series.astype(float) - lo) / (hi - lo)).clip(0, 1)


def compute_indices(raw: pd.DataFrame, params: dict, skip=()) -> pd.DataFrame:
    """Hitung enam indeks dari data mentah.

    skip: kolom komponen yang dilewati (mis. SAT/ACT bila tidak tersedia);
    rata-rata dihitung dari komponen yang tersisa.
    """
    norm = {}
    for col in RAW_COLUMNS:
        if col in raw.columns:
            p = params["ranges"][col]
            norm[col] = _minmax(raw[col], p["min"], p["max"])

    inc = params["income"]
    income_w = raw[INCOME_COLUMN].astype(float).clip(inc["lower"], inc["upper"])
    income_norm = _minmax(income_w, inc["min"], inc["max"])

    out = pd.DataFrame(index=raw.index)
    for index_name, comps in COMPONENTS.items():
        used = [c for c in comps if c not in skip]
        out[index_name] = pd.concat([norm[c] for c in used], axis=1).mean(axis=1)
    out["financial_need_index"] = 1 - income_norm
    return out[INDEX_COLUMNS]


def to_percentile(value: float, quantile_table) -> float:
    """Posisi relatif (0-1) sebuah nilai indeks terhadap populasi dataset."""
    q = np.asarray(quantile_table, dtype=float)
    left = np.searchsorted(q, value, side="left")
    right = np.searchsorted(q, value, side="right")
    return float(np.clip((left + right) / 2 / (len(q) - 1), 0, 1))


def out_of_range(values: dict, params: dict, skip=()):
    """Daftar fitur yang nilainya di luar rentang dataset (akan dibatasi ke tepi rentang)."""
    flagged = []
    for col, p in params["ranges"].items():
        if col in skip or col not in values:
            continue
        if values[col] < p["min"] or values[col] > p["max"]:
            flagged.append(FIELDS[col][0])
    inc = params["income"]
    if INCOME_COLUMN in values and (values[INCOME_COLUMN] > inc["upper"] or values[INCOME_COLUMN] < inc["min"]):
        flagged.append(FIELDS[INCOME_COLUMN][0])
    return flagged
