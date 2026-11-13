"""Inti perhitungan SPK: dipakai bersama oleh app.py, make_params.py, dan verify_indices.py.

Aturan pembentukan indeks mengikuti dokumen Feature Engineering:
- indeks K1..K5 = rata-rata nilai Min-Max dari komponen mentahnya
- relative_income = family_income_winsorized / reference_income
  * Untuk dataset AS (USD/tahun): dibagi REFERENCE_INCOME_USD ($83.730/tahun)
  * Untuk pengguna Indonesia (Rp/bulan): dibagi REFERENCE_INCOME_IDR_MONTHLY
    (sama nilainya dengan perhitungan tahunan karena faktor 12 saling menghilangkan:
    (gaji_bulanan * 12) / (acuan_bulanan * 12) = gaji_bulanan / acuan_bulanan)
- financial_need_index = 1 - Min-Max(relative_income)
"""
import numpy as np
import pandas as pd

MODEL_FILE = "random_forest_rekomendasi.pkl"
PARAMS_FILE = "spk_params.json"

# ------------------------------------------------------------
# Konstanta Reference Income (untuk perhitungan Relative Income)
# ------------------------------------------------------------
# Dataset asli menggunakan pendapatan keluarga tahunan dalam USD
REFERENCE_INCOME_USD = 83_730.0

# Acuan pendapatan keluarga bulanan di Indonesia (Rp/bulan) untuk penyetaraan
# relative_income (1,0x reference income). Menggunakan satuan bulanan karena
# lebih akrab bagi mahasiswa di Indonesia dan secara matematis identik dengan
# rasio tahunan: (Rp_bulan * 12) / (Acuan_bulan * 12) == Rp_bulan / Acuan_bulan.
REFERENCE_INCOME_IDR_MONTHLY = 10_000_000.0

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
# Untuk family_income, UI default menggunakan Rupiah per bulan (Rp/bulan).
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
    "family_income": (
        "Pendapatan keluarga per bulan (Rp)",
        "Total pendapatan orang tua/keluarga per bulan dalam Rupiah (contoh: 8.500.000). Boleh memakai titik sebagai pemisah ribuan.",
        0,
        200_000_000,
        500_000,
        True,
    ),
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
# Fungsi bantu pemformatan Rupiah (pemisah titik setiap 3 digit)
# ------------------------------------------------------------
def format_ribuan(value: float | int) -> str:
    """Format angka dengan pemisah titik setiap 3 digit (contoh: 10000000 -> '10.000.000')."""
    return f"{int(round(float(value))):,}".replace(",", ".")


def format_rupiah(value: float | int) -> str:
    """Format angka menjadi string Rupiah dengan pemisah titik (contoh: 'Rp 10.000.000')."""
    return f"Rp {format_ribuan(value)}"


def parse_ribuan(text: str | int | float) -> int:
    """Ubah input string berformat titik/koma (mis. '10.000.000') menjadi integer."""
    if isinstance(text, (int, float)):
        return int(round(float(text)))
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if not digits:
        raise ValueError("Input pendapatan harus berisi angka.")
    return int(digits)


# ------------------------------------------------------------
# Fungsi perhitungan Relative Income & Indeks
# ------------------------------------------------------------
def _minmax(series, lo, hi):
    return ((series.astype(float) - lo) / (hi - lo)).clip(0, 1)


def get_relative_income_bounds(params: dict) -> tuple[float, float, float, float]:
    """Kembalikan (rel_min, rel_max, rel_median, ref_usd) dari params['income']."""
    inc = params["income"]
    ref_usd = float(inc.get("reference_usd", REFERENCE_INCOME_USD))
    rel_min = float(inc.get("rel_min", inc["min"] / ref_usd))
    rel_max = float(inc.get("rel_max", inc["max"] / ref_usd))
    rel_median = float(inc.get("rel_median", inc["median"] / ref_usd))
    return rel_min, rel_max, rel_median, ref_usd


def compute_relative_income(
    income_series: pd.Series,
    params: dict,
    income_unit: str = "usd",
    reference_idr_monthly: float = REFERENCE_INCOME_IDR_MONTHLY,
) -> pd.Series:
    """Hitung relative_income yang sudah di-winsorize sesuai satuan input.

    - income_unit == 'usd': input dalam USD tahunan (dipakai oleh dataset asli & verify_indices.py).
      Winsorizing dilakukan pada batas IQR USD, lalu dibagi REFERENCE_INCOME_USD.
    - income_unit == 'idr_monthly': input dalam Rupiah per bulan (dipakai oleh pengguna di app.py).
      Rasio dihitung terhadap reference_idr_monthly (identik dengan rasio tahunan), lalu
      di-winsorize pada rentang [rel_min, rel_max] dataset.
    """
    inc = params["income"]
    rel_min, rel_max, _, ref_usd = get_relative_income_bounds(params)

    if income_unit == "idr_monthly":
        ref_idr = float(reference_idr_monthly) if reference_idr_monthly > 0 else REFERENCE_INCOME_IDR_MONTHLY
        rel_raw = income_series.astype(float) / ref_idr
        return rel_raw.clip(rel_min, rel_max)
    else:
        income_w = income_series.astype(float).clip(inc["lower"], inc["upper"])
        return income_w / ref_usd


def compute_indices(
    raw: pd.DataFrame,
    params: dict,
    skip=(),
    income_unit: str = "usd",
    reference_idr_monthly: float = REFERENCE_INCOME_IDR_MONTHLY,
) -> pd.DataFrame:
    """Hitung enam indeks dari data mentah.

    - skip: kolom komponen yang dilewati (mis. SAT/ACT bila tidak tersedia);
      rata-rata dihitung dari komponen yang tersisa.
    - income_unit: 'usd' (default untuk dataset asli) atau 'idr_monthly' (input Rupiah bulanan di app.py).
    """
    norm = {}
    for col in RAW_COLUMNS:
        if col in raw.columns:
            p = params["ranges"][col]
            norm[col] = _minmax(raw[col], p["min"], p["max"])

    rel_min, rel_max, _, _ = get_relative_income_bounds(params)
    rel_income = compute_relative_income(
        raw[INCOME_COLUMN],
        params,
        income_unit=income_unit,
        reference_idr_monthly=reference_idr_monthly,
    )
    rel_income_norm = _minmax(rel_income, rel_min, rel_max)

    out = pd.DataFrame(index=raw.index)
    for index_name, comps in COMPONENTS.items():
        used = [c for c in comps if c not in skip]
        out[index_name] = pd.concat([norm[c] for c in used], axis=1).mean(axis=1)
    out["financial_need_index"] = 1.0 - rel_income_norm
    return out[INDEX_COLUMNS]


def to_percentile(value: float, quantile_table) -> float:
    """Posisi relatif (0-1) sebuah nilai indeks terhadap populasi dataset."""
    q = np.asarray(quantile_table, dtype=float)
    left = np.searchsorted(q, value, side="left")
    right = np.searchsorted(q, value, side="right")
    return float(np.clip((left + right) / 2 / (len(q) - 1), 0, 1))


def out_of_range(
    values: dict,
    params: dict,
    skip=(),
    income_unit: str = "usd",
    reference_idr_monthly: float = REFERENCE_INCOME_IDR_MONTHLY,
):
    """Daftar fitur yang nilainya di luar rentang dataset (akan dibatasi ke tepi rentang)."""
    flagged = []
    for col, p in params["ranges"].items():
        if col in skip or col not in values:
            continue
        if values[col] < p["min"] or values[col] > p["max"]:
            flagged.append(FIELDS[col][0])

    if INCOME_COLUMN in values:
        if income_unit == "idr_monthly":
            rel_min, rel_max, _, _ = get_relative_income_bounds(params)
            ref_idr = float(reference_idr_monthly) if reference_idr_monthly > 0 else REFERENCE_INCOME_IDR_MONTHLY
            rel_val = float(values[INCOME_COLUMN]) / ref_idr
            if rel_val < rel_min or rel_val > rel_max:
                flagged.append(FIELDS[INCOME_COLUMN][0])
        else:
            inc = params["income"]
            if values[INCOME_COLUMN] > inc["upper"] or values[INCOME_COLUMN] < inc["min"]:
                flagged.append(FIELDS[INCOME_COLUMN][0])
    return flagged