"""Jalankan SEKALI di folder yang berisi dataset mentah dan hasil SPK.

Menghasilkan spk_params.json berisi:
- ranges   : min, max, median tiap fitur mentah (untuk Min-Max)
- income   : batas winsorizing IQR family_income, min/max setelah winsorizing,
             serta parameter relative_income (reference_usd, reference_idr_monthly,
             rel_min, rel_max, rel_median) untuk konversi Rupiah bulanan -> indeks K6
- quantiles: tabel persentil tiap indeks (untuk menampilkan posisi relatif & SAW)
"""
import json

import numpy as np
import pandas as pd

import spk_core as core

RAW_FILE = "genz_college_admission_prediction.csv"   # dataset mentah asli
INDEX_FILE = "hasil_spk_smarter_saw2.csv"             # berisi 6 kolom indeks

raw = pd.read_csv(RAW_FILE)
print(f"Dataset mentah: {len(raw):,} baris")

ranges = {
    col: {
        "min": float(raw[col].min()),
        "max": float(raw[col].max()),
        "median": float(raw[col].median()),
    }
    for col in core.RAW_COLUMNS
}

q1, q3 = raw[core.INCOME_COLUMN].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
income_w = raw[core.INCOME_COLUMN].clip(lower, upper)

ref_usd = float(core.REFERENCE_INCOME_USD)
ref_idr_monthly = float(core.REFERENCE_INCOME_IDR_MONTHLY)
rel_w = income_w / ref_usd

income = {
    "lower": lower,
    "upper": upper,
    "min": float(income_w.min()),
    "max": float(income_w.max()),
    "median": float(raw[core.INCOME_COLUMN].median()),
    "reference_usd": ref_usd,
    "reference_idr_monthly": ref_idr_monthly,
    "rel_min": float(rel_w.min()),
    "rel_max": float(rel_w.max()),
    "rel_median": float(raw[core.INCOME_COLUMN].median() / ref_usd),
}

idx = pd.read_csv(INDEX_FILE, usecols=core.INDEX_COLUMNS)
quantiles = {
    col: np.quantile(idx[col].to_numpy(), core.QUANTILE_POINTS).tolist()
    for col in core.INDEX_COLUMNS
}

params = {"ranges": ranges, "income": income, "quantiles": quantiles}
with open(core.PARAMS_FILE, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

print(f"Tersimpan: {core.PARAMS_FILE}\n")
print(f"{'fitur':26s}{'min':>12s}{'max':>12s}{'median':>12s}")
for col, p in ranges.items():
    print(f"{col:26s}{p['min']:12.2f}{p['max']:12.2f}{p['median']:12.2f}")
print(f"{'family_income (winsor)':26s}{income['min']:12.1f}{income['max']:12.1f}{income['median']:12.1f}")
print(f"{'relative_income (winsor)':26s}{income['rel_min']:12.4f}{income['rel_max']:12.4f}{income['rel_median']:12.4f}")
print(f"\nBatas IQR income (USD/thn): lower={lower:,.1f}  upper={upper:,.1f}")
print(f"Reference income AS       : ${core.format_ribuan(ref_usd)}/tahun")
print(
    f"Reference income Indonesia: {core.format_rupiah(ref_idr_monthly)}/bulan "
    f"(setara {core.format_rupiah(ref_idr_monthly * 12)}/tahun)"
)
print(
    f"Rentang efektif Rupiah/bln: {core.format_rupiah(income['rel_min'] * ref_idr_monthly)} "
    f"s/d {core.format_rupiah(income['rel_max'] * ref_idr_monthly)} "
    f"(median {core.format_rupiah(income['rel_median'] * ref_idr_monthly)})"
)