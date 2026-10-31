"""Jalankan SEKALI di folder yang berisi dataset mentah dan hasil SPK.

Menghasilkan spk_params.json berisi:
- ranges   : min, max, median tiap fitur mentah (untuk Min-Max)
- income   : batas winsorizing IQR family_income + min/max setelah winsorizing
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
income = {
    "lower": lower,
    "upper": upper,
    "min": float(income_w.min()),
    "max": float(income_w.max()),
    "median": float(raw[core.INCOME_COLUMN].median()),
}

idx = pd.read_csv(INDEX_FILE, usecols=core.INDEX_COLUMNS)
quantiles = {
    col: np.quantile(idx[col].to_numpy(), core.QUANTILE_POINTS).tolist()
    for col in core.INDEX_COLUMNS
}

params = {"ranges": ranges, "income": income, "quantiles": quantiles}
with open(core.PARAMS_FILE, "w", encoding="utf-8") as f:
    json.dump(params, f)

print(f"Tersimpan: {core.PARAMS_FILE}\n")
print(f"{'fitur':24s}{'min':>12s}{'max':>12s}{'median':>12s}")
for col, p in ranges.items():
    print(f"{col:24s}{p['min']:12.2f}{p['max']:12.2f}{p['median']:12.2f}")
print(f"{'family_income (winsor)':24s}{income['min']:12.1f}{income['max']:12.1f}{income['median']:12.1f}")
print(f"\nBatas IQR income: lower={lower:,.1f}  upper={upper:,.1f}")
print("Bandingkan dengan dokumen: income winsor min 2124, max 164290,5.")
