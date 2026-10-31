"""Buktikan bahwa perhitungan indeks di aplikasi SAMA dengan indeks di datamu.

Menghitung ulang 6 indeks dari data mentah memakai spk_core.compute_indices,
lalu membandingkannya dengan kolom indeks di hasil_spk_smarter_saw2.csv.
Jalankan SETELAH make_params.py.
"""
import json

import numpy as np
import pandas as pd

import spk_core as core

RAW_FILE = "genz_college_admission_prediction.csv"
INDEX_FILE = "hasil_spk_smarter_saw2.csv"

with open(core.PARAMS_FILE, encoding="utf-8") as f:
    params = json.load(f)

raw = pd.read_csv(RAW_FILE)
ref = pd.read_csv(INDEX_FILE)

if len(raw) != len(ref):
    raise SystemExit(
        f"Jumlah baris berbeda: mentah={len(raw):,}, hasil SPK={len(ref):,}. "
        "Baris harus sejajar agar bisa dibandingkan."
    )

# Gunakan student_id bila ada di kedua file; kalau tidak, andalkan urutan baris.
if "student_id" in raw.columns and "student_id" in ref.columns:
    same_order = bool((raw["student_id"].to_numpy() == ref["student_id"].to_numpy()).all())
    print(f"student_id sejajar dengan urutan baris: {same_order}")
    if not same_order:
        ref = ref.set_index("student_id").loc[raw["student_id"]].reset_index()
else:
    print("student_id tidak ada di salah satu file: membandingkan berdasarkan urutan baris.")

mine = core.compute_indices(raw, params)

print(f"\n{'indeks':36s}{'selisih maks':>14s}{'selisih rata2':>16s}")
ok = True
for col in core.INDEX_COLUMNS:
    diff = (mine[col].to_numpy() - ref[col].to_numpy())
    max_abs, mean_abs = float(np.abs(diff).max()), float(np.abs(diff).mean())
    ok &= max_abs < 1e-6
    print(f"{col:36s}{max_abs:14.2e}{mean_abs:16.2e}")

print("\nHASIL:", "SAMA persis (aman dipakai)." if ok else
      "ADA SELISIH. Rumus indeks di aplikasi berbeda dari pipeline datamu; "
      "kirim hasil ini agar kita cari penyebabnya.")
