# ============================================================
# EVALUASI LENGKAP RANDOM FOREST
# Menggunakan model final dari random_forest_rekomendasi.pkl
# Tidak melakukan training ulang atau hyperparameter tuning.
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. KONFIGURASI DAN LOKASI FILE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "hasil_spk_smarter_saw2.csv"
MODEL_FILE = BASE_DIR / "random_forest_rekomendasi.pkl"
OUTPUT_DIR = BASE_DIR / "hasil_evaluasi"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
# Mengikuti konfigurasi training di randomforestv2.py.
# Nilai random_state diambil dari paket model.

print("=" * 70)
print("EVALUASI RANDOM FOREST - MODEL FINAL")
print("=" * 70)

# ============================================================
# 2. MEMUAT MODEL DAN INFORMASI MODEL
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_FILE}\n"
        "Pastikan random_forest_rekomendasi.pkl berada "
        "di folder yang sama dengan evaluasi.py."
    )

model_package = joblib.load(MODEL_FILE)

if not isinstance(model_package, dict):
    raise TypeError(
        "Isi file PKL bukan dictionary model_package. "
        "Periksa kembali format penyimpanan model."
    )

required_keys = ["model", "features", "target", "best_params", "random_state"]
missing_keys = [key for key in required_keys if key not in model_package]

if missing_keys:
    raise KeyError(
        f"Informasi berikut tidak ditemukan di model PKL: {missing_keys}"
    )

model = model_package["model"]
features = model_package["features"]
target = model_package["target"]
best_params = model_package["best_params"]
RANDOM_STATE = model_package["random_state"]

print("\nModel berhasil dimuat.")
print(f"Algoritma    : {type(model).__name__}")
print(f"Random state : {RANDOM_STATE}")
print(f"Target       : {target}")
print(f"Jumlah fitur : {len(features)}")

# ============================================================
# 3. MEMUAT DATASET
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {INPUT_FILE}\n"
        "Pastikan hasil_spk_smarter_saw2.csv berada "
        "di folder yang sama dengan evaluasi.py."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset berhasil dimuat.")
print(f"Jumlah baris awal : {len(df):,}")
print(f"Jumlah kolom      : {len(df.columns):,}")

# ============================================================
# 4. VALIDASI KOLOM DAN MENYIAPKAN X DAN y
# ============================================================

required_columns = list(features) + [target]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        "Kolom yang dibutuhkan model tidak ditemukan:\n"
        + "\n".join(missing_columns)
    )

X = df[features].copy()
y = df[target].copy()

# Mengikuti preprocessing pada randomforestv2.py:
# baris dengan missing value pada fitur atau target dihapus.
valid_rows = X.notnull().all(axis=1) & y.notnull()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print(f"Jumlah baris setelah menghapus missing value: {len(X):,}")

if len(X) == 0:
    raise ValueError("Tidak ada data tersisa setelah menghapus missing value.")

# Pastikan urutan fitur sama dengan saat model dilatih.
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if model_features != list(features):
        raise ValueError(
            "Urutan/nama fitur dalam model berbeda dengan fitur pada paket PKL.\n"
            f"Fitur model : {model_features}\n"
            f"Fitur paket : {list(features)}"
        )

# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Data training : {len(X_train):,} ({len(X_train) / len(X) * 100:.2f}%)")
print(f"Data testing  : {len(X_test):,} ({len(X_test) / len(X) * 100:.2f}%)")

# ============================================================
# 6. PREDIKSI TRAINING DAN TESTING
# ============================================================

print("\n" + "=" * 70)
print("PREDIKSI TRAINING DAN TESTING")
print("=" * 70)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Prediksi training dan testing selesai.")

# ============================================================
# 7. CLASSIFICATION REPORT PER KELAS
# ============================================================

report = classification_report(
    y_test,
    y_test_pred,
    labels=model.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T
report_df.index.name = "Kelas"

for column in ["precision", "recall", "f1-score"]:
    if column in report_df.columns:
        report_df[column] = report_df[column].round(4)

if "support" in report_df.columns:
    report_df["support"] = report_df["support"].fillna(0).astype(int)

report_path = OUTPUT_DIR / "evaluasi_classification_report.csv"
report_df.to_csv(report_path, encoding="utf-8-sig")

print("\n--- CLASSIFICATION REPORT ---")
print(report_df.to_string())

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.index.name = "Label Aktual"
cm_df.columns.name = "Label Prediksi"

cm_path = OUTPUT_DIR / "evaluasi_confusion_matrix.csv"
cm_df.to_csv(cm_path, encoding="utf-8-sig")

print("\n--- CONFUSION MATRIX ---")
print(cm_df)

fig, ax = plt.subplots(figsize=(12, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

ax.set_title("Confusion Matrix - Random Forest")
plt.tight_layout()

cm_image_path = OUTPUT_DIR / "evaluasi_confusion_matrix.png"
plt.savefig(cm_image_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 9. PARAMETER RANDOM FOREST FINAL
# ============================================================

parameter_final = {
    "algoritma": type(model).__name__,
    "fitur": list(features),
    "target": target,
    "random_state_split": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "stratify": True,
    "best_params_hasil_tuning": best_params,
    "parameter_model_final": model.get_params()
}

parameter_path = OUTPUT_DIR / "parameter_random_forest_final.json"

with open(parameter_path, "w", encoding="utf-8") as file:
    json.dump(
        parameter_final,
        file,
        indent=4,
        ensure_ascii=False,
        default=str
    )

print("\n--- PARAMETER RANDOM FOREST FINAL ---")
print(json.dumps(parameter_final, indent=4, ensure_ascii=False, default=str))

# ============================================================
# 10. METRIK TRAINING VS TESTING
# ============================================================

def hitung_metrik(y_asli, y_prediksi):
    return {
        "accuracy": accuracy_score(y_asli, y_prediksi),
        "precision_weighted": precision_score(
            y_asli, y_prediksi, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_asli, y_prediksi, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_asli, y_prediksi, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            y_asli, y_prediksi, average="macro", zero_division=0
        )
    }

metrik_train = hitung_metrik(y_train, y_train_pred)
metrik_test = hitung_metrik(y_test, y_test_pred)

perbandingan = pd.DataFrame(
    [metrik_train, metrik_test],
    index=["Training", "Testing"]
)

perbandingan.loc["Gap Train-Test"] = (
    perbandingan.loc["Training"] - perbandingan.loc["Testing"]
)

perbandingan.index.name = "Dataset"

comparison_path = OUTPUT_DIR / "evaluasi_training_vs_testing.csv"
perbandingan.to_csv(comparison_path, encoding="utf-8-sig")

print("\n--- PERBANDINGAN TRAINING VS TESTING ---")
print(perbandingan.round(4).to_string())

# ============================================================
# 11. INTERPRETASI AWAL GAP
# ============================================================

gap_accuracy = metrik_train["accuracy"] - metrik_test["accuracy"]
gap_f1 = metrik_train["f1_weighted"] - metrik_test["f1_weighted"]

print("\n--- ANALISIS GAP ---")
print(f"Accuracy training : {metrik_train['accuracy']:.4f}")
print(f"Accuracy testing  : {metrik_test['accuracy']:.4f}")
print(f"Gap accuracy      : {gap_accuracy:.4f}")
print(f"Gap F1 weighted   : {gap_f1:.4f}")

if gap_accuracy > 0.05:
    print(
        "Catatan: gap accuracy lebih dari 5 poin persentase. "
        "Ini dapat menjadi indikasi overfitting dan perlu diperiksa lebih lanjut."
    )
elif gap_accuracy > 0.02:
    print(
        "Catatan: terdapat gap accuracy. "
        "Evaluasi generalisasi model lebih lanjut diperlukan."
    )
else:
    print(
        "Catatan: gap accuracy relatif kecil. "
        "Tetap periksa metrik testing dan distribusi kelas."
    )

# ============================================================
# 12. RINGKASAN FILE HASIL
# ============================================================

print("\n" + "=" * 70)
print("EVALUASI SELESAI")
print("=" * 70)

print(f"Folder hasil: {OUTPUT_DIR}")
print("- evaluasi_classification_report.csv")
print("- evaluasi_confusion_matrix.csv")
print("- evaluasi_confusion_matrix.png")
print("- parameter_random_forest_final.json")
print("- evaluasi_training_vs_testing.csv")

print("\nCatatan: script ini memuat model final dari PKL dan tidak")
print("menjalankan ulang RandomizedSearchCV maupun model.fit().")
