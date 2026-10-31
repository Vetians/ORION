
# ============================================================
# EVALUASI LENGKAP RANDOM FOREST
# (a) Precision, Recall, F1-score per kelas
# (b) Confusion Matrix
# (c) Parameter Random Forest final
# (d) Perbandingan Training dan Testing
# ============================================================

import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. PREDIKSI DATA TRAINING DAN TESTING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUASI TRAINING DAN TESTING")
print("=" * 70)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Prediksi training dan testing selesai.")

# ------------------------------------------------------------
# 2. CLASSIFICATION REPORT PER KELAS
# ------------------------------------------------------------

report = classification_report(
    y_test,
    y_test_pred,
    labels=model.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T

report_df.index.name = "Kelas"

# Membulatkan metrik, support tetap bilangan bulat
for kolom in ["precision", "recall", "f1-score"]:
    if kolom in report_df.columns:
        report_df[kolom] = report_df[kolom].round(4)

if "support" in report_df.columns:
    report_df["support"] = report_df["support"].fillna(0).astype(int)

report_df.to_csv(
    "evaluasi_classification_report.csv",
    encoding="utf-8-sig"
)

print("\n--- CLASSIFICATION REPORT ---")
print(report_df.to_string())

# ------------------------------------------------------------
# 3. CONFUSION MATRIX
# ------------------------------------------------------------

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

cm_df.to_csv(
    "evaluasi_confusion_matrix.csv",
    encoding="utf-8-sig"
)

print("\n--- CONFUSION MATRIX ---")
print(cm_df)

# Simpan visualisasi confusion matrix
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

plt.savefig(
    "evaluasi_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# 4. PARAMETER RANDOM FOREST FINAL
# ------------------------------------------------------------

parameter_final = {
    "algoritma": "RandomForestClassifier",
    "fitur": list(features),
    "target": target,
    "random_state_split": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "best_params_hasil_tuning": best_params,
    "parameter_model_final": model.get_params()
}

with open(
    "parameter_random_forest_final.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        parameter_final,
        file,
        indent=4,
        ensure_ascii=False,
        default=str
    )

print("\n--- PARAMETER RANDOM FOREST FINAL ---")
print(json.dumps(
    parameter_final,
    indent=4,
    ensure_ascii=False,
    default=str
))

# ------------------------------------------------------------
# 5. METRIK TRAINING VS TESTING
# ------------------------------------------------------------

def hitung_metrik(y_asli, y_prediksi):
    return {
        "accuracy": accuracy_score(y_asli, y_prediksi),
        "precision_weighted": precision_score(
            y_asli,
            y_prediksi,
            average="weighted",
            zero_division=0
        ),
        "recall_weighted": recall_score(
            y_asli,
            y_prediksi,
            average="weighted",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_asli,
            y_prediksi,
            average="weighted",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_asli,
            y_prediksi,
            average="macro",
            zero_division=0
        )
    }

metrik_train = hitung_metrik(
    y_train,
    y_train_pred
)

metrik_test = hitung_metrik(
    y_test,
    y_test_pred
)

perbandingan = pd.DataFrame(
    [metrik_train, metrik_test],
    index=["Training", "Testing"]
)

perbandingan.loc["Gap Train-Test"] = (
    perbandingan.loc["Training"]
    - perbandingan.loc["Testing"]
)

perbandingan.index.name = "Dataset"

perbandingan.to_csv(
    "evaluasi_training_vs_testing.csv",
    encoding="utf-8-sig"
)

print("\n--- PERBANDINGAN TRAINING VS TESTING ---")
print(perbandingan.round(4).to_string())

# ------------------------------------------------------------
# 6. INTERPRETASI AWAL GAP ACCURACY
# ------------------------------------------------------------

gap_accuracy = (
    metrik_train["accuracy"]
    - metrik_test["accuracy"]
)

gap_f1 = (
    metrik_train["f1_weighted"]
    - metrik_test["f1_weighted"]
)

print("\n--- ANALISIS GAP ---")
print(f"Accuracy training : {metrik_train['accuracy']:.4f}")
print(f"Accuracy testing  : {metrik_test['accuracy']:.4f}")
print(f"Gap accuracy      : {gap_accuracy:.4f}")
print(f"Gap F1 weighted   : {gap_f1:.4f}")

if gap_accuracy > 0.05:
    print(
        "Catatan: terdapat gap accuracy lebih dari 5 poin "
        "persentase. Periksa kemungkinan overfitting."
    )
elif gap_accuracy > 0.02:
    print(
        "Catatan: terdapat gap accuracy. Evaluasi lebih lanjut "
        "diperlukan untuk menilai generalisasi model."
    )
else:
    print(
        "Catatan: gap accuracy relatif kecil. "
        "Tetap periksa metrik testing dan distribusi kelas."
    )

print("\nEvaluasi selesai. File yang dibuat:")
print("- evaluasi_classification_report.csv")
print("- evaluasi_confusion_matrix.csv")
print("- evaluasi_confusion_matrix.png")
print("- parameter_random_forest_final.json")
print("- evaluasi_training_vs_testing.csv")
