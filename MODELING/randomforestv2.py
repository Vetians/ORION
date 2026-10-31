# ============================================================
# RANDOM FOREST
# Prediksi recommendation_activities
# + HYPERPARAMETER TUNING
# + CROSS VALIDATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier

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
# 1. KONFIGURASI
# ============================================================

INPUT_FILE = "hasil_spk_smarter_saw2.csv"

OUTPUT_FILE = "hasil_random_forest_rekomendasiv2.csv"
MODEL_FILE = "random_forest_rekomendasi.pkl"

RANDOM_STATE = 42

# 80% Training
# 20% Testing
TEST_SIZE = 0.20

# ============================================================
# KONFIGURASI TUNING
# ============================================================

# Jumlah percobaan kombinasi parameter
N_ITER_SEARCH = 5

# Cross Validation
CV_FOLDS = 3

# Jumlah data training yang digunakan khusus
# untuk proses tuning.
#
# Setelah parameter terbaik ditemukan,
# model final tetap dilatih menggunakan
# seluruh data training 80%.
TUNING_SAMPLE_SIZE = 200_000

# ============================================================
# RANDOM FOREST FINAL
# ============================================================

N_JOBS = -1


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
# 3. MENENTUKAN FEATURE (X)
# ============================================================

features = [
    "academic_performance_index",
    "academic_commitment_index",
    "organizational_engagement_index",
    "career_skill_index",
    "communication_profile_index",
    "financial_need_index"
]

target = "recommendation_activities"


# ============================================================
# 4. CEK KOLOM
# ============================================================

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Kolom berikut tidak ditemukan:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 5. MEMBUAT X DAN y
# ============================================================

X = df[features].copy()

y = df[target].copy()


print("\n" + "=" * 70)
print("FEATURE DAN TARGET")
print("=" * 70)

print("\nFeature X:")

for feature in features:
    print("-", feature)

print("\nTarget y:")
print("-", target)

print(f"\nJumlah data X : {len(X):,}")
print(f"Jumlah data y : {len(y):,}")


# ============================================================
# 6. CEK MISSING VALUE
# ============================================================

print("\n" + "=" * 70)
print("CEK MISSING VALUE")
print("=" * 70)

missing_X = X.isnull().sum()
missing_y = y.isnull().sum()

print("\nMissing value pada X:")
print(missing_X)

print("\nMissing value pada y:")
print(missing_y)


# ============================================================
# 7. HAPUS DATA MISSING
# ============================================================

if X.isnull().any().any() or y.isnull().any():

    valid_rows = (
        X.notnull().all(axis=1)
        & y.notnull()
    )

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    print(
        f"\nData setelah menghapus missing value: "
        f"{len(X):,}"
    )


# ============================================================
# 8. DISTRIBUSI TARGET
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUSI TARGET")
print("=" * 70)

target_distribution = y.value_counts()

target_percentage = (
    y.value_counts(normalize=True) * 100
)

distribution_df = pd.DataFrame({
    "Jumlah": target_distribution,
    "Persentase": target_percentage
})

print(distribution_df)


# ============================================================
# 9. TRAIN TEST SPLIT
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

print(f"Data training : {len(X_train):,}")
print(f"Data testing  : {len(X_test):,}")

print(
    f"Training      : "
    f"{len(X_train) / len(X) * 100:.2f}%"
)

print(
    f"Testing       : "
    f"{len(X_test) / len(X) * 100:.2f}%"
)


# ============================================================
# 10. MEMBUAT DATA SAMPLE UNTUK TUNING
# ============================================================

print("\n" + "=" * 70)
print("MENYIAPKAN DATA TUNING")
print("=" * 70)

if len(X_train) > TUNING_SAMPLE_SIZE:

    X_tuning, _, y_tuning, _ = train_test_split(
        X_train,
        y_train,
        train_size=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

else:

    X_tuning = X_train.copy()
    y_tuning = y_train.copy()

print(
    f"Data training untuk tuning : "
    f"{len(X_tuning):,}"
)

print(
    f"Data training penuh         : "
    f"{len(X_train):,}"
)


# ============================================================
# 11. MODEL DASAR UNTUK TUNING
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT MODEL RANDOM FOREST UNTUK TUNING")
print("=" * 70)

rf_base = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)


# ============================================================
# 12. PARAMETER YANG AKAN DITUNING
# ============================================================

param_distributions = {

    # Jumlah pohon
    "n_estimators": [
        100,
        200,
        300
    ],

    # Kedalaman maksimum pohon
    "max_depth": [
        10,
        15,
        20,
        30,
        None
    ],

    # Jumlah minimum sampel untuk melakukan split
    "min_samples_split": [
        2,
        5,
        10
    ],

    # Jumlah minimum sampel pada leaf
    "min_samples_leaf": [
        1,
        2,
        4
    ],

    # Jumlah feature yang dipertimbangkan
    # pada setiap split
    "max_features": [
        "sqrt",
        "log2"
    ]
}


# ============================================================
# 13. RANDOMIZED SEARCH + CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RANDOMIZED SEARCH + 3-FOLD CROSS VALIDATION")
print("=" * 70)

print(
    f"Jumlah kombinasi yang dicoba : "
    f"{N_ITER_SEARCH}"
)

print(
    f"Cross Validation             : "
    f"{CV_FOLDS}-Fold"
)

print(
    f"Scoring                      : "
    f"F1 Weighted"
)

random_search = RandomizedSearchCV(
    estimator=rf_base,

    param_distributions=param_distributions,

    n_iter=N_ITER_SEARCH,

    cv=CV_FOLDS,

    scoring="f1_weighted",

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbose=2,

    return_train_score=True
)


# ============================================================
# 14. PROSES TUNING
# ============================================================

print("\nProses tuning dimulai...")
print("Proses ini dapat membutuhkan waktu cukup lama.")

random_search.fit(
    X_tuning,
    y_tuning
)

print("\nTuning selesai.")


# ============================================================
# 15. HASIL PARAMETER TERBAIK
# ============================================================

print("\n" + "=" * 70)
print("PARAMETER TERBAIK")
print("=" * 70)

print(
    random_search.best_params_
)


# ============================================================
# 16. HASIL CROSS VALIDATION TERBAIK
# ============================================================

print("\n" + "=" * 70)
print("HASIL CROSS VALIDATION")
print("=" * 70)

print(
    f"Best CV F1-Score : "
    f"{random_search.best_score_:.4f}"
)


# ============================================================
# 17. RINGKASAN HASIL TUNING
# ============================================================

cv_results = pd.DataFrame(
    random_search.cv_results_
)

tuning_results = cv_results[
    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "params"
    ]
].sort_values(
    by="rank_test_score"
)

print("\nHasil percobaan parameter:")

print(
    tuning_results.to_string(
        index=False
    )
)


# ============================================================
# 18. MEMBUAT MODEL FINAL
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT MODEL FINAL")
print("=" * 70)

best_params = random_search.best_params_

print("\nParameter model final:")

for parameter, value in best_params.items():
    print(
        f"{parameter} : {value}"
    )


model = RandomForestClassifier(
    **best_params,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)


# ============================================================
# 19. TRAINING MODEL FINAL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL FINAL")
print("=" * 70)

print(
    "Model final dilatih menggunakan "
    "seluruh data training 80%."
)

model.fit(
    X_train,
    y_train
)

print("Training model final selesai.")

# ============================================================
# 21. SIMPAN MODEL FINAL KE FORMAT PKL
# ============================================================

print("\n" + "=" * 70)
print("MENYIMPAN MODEL FINAL KE FORMAT PKL")
print("=" * 70)

# Menyimpan model beserta informasi yang diperlukan
# untuk digunakan kembali tanpa melakukan training ulang.
model_package = {
    "model": model,
    "features": features,
    "target": target,
    "best_params": best_params,
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print(f"Model berhasil disimpan: {MODEL_FILE}")
print("Isi file PKL:")
print("- Model Random Forest final")
print("- Daftar feature")
print("- Target")
print("- Best hyperparameters")
print("- Random state")



# ============================================================
# 21. PREDIKSI DATA TESTING
# ============================================================

print("\n" + "=" * 70)
print("PREDIKSI DATA TESTING")
print("=" * 70)

y_pred = model.predict(
    X_test
)

print("Prediksi testing selesai.")


# ============================================================
# 22. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 23. PRECISION
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 24. RECALL
# ============================================================

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 25. F1-SCORE
# ============================================================

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 26. HASIL EVALUASI
# ============================================================

print("\n" + "=" * 70)
print("HASIL EVALUASI RANDOM FOREST")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nClass:")
print(model.classes_)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 29. VISUALISASI CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.tight_layout()

plt.show()


# ============================================================
# 30. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    feature_importance
)


# ============================================================
# 31. VISUALISASI FEATURE IMPORTANCE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Feature Importance - Random Forest"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 32. PREDIKSI SELURUH DATA
# ============================================================

print("\n" + "=" * 70)
print("PREDIKSI SELURUH DATA")
print("=" * 70)

df["rf_prediction"] = model.predict(
    df[features]
)

print(
    "Prediksi seluruh dataset selesai."
)


# ============================================================
# 33. CEK HASIL PREDIKSI
# ============================================================

print("\nContoh hasil:")

result_columns = [
    "recommendation_activities",
    "rf_prediction"
]

print(
    df[result_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 34. CEK KESESUAIAN PREDIKSI
# ============================================================

df["rf_correct"] = (
    df["recommendation_activities"]
    ==
    df["rf_prediction"]
)

prediction_accuracy = (
    df["rf_correct"].mean()
)


print("\n" + "=" * 70)
print("AKURASI PREDIKSI SELURUH DATA")
print("=" * 70)

print(
    f"Akurasi terhadap label SMARTER + SAW: "
    f"{prediction_accuracy:.4f}"
)


# ============================================================
# 35. DISTRIBUSI HASIL PREDIKSI RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUSI PREDIKSI RANDOM FOREST")
print("=" * 70)

rf_distribution = (
    df["rf_prediction"]
    .value_counts()
)

rf_percentage = (
    df["rf_prediction"]
    .value_counts(
        normalize=True
    )
    * 100
)

rf_distribution_df = pd.DataFrame({

    "Jumlah":
        rf_distribution,

    "Persentase":
        rf_percentage

})

print(
    rf_distribution_df
)


# ============================================================
# 36. SIMPAN HASIL MODEL
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 37. INFORMASI FILE
# ============================================================

print("\n" + "=" * 70)
print("FILE HASIL")
print("=" * 70)

print(
    f"File hasil prediksi : {OUTPUT_FILE}\n"
    f"Model Random Forest : {MODEL_FILE}"
)

print(
    f"\nJumlah data akhir: "
    f"{len(df):,} baris"
)

print("\nSELESAI.")
