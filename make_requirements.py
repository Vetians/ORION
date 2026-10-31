"""Jalankan di lingkungan tempat model RF dilatih agar versi library cocok dengan file .pkl."""
import joblib, numpy, pandas, sklearn

lines = [
    f"scikit-learn=={sklearn.__version__}",
    f"numpy=={numpy.__version__}",
    f"pandas=={pandas.__version__}",
    f"joblib=={joblib.__version__}",
    "streamlit",
]
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
