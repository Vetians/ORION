import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import spk_core as core

st.set_page_config(page_title="Rekomendasi Kegiatan Mahasiswa", page_icon="🎓")


@st.cache_resource
def load_assets():
    pkg = joblib.load(core.MODEL_FILE)
    with open(core.PARAMS_FILE, encoding="utf-8") as f:
        params = json.load(f)
    return pkg, params


pkg, params = load_assets()
model, features = pkg["model"], pkg["features"]

st.title("🎓 Rekomendasi Kegiatan Mahasiswa")
st.caption(
    "Sistem pendukung keputusan SMARTER + SAW yang ditiru oleh model Random Forest. "
    "Ini alat bantu berbasis tugas akademik, bukan keputusan final."
)

# Di luar form agar kolom SAT/ACT bisa langsung dinonaktifkan
skip_sat_act = st.checkbox(
    "Aku tidak punya skor SAT/ACT",
    value=True,
    help="Indeks performa akademik akan dihitung dari komponen yang tersedia.",
)
if skip_sat_act:
    st.caption(
        "Indeks performa akademik dihitung dari IPK dan kehadiran saja. Ini sedikit "
        "berbeda dari data latih yang selalu memuat SAT dan ACT."
    )
skip = core.SAT_ACT if skip_sat_act else ()


def default_value(col):
    median = params["income"]["median"] if col == core.INCOME_COLUMN else params["ranges"][col]["median"]
    lo, hi, is_int = core.FIELDS[col][2], core.FIELDS[col][3], core.FIELDS[col][5]
    median = min(max(median, lo), hi)
    return int(round(median)) if is_int else round(float(median), 2)


with st.form("data_mahasiswa"):
    st.markdown("Isi data mentah kamu. Nilai awal adalah median dataset.")
    values = {}
    for group, cols in core.GROUPS.items():
        st.subheader(group)
        for col in cols:
            label, help_text, lo, hi, step, is_int = core.FIELDS[col]
            cast = int if is_int else float
            values[col] = st.number_input(
                label,
                min_value=cast(lo),
                max_value=cast(hi),
                value=cast(default_value(col)),
                step=cast(step),
                help=help_text,
                disabled=(col in skip),
                key=f"in_{col}",
            )
    submitted = st.form_submit_button("Dapatkan rekomendasi")

st.caption(
    "Catatan: SAT/ACT, AP, dan pendapatan memakai satuan dataset latih (dataset mahasiswa "
    "Amerika). Jika pendapatan keluarga kamu dalam Rupiah, konversi dulu ke satuan dataset."
)

if submitted:
    flagged = core.out_of_range(values, params, skip=skip)
    if flagged:
        st.warning(
            "Nilai berikut di luar rentang dataset latih dan dihitung sebagai batas rentang: "
            + ", ".join(flagged)
        )

    # 1. Data mentah -> enam indeks (rumus sama dengan pipeline data preparation)
    raw_df = pd.DataFrame([values])
    idx = core.compute_indices(raw_df, params, skip=skip).iloc[0]

    # 2. Indeks -> posisi relatif (persentil) terhadap dataset, dipakai SAW
    pct = {
        k: core.to_percentile(idx[core.CRITERIA[k][0]], params["quantiles"][core.CRITERIA[k][0]])
        for k in core.KEYS
    }

    # 3. Random Forest memakai indeks mentah (sama dengan saat training)
    x = pd.DataFrame([[idx[f] for f in features]], columns=features)
    proba = pd.Series(model.predict_proba(x)[0], index=model.classes_)
    rf_rank = proba.sort_values(ascending=False)

    # 4. SAW langsung untuk penjelasan
    s = np.array([pct[k] for k in core.KEYS])
    compat = 1 - np.abs(s[None, :] - core.REQUIREMENT)
    saw_rank = pd.Series(
        (compat * core.WEIGHTS).sum(axis=1), index=core.ALTERNATIVES
    ).sort_values(ascending=False)

    st.subheader("Rekomendasi teratas (Random Forest)")
    cols = st.columns(3)
    for col, (name, prob) in zip(cols, rf_rank.head(3).items()):
        col.metric(name, f"{prob * 100:.1f}%")
    st.caption("Persentase = proporsi pohon yang memilih kegiatan tersebut.")

    if rf_rank.index[0] == saw_rank.index[0]:
        st.success("Hasil Random Forest sama dengan perhitungan SAW langsung.")
    else:
        st.info(
            f"Random Forest memilih **{rf_rank.index[0]}**, sedangkan SAW langsung memilih "
            f"**{saw_rank.index[0]}**. Biasanya ini terjadi saat skor keduanya sangat berdekatan."
        )

    with st.expander("Profil kamu setelah diolah (indeks dan posisi relatif)", expanded=True):
        profile = pd.DataFrame({
            "Kriteria": [core.CRITERIA[k][1] for k in core.KEYS],
            "Indeks (0-1)": [float(idx[core.CRITERIA[k][0]]) for k in core.KEYS],
            "Posisi relatif (%)": [pct[k] * 100 for k in core.KEYS],
        }).round(3)
        st.dataframe(profile, hide_index=True)
        st.caption("Posisi relatif = kamu berada di atas sekian persen mahasiswa di dataset.")

    with st.expander("Rincian skor SAW (semua kegiatan)"):
        st.dataframe(saw_rank.rename("Skor SAW").round(4).to_frame())

    top = rf_rank.index[0]
    with st.expander(f"Kecocokan per kriteria: {top}"):
        if top in core.ALTERNATIVES:
            i = core.ALTERNATIVES.index(top)
            detail = pd.DataFrame({
                "Kriteria": [core.CRITERIA[k][1] for k in core.KEYS],
                "Posisi kamu": s,
                "Kebutuhan kegiatan": core.REQUIREMENT[i],
                "Kecocokan": compat[i],
                "Bobot": core.WEIGHTS,
            }).round(3)
            st.dataframe(detail, hide_index=True)
