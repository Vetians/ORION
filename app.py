import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import spk_core as core

st.set_page_config(
    page_title="Rekomendasi Kegiatan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1100px 540px at 8% -8%, #eef2ff 0%, transparent 55%),
            radial-gradient(850px 460px at 100% 0%, #fdf2f8 0%, transparent 52%),
            #f8fafc;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3, h4 { color: #0f172a; letter-spacing: -0.025em; }
    p, li, label { color: #334155; }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #111827 0%, #312e81 56%, #4f46e5 100%);
        border-radius: 28px;
        padding: 38px 42px;
        color: white;
        box-shadow: 0 24px 60px rgba(49, 46, 129, 0.24);
        margin-bottom: 22px;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        right: -100px;
        top: -130px;
    }
    .hero .eyebrow {
        color: #c7d2fe;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: white;
        font-size: 2.3rem;
        line-height: 1.12;
        margin: 10px 0 12px;
    }
    .hero p {
        color: #e0e7ff;
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #eef2ff;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 4px;
    }
    .section-title .num {
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: #eef2ff;
        color: #4338ca;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 13px;
    }
    .section-title h3 { margin: 0; font-size: 1.05rem; }
    .section-sub {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0 0 10px 38px;
    }

    div[data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        padding: 0.35rem 0.45rem;
    }
    div[data-baseweb="input"] { border-radius: 12px; }
    div[data-testid="stAlert"] { border-radius: 14px; }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.2rem !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.28);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(79, 70, 229, 0.34);
    }

    .rec-card {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid #bbf7d0;
        border-radius: 24px;
        padding: 26px 28px;
        margin: 10px 0 16px;
    }
    .rec-eyebrow {
        color: #047857;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .rec-title {
        color: #064e3b;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 6px 0 4px;
    }
    .rec-desc { color: #065f46; font-size: 0.98rem; line-height: 1.55; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .chip {
        background: #ffffff;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.03);
    }
    .metric-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 6px;
        line-height: 1.3;
    }
    .metric-note { color: #64748b; font-size: 0.82rem; margin-top: 4px; line-height: 1.4; }

    .insight-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px 22px;
    }
    .insight-card ul { margin: 8px 0 0 18px; padding: 0; }
    .insight-card li { margin-bottom: 8px; line-height: 1.55; color: #334155; }
    .insight-title { font-weight: 800; color: #0f172a; margin-bottom: 2px; }

    .criterion { margin: 13px 0; }
    .criterion-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 6px;
    }
    .criterion-name { font-weight: 700; color: #334155; font-size: 0.9rem; }
    .tag {
        font-size: 11px;
        font-weight: 800;
        padding: 4px 9px;
        border-radius: 999px;
        white-space: nowrap;
    }
    .tag-strong { background: #dcfce7; color: #166534; }
    .tag-medium { background: #fef3c7; color: #92400e; }
    .tag-low { background: #fee2e2; color: #991b1b; }
    .bar {
        height: 9px;
        background: #eef2f7;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 999px; }
    .bar-green { background: linear-gradient(90deg, #10b981, #34d399); }
    .bar-amber { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .bar-rose { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .bar-indigo { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

    .alt-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 13px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .alt-row:last-child { border-bottom: none; }
    .alt-name { font-weight: 700; color: #334155; font-size: 0.92rem; }
    .alt-note { color: #64748b; font-size: 0.8rem; margin-top: 2px; }
    .alt-bar { width: 38%; min-width: 120px; }

    .footer-note {
        color: #94a3b8;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 32px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    pkg = joblib.load(core.MODEL_FILE)
    with open(core.PARAMS_FILE, encoding="utf-8") as f:
        params = json.load(f)
    return pkg, params


pkg, params = load_assets()
model, features = pkg["model"], pkg["features"]

rel_min, rel_max, rel_median, ref_usd = core.get_relative_income_bounds(params)
default_ref_idr = int(
    round(float(params["income"].get("reference_idr_monthly", core.REFERENCE_INCOME_IDR_MONTHLY)))
)


# ---------------------------------------------------------------------------
# Helper tampilan
# ---------------------------------------------------------------------------
def format_juta(value: float | int) -> str:
    """Format Rupiah singkat, mis. 7.151.200 -> 'Rp 7,2 juta'."""
    value = float(value)
    if abs(value) < 1_000_000:
        return core.format_rupiah(value)
    juta = value / 1_000_000
    txt = f"{juta:.1f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"Rp {txt} juta"


def confidence_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Sangat yakin", "Model cukup bulat memilih kegiatan ini."
    if prob >= 0.50:
        return "Cukup yakin", "Ada beberapa pilihan yang berdekatan, tetapi ini yang paling kuat."
    if prob >= 0.35:
        return "Masih berimbang", "Beberapa kegiatan punya peluang yang mirip. Pertimbangkan juga pilihan lain."
    return "Belum ada yang dominan", "Profilmu cocok dengan beberapa kegiatan. Lihat daftar alternatif di bawah."


def strength_label(score: float) -> tuple[str, str, str]:
    if score >= 0.75:
        return "Kuat", "tag-strong", "bar-green"
    if score >= 0.50:
        return "Cukup", "tag-medium", "bar-amber"
    return "Perlu ditingkatkan", "tag-low", "bar-rose"


def need_label(score: float) -> tuple[str, str, str]:
    if score >= 0.70:
        return "Tinggi", "tag-low", "bar-rose"
    if score >= 0.40:
        return "Sedang", "tag-medium", "bar-amber"
    return "Rendah", "tag-strong", "bar-green"


def income_position(rel_raw: float) -> tuple[str, str]:
    if rel_raw < 0.50:
        return "Jauh di bawah acuan", "Kebutuhan finansial kemungkinan besar menjadi prioritas."
    if rel_raw < 0.85:
        return "Di bawah acuan", "Kebutuhan finansial cenderung lebih tinggi."
    if rel_raw <= 1.15:
        return "Sekitar acuan", "Kebutuhan finansial tergolong moderat."
    if rel_raw <= 1.50:
        return "Di atas acuan", "Kamu punya lebih banyak ruang untuk memilih kegiatan pengembangan diri."
    return "Jauh di atas acuan", "Kebutuhan finansial relatif rendah."


FRIENDLY_CRITERIA = {
    "K1": "performa akademik",
    "K2": "komitmen belajar",
    "K3": "keterlibatan organisasi",
    "K4": "kemampuan komunikasi",
    "K5": "skill dan prestasi",
    "K6": "kebutuhan finansial",
}


def friendly_name(key: str) -> str:
    """Label kriteria dengan huruf kapital di awal saja."""
    return FRIENDLY_CRITERIA[key][:1].upper() + FRIENDLY_CRITERIA[key][1:]


def section_title(number: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title"><span class="num">{number}</span><h3>{title}</h3></div>
        <div class="section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SPK Mahasiswa · SMARTER + SAW</div>
        <h1>Pilih kegiatan yang paling cocok untukmu, bukan yang paling ramai.</h1>
        <p>
            Isi profil singkat. Sistem akan mencocokkan kemampuan, pengalaman, dan kebutuhan
            finansialmu dengan tujuh pilihan kegiatan. Hasilnya berupa rekomendasi yang bisa
            kamu pertimbangkan, bukan keputusan final.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ 2–3 menit</span>
            <span class="hero-badge">🎯 7 alternatif kegiatan</span>
            <span class="hero-badge">🔍 Alasan rekomendasi transparan</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Pengaturan di sidebar
# ---------------------------------------------------------------------------
def _format_ref_Callback():
    """Format ulang input acuan Rupiah dengan pemisah titik setiap 3 digit."""
    raw_txt = st.session_state.get("in_ref_idr", "")
    try:
        val = core.parse_ribuan(raw_txt)
        val = min(max(val, 1_000_000), 50_000_000)
        st.session_state["in_ref_idr"] = core.format_ribuan(val)
    except ValueError:
        st.session_state["in_ref_idr"] = core.format_ribuan(default_ref_idr)


if "in_ref_idr" not in st.session_state:
    st.session_state["in_ref_idr"] = core.format_ribuan(default_ref_idr)

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.caption("Opsional. Kamu bisa langsung mengisi profil tanpa mengubah apa pun.")

    skip_sat_act = st.checkbox(
        "Saya tidak punya skor SAT/ACT",
        value=True,
        help="Kalau dicentang, performa akademik dinilai dari IPK dan kehadiran saja.",
    )
    if skip_sat_act:
        st.info("Performa akademik akan dinilai dari **IPK dan kehadiran**.")
    else:
        st.caption("Kolom SAT dan ACT akan muncul di bagian profil akademik.")

    st.divider()
    st.markdown("### 💰 Acuan pendapatan")
    st.caption(
        "Dipakai untuk menilai kebutuhan finansial secara relatif, bukan sebagai nilai absolut."
    )
    ref_idr_str = st.text_input(
        "Pendapatan keluarga acuan per bulan (Rp)",
        key="in_ref_idr",
        on_change=_format_ref_Callback,
        help="Tekan Enter setelah mengisi agar format titik ribuan dirapikan.",
        placeholder="Contoh: 10.000.000",
    )
    try:
        reference_idr_monthly = float(
            min(max(core.parse_ribuan(ref_idr_str), 1_000_000), 50_000_000)
        )
    except ValueError:
        reference_idr_monthly = float(default_ref_idr)

    st.caption(f"Acuan saat ini: **{format_juta(reference_idr_monthly)}/bulan**.")

    with st.expander("Lihat detail acuan"):
        st.write(
            f"Rentang pendapatan yang paling informatif: "
            f"**{format_juta(rel_min * reference_idr_monthly)} – "
            f"{format_juta(rel_max * reference_idr_monthly)}/bulan**."
        )
        st.write(
            f"Nilai tengah dataset setara **{format_juta(rel_median * reference_idr_monthly)}/bulan**."
        )
        st.caption(
            "Angka ini hanya untuk transparansi. Kamu tidak perlu menghitung manual."
        )

    st.divider()
    st.markdown("### ℹ️ Tentang alat ini")
    st.caption(
        "Alat bantu akademik untuk membandingkan tujuh pilihan kegiatan. "
        "Hasilnya bukan keputusan mutlak dan tetap perlu kamu pertimbangkan."
    )

skip = core.SAT_ACT if skip_sat_act else ()


# ---------------------------------------------------------------------------
# Form profil
# ---------------------------------------------------------------------------
def default_value(col):
    if col == core.INCOME_COLUMN:
        median_idr = round((rel_median * reference_idr_monthly) / 100_000) * 100_000
        lo, hi = core.FIELDS[col][2], core.FIELDS[col][3]
        return int(min(max(median_idr, lo), hi))
    median = params["ranges"][col]["median"]
    lo, hi, is_int = core.FIELDS[col][2], core.FIELDS[col][3], core.FIELDS[col][5]
    median = min(max(median, lo), hi)
    return int(round(median)) if is_int else round(float(median), 2)


with st.form("profile_form"):
    st.markdown("### Isi profilmu")
    st.caption(
        "Nilai awal sudah diisi dengan nilai tengah dataset. Ubah hanya bagian yang kamu tahu; "
        "sisanya boleh dibiarkan."
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        section_title("1", "Profil akademik", "Dua angka ini sudah cukup untuk memulai.")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                gpa = st.slider(
                    "IPK / GPA (skala 4,0)",
                    min_value=0.0,
                    max_value=4.0,
                    value=float(default_value("high_school_gpa")),
                    step=0.01,
                    key="ui_gpa",
                    help="Rata-rata nilai rapor/IPK dalam skala 4,0.",
                )
            with c2:
                attendance = st.slider(
                    "Kehadiran (%)",
                    min_value=0,
                    max_value=100,
                    value=int(round(default_value("attendance_rate"))),
                    step=1,
                    key="ui_attendance",
                    help="Persentase kehadiran di sekolah atau kampus.",
                )

            if not skip_sat_act:
                c3, c4 = st.columns(2)
                with c3:
                    sat = st.number_input(
                        "Skor SAT",
                        min_value=400,
                        max_value=1600,
                        value=int(default_value("sat_score")),
                        step=10,
                        key="ui_sat",
                        help="Isi jika kamu punya skor SAT.",
                    )
                with c4:
                    act = st.number_input(
                        "Skor ACT",
                        min_value=1,
                        max_value=36,
                        value=int(default_value("act_score")),
                        step=1,
                        key="ui_act",
                        help="Isi jika kamu punya skor ACT.",
                    )

            ap_courses = st.number_input(
                "Mata pelajaran lanjutan / AP",
                min_value=0,
                max_value=50,
                value=int(default_value("ap_courses")),
                step=1,
                key="ui_ap",
                help="Jumlah mata pelajaran tingkat lanjut. Isi 0 jika tidak ada.",
            )

        section_title(
            "2",
            "Pengalaman & keterampilan",
            "Perkiraan saja. Tidak perlu tepat.",
        )
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                extracurricular = st.number_input(
                    "Organisasi / ekstrakurikuler",
                    min_value=0,
                    max_value=50,
                    value=int(default_value("extracurricular_count")),
                    step=1,
                    key="ui_extracurricular",
                    help="Jumlah organisasi atau ekstrakurikuler yang diikuti.",
                )
                volunteer = st.number_input(
                    "Total jam volunteer",
                    min_value=0,
                    max_value=5000,
                    value=int(default_value("volunteer_hours")),
                    step=5,
                    key="ui_volunteer",
                    help="Total jam kegiatan sukarela.",
                )
                leadership = st.number_input(
                    "Posisi kepemimpinan",
                    min_value=0,
                    max_value=50,
                    value=int(default_value("leadership_positions")),
                    step=1,
                    key="ui_leadership",
                    help="Jumlah posisi/peran kepemimpinan yang pernah dipegang.",
                )
            with c2:
                coding = st.number_input(
                    "Proyek pemrograman",
                    min_value=0,
                    max_value=50,
                    value=int(default_value("coding_projects")),
                    step=1,
                    key="ui_coding",
                    help="Jumlah proyek pemrograman yang pernah dikerjakan.",
                )
                certifications = st.number_input(
                    "Sertifikasi online",
                    min_value=0,
                    max_value=50,
                    value=int(default_value("online_certifications")),
                    step=1,
                    key="ui_certifications",
                    help="Jumlah sertifikasi online yang dimiliki.",
                )

    with right:
        section_title("3", "Komunikasi & profil", "Nilai perkiraan dalam skala 0–100.")
        with st.container(border=True):
            essay = st.slider(
                "Nilai esai / portofolio",
                min_value=0,
                max_value=100,
                value=int(round(default_value("essay_score"))),
                step=1,
                key="ui_essay",
                help="Perkiraan nilai esai atau portofolio.",
            )
            recommendation = st.slider(
                "Nilai surat rekomendasi",
                min_value=0,
                max_value=100,
                value=int(round(default_value("recommendation_score"))),
                step=1,
                key="ui_recommendation",
                help="Perkiraan nilai surat rekomendasi.",
            )
            interview = st.slider(
                "Nilai wawancara",
                min_value=0,
                max_value=100,
                value=int(round(default_value("interview_score"))),
                step=1,
                key="ui_interview",
                help="Perkiraan nilai wawancara.",
            )

        section_title("4", "Kondisi finansial", "Cukup masukkan perkiraan pendapatan keluarga.")
        with st.container(border=True):
            raw_income_text = st.text_input(
                "Pendapatan keluarga per bulan (Rp)",
                value=core.format_ribuan(default_value(core.INCOME_COLUMN)),
                key="ui_income",
                help=(
                    "Total pendapatan orang tua/keluarga per bulan. "
                    "Boleh memakai titik, misalnya 7.500.000."
                ),
                placeholder="Contoh: 7.500.000",
            )
            st.caption(
                "Perkiraan saja. Kamu bisa menulis **7500000** atau **7.500.000**. "
                f"Nilai tengah dataset saat ini sekitar **{format_juta(rel_median * reference_idr_monthly)}/bulan**."
            )

        submitted = st.form_submit_button(
            "✨ Lihat rekomendasiku",
            type="primary",
            width="stretch",
        )
        st.caption("Hasilnya berupa saran. Kamu tetap yang mengambil keputusan.")


# ---------------------------------------------------------------------------
# Hasil
# ---------------------------------------------------------------------------
if submitted:
    st.session_state["show_results"] = True

if st.session_state.get("show_results"):
    if not submitted:
        st.caption(
            "ℹ️ Menampilkan hasil dari data terakhir yang kamu kirim. "
            "Tekan tombol lagi setelah mengubah isian."
        )
    inc_lo, inc_hi = core.FIELDS[core.INCOME_COLUMN][2], core.FIELDS[core.INCOME_COLUMN][3]
    try:
        parsed_income = core.parse_ribuan(raw_income_text)
    except ValueError:
        st.error("Pendapatan keluarga harus berupa angka yang valid, misalnya 8.500.000.")
        st.stop()

    if parsed_income < inc_lo or parsed_income > inc_hi:
        st.info(
            "Pendapatan keluarga disesuaikan ke rentang form agar perhitungan tetap stabil."
        )
        parsed_income = min(max(parsed_income, inc_lo), inc_hi)

    values = {
        "high_school_gpa": gpa,
        "attendance_rate": float(attendance),
        "ap_courses": int(ap_courses),
        "extracurricular_count": int(extracurricular),
        "volunteer_hours": int(volunteer),
        "leadership_positions": int(leadership),
        "coding_projects": int(coding),
        "online_certifications": int(certifications),
        "essay_score": float(essay),
        "recommendation_score": float(recommendation),
        "interview_score": float(interview),
        core.INCOME_COLUMN: int(parsed_income),
    }
    if not skip_sat_act:
        values["sat_score"] = int(sat)
        values["act_score"] = int(act)

    flagged = core.out_of_range(
        values,
        params,
        skip=skip,
        income_unit="idr_monthly",
        reference_idr_monthly=reference_idr_monthly,
    )
    if flagged:
        st.caption(
            "ℹ️ Beberapa nilai berada di luar rentang data latih. "
            "Sistem menyesuaikannya agar rekomendasi tetap stabil."
        )

    # 1. Data mentah -> enam indeks
    raw_df = pd.DataFrame([values])
    idx = core.compute_indices(
        raw_df,
        params,
        skip=skip,
        income_unit="idr_monthly",
        reference_idr_monthly=reference_idr_monthly,
    ).iloc[0]

    # Detail relative_income untuk bagian transparansi
    rel_raw = float(values[core.INCOME_COLUMN]) / float(reference_idr_monthly)
    rel_clipped = float(
        core.compute_relative_income(
            raw_df[core.INCOME_COLUMN],
            params,
            income_unit="idr_monthly",
            reference_idr_monthly=reference_idr_monthly,
        ).iloc[0]
    )
    usd_equiv = rel_clipped * ref_usd

    # 2. Indeks -> posisi relatif (persentil) untuk SAW
    pct = {
        k: core.to_percentile(idx[core.CRITERIA[k][0]], params["quantiles"][core.CRITERIA[k][0]])
        for k in core.KEYS
    }

    # 3. Random Forest memakai indeks mentah
    x = pd.DataFrame([[idx[f] for f in features]], columns=features)
    proba = pd.Series(model.predict_proba(x)[0], index=model.classes_)
    rf_rank = proba.sort_values(ascending=False)

    # 4. SAW langsung untuk pembanding
    s = np.array([pct[k] for k in core.KEYS])
    compat = 1 - np.abs(s[None, :] - core.REQUIREMENT)
    saw_rank = pd.Series(
        (compat * core.WEIGHTS).sum(axis=1), index=core.ALTERNATIVES
    ).sort_values(ascending=False)

    top = rf_rank.index[0]
    top_prob = float(rf_rank.iloc[0])
    second = rf_rank.index[1] if len(rf_rank) > 1 else "-"
    conf_label, conf_note = confidence_label(top_prob)
    need_score = float(idx["financial_need_index"])
    need_text, need_tag, need_fill = need_label(need_score)
    pos_text, pos_note = income_position(rel_raw)

    st.markdown("---")
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-eyebrow">Rekomendasi utama</div>
            <div class="rec-title">{top}</div>
            <div class="rec-desc">{conf_note}</div>
            <div class="chip-row">
                <span class="chip">✓ {conf_label}</span>
                <span class="chip">💰 Kebutuhan finansial: {need_text.lower()}</span>
                <span class="chip">🎯 Berdasarkan 6 kriteria</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tingkat keyakinan</div>
                <div class="metric-value">{conf_label}</div>
                <div class="metric-note">Seberapa bulat model memilih kegiatan ini.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Kebutuhan finansial</div>
                <div class="metric-value">{need_text}</div>
                <div class="metric-note">Dibandingkan dengan pendapatan acuan.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Alternatif berikutnya</div>
                <div class="metric-value">{second}</div>
                <div class="metric-note">Pilihan cadangan yang bisa dipertimbangkan.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Alasan rekomendasi
    st.markdown("#### Kenapa ini cocok?")
    if top in core.ALTERNATIVES:
        alt_i = core.ALTERNATIVES.index(top)
        fit = compat[alt_i]
        best = np.argsort(fit)[::-1]
        reasons = []
        for j in best[:2]:
            k = core.KEYS[j]
            if k == "K6":
                reasons.append(
                    "Kebutuhan finansialmu sejalan dengan tuntutan kegiatan ini."
                )
            else:
                reasons.append(
                    f"Kamu punya modal di {FRIENDLY_CRITERIA[k]}, dan kegiatan ini membutuhkannya."
                )
        weak = best[-1]
        if fit[weak] < 0.55:
            k = core.KEYS[weak]
            if k == "K6":
                reasons.append(
                    "Kebutuhan finansial bukan fokus utama kegiatan ini; pertimbangkan apakah kamu tetap nyaman."
                )
            else:
                reasons.append(
                    f"Yang perlu dipertimbangkan: {FRIENDLY_CRITERIA[k]} masih bisa ditingkatkan."
                )
        else:
            reasons.append("Kombinasi profilmu cukup seimbang untuk kegiatan ini.")
        items = "".join(f"<li>{r}</li>" for r in reasons)
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Tiga hal yang membuat kegiatan ini masuk akal</div>
                <ul>{items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Profil kriteria
    st.markdown("#### Profil singkatmu")
    st.caption(
        "Bar menunjukkan seberapa kuat modalmu di setiap kriteria. "
        "Tidak perlu membaca angka untuk memahami hasilnya."
    )
    pcol1, pcol2 = st.columns(2)
    for i, k in enumerate(core.KEYS):
        score = float(idx[core.CRITERIA[k][0]])
        if k == "K6":
            tag, tag_class, fill_class = need_label(score)
        else:
            tag, tag_class, fill_class = strength_label(score)
        with (pcol1 if i % 2 == 0 else pcol2):
            st.markdown(
                f"""
                <div class="criterion">
                    <div class="criterion-head">
                        <span class="criterion-name">{friendly_name(k)}</span>
                        <span class="tag {tag_class}">{tag}</span>
                    </div>
                    <div class="bar"><div class="bar-fill {fill_class}" style="width:{score * 100:.0f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Kondisi finansial
    st.markdown("#### Kondisi finansial")
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="metric-label">Pendapatan keluargamu</div>
            <div class="metric-value">{format_juta(values[core.INCOME_COLUMN])}/bulan</div>
            <div class="metric-note">
                {pos_text}. Acuan yang dipakai: {format_juta(reference_idr_monthly)}/bulan.
            </div>
            <div class="criterion">
                <div class="criterion-head">
                    <span class="criterion-name">Kebutuhan finansial</span>
                    <span class="tag {need_tag}">{need_text}</span>
                </div>
                <div class="bar"><div class="bar-fill {need_fill}" style="width:{need_score * 100:.0f}%"></div></div>
            </div>
            <div class="metric-note">{pos_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat perhitungan pendapatan"):
        st.write(
            f"Pendapatan keluarga **{core.format_rupiah(values[core.INCOME_COLUMN])}/bulan** "
            f"dibagi acuan **{core.format_rupiah(reference_idr_monthly)}/bulan** menghasilkan "
            f"`relative_income` **{rel_raw:.3f}×**."
        )
        st.write(
            f"Setelah penyesuaian batas data (winsorizing), nilainya menjadi "
            f"**{rel_clipped:.3f}×**, setara **${core.format_ribuan(usd_equiv)}/tahun** "
            f"pada skala dataset AS. Nilai K6 (kebutuhan finansial) adalah "
            f"**{need_score:.3f}** pada skala 0–1."
        )

    # Pilihan lain
    st.markdown("#### Pilihan lain yang bisa dipertimbangkan")
    st.caption(
        "Panjang bar menunjukkan seberapa sering model memilih kegiatan tersebut, "
        "bukan nilai mutlak."
    )
    alt_rows = []
    for rank, (name, prob) in enumerate(rf_rank.head(3).items()):
        if rank == 0:
            note = "Paling cocok"
        elif rank == 1:
            note = "Alternatif kuat"
        else:
            note = "Bisa dipertimbangkan"
        alt_rows.append(
            f'<div class="alt-row">'
            f'<div><div class="alt-name">{name}</div>'
            f'<div class="alt-note">{note}</div></div>'
            f'<div class="alt-bar"><div class="bar">'
            f'<div class="bar-fill bar-indigo" style="width:{prob * 100:.0f}%"></div>'
            f"</div></div></div>"
        )
    st.markdown(
        f'<div class="insight-card">{"".join(alt_rows)}</div>',
        unsafe_allow_html=True,
    )

    # Perbandingan RF vs SAW
    saw_top = saw_rank.index[0]
    if top == saw_top:
        st.success(
            "Model Random Forest dan perhitungan SAW memberi rekomendasi yang sama. "
            "Hasil ini relatif konsisten."
        )
    else:
        st.info(
            f"Model memilih **{top}**, sedangkan perhitungan SAW memilih **{saw_top}**. "
            "Keduanya berdekatan, jadi anggap sebagai dua kandidat yang sama-sama kuat."
        )

    # Detail angka, disembunyikan agar tampilan utama tetap sederhana
    with st.expander("Lihat detail angka dan perhitungan"):
        st.markdown("**Profil kriteria (skala 0–1 dan posisi relatif)**")
        profile = pd.DataFrame(
            {
                "Kriteria": [core.CRITERIA[k][1] for k in core.KEYS],
                "Indeks (0–1)": [float(idx[core.CRITERIA[k][0]]) for k in core.KEYS],
                "Posisi relatif (%)": [pct[k] * 100 for k in core.KEYS],
            }
        ).round(3)
        st.dataframe(profile, hide_index=True, width="stretch")

        st.markdown("**Skor semua alternatif (SAW)**")
        st.dataframe(
            saw_rank.rename("Skor SAW").round(4).to_frame(),
            width="stretch",
        )

        if top in core.ALTERNATIVES:
            i = core.ALTERNATIVES.index(top)
            detail = pd.DataFrame(
                {
                    "Kriteria": [core.CRITERIA[k][1] for k in core.KEYS],
                    "Posisi kamu": s,
                    "Kebutuhan kegiatan": core.REQUIREMENT[i],
                    "Kecocokan": compat[i],
                    "Bobot": core.WEIGHTS,
                }
            ).round(3)
            st.markdown(f"**Rincian kecocokan: {top}**")
            st.dataframe(detail, hide_index=True, width="stretch")

        st.markdown("**Probabilitas Random Forest**")
        st.dataframe(
            (rf_rank * 100).round(1).rename("Peluang (%)").to_frame(),
            width="stretch",
        )

    with st.expander("Bagaimana sistem ini bekerja?"):
        st.markdown(
            """
            1. **Profil** — jawabanmu diubah menjadi enam indikator dengan skala 0–1.
            2. **Pencocokan** — setiap kegiatan punya kebutuhan ideal. Sistem mengukur seberapa
               dekat profilmu dengan kebutuhan itu.
            3. **Peringkat** — Random Forest mempelajari pola dari data dan memberi urutan
               kegiatan. Perhitungan SAW dipakai sebagai pembanding agar hasilnya bisa dijelaskan.
            """
        )
        st.caption(
            "Karena data latih berasal dari konteks Amerika Serikat, pendapatan disesuaikan "
            "menjadi rasio terhadap acuan. Ini membuat skala Rp dan USD bisa dibandingkan "
            "tanpa memakai kurs nominal."
        )

else:
    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-title">💡 Belum yakin harus mengisi apa?</div>
            <div class="metric-note">
                Isi perkiraan saja. Nilai awal sudah diisi dengan nilai tengah dataset,
                jadi kamu bisa langsung menekan tombol rekomendasi tanpa mengubah semuanya.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Bagaimana sistem ini bekerja?"):
        st.markdown(
            """
            1. **Profil** — jawabanmu diubah menjadi enam indikator dengan skala 0–1.
            2. **Pencocokan** — setiap kegiatan punya kebutuhan ideal. Sistem mengukur seberapa
               dekat profilmu dengan kebutuhan itu.
            3. **Peringkat** — Random Forest mempelajari pola dari data dan memberi urutan
               kegiatan. Perhitungan SAW dipakai sebagai pembanding agar hasilnya bisa dijelaskan.
            """
        )

st.markdown(
    '<div class="footer-note">'
    "Alat bantu akademik · SMARTER + SAW + Random Forest · "
    "Hasil bukan keputusan final."
    "</div>",
    unsafe_allow_html=True,
)
