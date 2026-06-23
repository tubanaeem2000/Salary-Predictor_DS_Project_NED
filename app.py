# app.py — Salary Predictor (GUI Edition)
#
# A Streamlit interface for the Random Forest salary-prediction model.
# This is a GUI rebuild of the original terminal tool — the CLI version
# is preserved in cli_app.py for anyone who prefers the terminal.
#
# Run with: streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# Load model artefacts (cached so the 190MB model loads only once)
# ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "best_model_rf.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    return model, encoders


rf_model, label_encoders = load_artifacts()

FEATURES = [
    "job_title", "experience_years", "education_level",
    "skills_count", "industry", "company_size",
    "location", "remote_work", "certifications",
]

CATEGORICAL = [
    "job_title", "education_level", "industry",
    "company_size", "location", "remote_work",
]


def predict_salary(**kwargs):
    row = dict(kwargs)
    for col in CATEGORICAL:
        le = label_encoders[col]
        row[col] = le.transform([row[col]])[0]
    X = pd.DataFrame([row])[FEATURES]
    return round(float(rf_model.predict(X)[0]), 2)


# ──────────────────────────────────────────────────────────────────
# Styling — black / sea-green gradient, terminal-inspired signature
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg-black: #050a08;
    --bg-deep: #0a1f17;
    --sea-green: #2E8B57;
    --sea-green-bright: #3ddc97;
    --text-light: #eafff3;
    --text-dim: #8fb9a4;
    --card-bg: rgba(13, 30, 24, 0.55);
    --card-border: rgba(61, 220, 151, 0.22);
}

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 12% -10%, var(--bg-deep) 0%, var(--bg-black) 55%, #000000 100%) !important;
    color: var(--text-light);
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.1;
    background: linear-gradient(90deg, #ffffff 0%, var(--sea-green-bright) 55%, var(--sea-green) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.2rem 0 0 0;
}
.hero-sub {
    color: var(--text-dim);
    font-size: 1rem;
    margin-top: 0.35rem;
    max-width: 46rem;
}
.gp-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--sea-green-bright);
    margin-bottom: 0.3rem;
}

.gp-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.5rem 1.7rem 0.4rem 1.7rem;
    box-shadow: 0 0 40px rgba(46, 139, 87, 0.07);
    margin-bottom: 1rem;
}

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    color: var(--text-dim) !important;
    font-weight: 500;
}
div[data-baseweb="select"] > div {
    background-color: #0d1f17 !important;
    border-color: var(--card-border) !important;
    color: var(--text-light) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--sea-green-bright) !important;
    box-shadow: 0 0 10px rgba(61, 220, 151, 0.7);
}
[data-testid="stSliderTickBar"] { display: none; }

div.stButton > button, button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, var(--sea-green) 0%, var(--sea-green-bright) 100%);
    color: #04140d;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.6rem;
    letter-spacing: 0.02em;
    box-shadow: 0 0 22px rgba(61, 220, 151, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
}
div.stButton > button:hover, button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 30px rgba(61, 220, 151, 0.55);
}

.terminal {
    background: #03100a;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.5rem 1.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--sea-green-bright);
    box-shadow: 0 0 50px rgba(46, 139, 87, 0.16) inset;
    min-height: 230px;
}
.terminal .line { color: var(--text-dim); font-size: 0.85rem; line-height: 1.7; }
.terminal .prompt { color: var(--sea-green-bright); }
.terminal .result-label {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-top: 0.6rem;
}
.terminal .result-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(61, 220, 151, 0.6);
    margin: 0.1rem 0 0.6rem 0;
}
.cursor {
    display: inline-block;
    width: 8px; height: 1rem;
    background: var(--sea-green-bright);
    animation: blink 1s step-start infinite;
    vertical-align: middle;
}
@keyframes blink { 50% { opacity: 0; } }

hr.gp-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--card-border), transparent);
    margin: 1.5rem 0;
}
.gp-footer {
    color: var(--text-dim);
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="gp-eyebrow">RANDOM FOREST · ~95.6% R² · 9 FEATURES</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Salary Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Estimate a fair market salary from role, experience, '
    'and skills — the same model that used to only speak through a terminal.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)

opts = {k: list(v.classes_) for k, v in label_encoders.items()}

col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown('<div class="gp-card">', unsafe_allow_html=True)
    st.markdown('<div class="gp-eyebrow">Input</div>', unsafe_allow_html=True)
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            job_title = st.selectbox("Job title", opts["job_title"])
            education_level = st.selectbox("Education level", opts["education_level"])
            industry = st.selectbox("Industry", opts["industry"])
            company_size = st.selectbox("Company size", opts["company_size"])
        with c2:
            location = st.selectbox("Location", opts["location"])
            remote_work = st.selectbox("Remote work", opts["remote_work"])
            experience_years = st.slider("Experience (years)", 0, 20, 3)
            skills_count = st.slider("Skills count", 1, 19, 6)

        certifications = st.slider("Certifications", 0, 5, 1)
        submitted = st.form_submit_button("🔮  Predict salary")
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="gp-eyebrow">Output</div>', unsafe_allow_html=True)

    if submitted:
        salary = predict_salary(
            job_title=job_title,
            experience_years=experience_years,
            education_level=education_level,
            skills_count=skills_count,
            industry=industry,
            company_size=company_size,
            location=location,
            remote_work=remote_work,
            certifications=certifications,
        )
        st.markdown(f"""
        <div class="terminal">
            <div class="line"><span class="prompt">$</span> predict_salary --job "{job_title}" --exp {experience_years}y --edu "{education_level}"</div>
            <div class="line">&nbsp;</div>
            <div class="result-label">PREDICTED ANNUAL SALARY</div>
            <div class="result-value">${salary:,.2f}</div>
            <div class="line">model: RandomForestRegressor &nbsp;·&nbsp; 6 categorical features encoded &nbsp;·&nbsp; 3 numeric<span class="cursor"></span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="terminal">
            <div class="line"><span class="prompt">$</span> waiting for input<span class="cursor"></span></div>
            <div class="line">&nbsp;</div>
            <div class="line">Fill in the form on the left and run a prediction</div>
            <div class="line">to see the estimated salary printed here.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="gp-footer">Trained on a 9-feature job-market dataset · '
    'GUI rebuilt from the original CLI tool (see cli_app.py)</div>',
    unsafe_allow_html=True,
)
