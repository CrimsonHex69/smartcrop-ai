import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from google import genai
except Exception:
    genai = None

st.set_page_config(
    page_title="SmartCrop AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_URL = "https://raw.githubusercontent.com/the-amazing-atharva/Crop-Recommendation/main/Crop_recommendation.csv"
DATA_FILE = Path("Crop_recommendation.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}

DEMO_VALUES = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 25.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 200.0,
}


def download_dataset_if_needed():
    if DATA_FILE.exists():
        return
    req = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as response:
        DATA_FILE.write_bytes(response.read())


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_FILE.exists():
        download_dataset_if_needed()
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=FEATURES + ["label"])


@st.cache_resource(show_spinner=False)
def train_model(df):
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def get_feature_ranges(df):
    return {
        key: (float(df[key].min()), float(df[key].max()))
        for key in FEATURES
    }


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def relative_position(value, lo, hi):
    if hi == lo:
        return 50
    return ((value - lo) / (hi - lo)) * 100


def crop_explanation(crop, values, top_features):
    signals = ", ".join(top_features)
    return (
        f"The model selected {crop.title()} from the seven soil and climate inputs provided. "
        f"The model's globally important signals in this trained dataset include {signals}. "
        "Treat this output as a prototype decision-support result, not a guaranteed farming recommendation."
    )


def gemini_explanation(crop, values, top3):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
Explain this crop-recommendation prototype to a farmer in simple language.
Prediction: {crop}
Top model outputs: {top3}
Inputs: N={values['N']}, P={values['P']}, K={values['K']}, temperature={values['temperature']} C,
humidity={values['humidity']} %, pH={values['ph']}, rainfall={values['rainfall']} mm.
Return exactly 3 concise bullet points and then one caution sentence.
Do not claim guaranteed yield, profit, or suitability.
"""
        response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        text = (response.text or "").strip()
        return text if text else None
    except Exception:
        return None


try:
    df = load_data()
    model, accuracy = train_model(df)
    ranges = get_feature_ranges(df)
except Exception as exc:
    st.error(f"Could not load the crop dataset: {exc}")
    st.info("Place Crop_recommendation.csv in the same folder as app.py and refresh the app.")
    st.stop()

# ---------- Styling ----------
st.markdown(
    """
    <style>
    :root {
        --sc-bg: #070b0d;
        --sc-panel: #0e1417;
        --sc-panel-2: #111a1d;
        --sc-border: #203034;
        --sc-text: #eff7f1;
        --sc-muted: #8fa19a;
        --sc-green: #78d98a;
        --sc-green-2: #b7f0bc;
        --sc-cyan: #7be5dc;
        --sc-warning: #f2c66d;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(120,217,138,.08), transparent 26%),
            radial-gradient(circle at 90% 8%, rgba(123,229,220,.06), transparent 24%),
            var(--sc-bg);
        color: var(--sc-text);
    }

    [data-testid="stHeader"] { background: rgba(7,11,13,.86); }
    [data-testid="stToolbar"] { right: .6rem; }
    [data-testid="stSidebar"] {
        background: #091012;
        border-right: 1px solid var(--sc-border);
    }
    [data-testid="stSidebar"] * { color: var(--sc-text) !important; }

    .block-container {
        max-width: 1260px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: .6rem 0 1.5rem;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 11px;
        border: 1px solid #27443a;
        border-radius: 999px;
        background: #0b1711;
        color: var(--sc-green-2);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .7rem 0 .25rem;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 1;
        letter-spacing: -.045em;
        color: var(--sc-text);
    }
    .hero h1 span { color: var(--sc-green); }
    .hero p {
        margin: 0;
        max-width: 760px;
        color: var(--sc-muted);
        font-size: 1.05rem;
    }

    .section-title {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 1rem;
        margin: 1.5rem 0 .7rem;
    }
    .section-title h2 {
        margin: 0;
        font-size: 1.15rem;
        color: var(--sc-text);
    }
    .section-title span {
        color: var(--sc-muted);
        font-size: .82rem;
    }

    .panel {
        background: linear-gradient(180deg, rgba(17,26,29,.98), rgba(12,18,20,.98));
        border: 1px solid var(--sc-border);
        border-radius: 20px;
        padding: 1.1rem 1.1rem .8rem;
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
    }

    .metric-card {
        background: var(--sc-panel-2);
        border: 1px solid var(--sc-border);
        border-radius: 16px;
        padding: 1rem;
        min-height: 118px;
    }
    .metric-label { color: var(--sc-muted); font-size: .8rem; margin-bottom: .4rem; }
    .metric-value { color: var(--sc-text); font-size: 1.5rem; font-weight: 800; }
    .metric-note { color: #6f847c; font-size: .76rem; margin-top: .25rem; }

    .recommendation {
        background: linear-gradient(135deg, #0d261a 0%, #10251f 55%, #102027 100%);
        border: 1px solid #2c624b;
        border-radius: 22px;
        padding: 1.3rem 1.4rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.03), 0 18px 40px rgba(0,0,0,.24);
    }
    .recommendation .label {
        color: var(--sc-green-2);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .recommendation .crop {
        color: #f5fff6;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 850;
        line-height: 1;
        margin: .25rem 0 .4rem;
    }
    .recommendation .score { color: #b6cdc3; font-size: .92rem; }
    .recommendation .score strong { color: var(--sc-green-2); }

    .reason {
        background: #0b1113;
        border: 1px solid var(--sc-border);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        color: #c9d7d1;
        line-height: 1.55;
    }
    .reason-title { color: var(--sc-text); font-weight: 800; margin-bottom: .25rem; }

    .mini-card {
        background: #0c1315;
        border: 1px solid var(--sc-border);
        border-radius: 15px;
        padding: .85rem .95rem;
    }
    .mini-card .k { color: var(--sc-muted); font-size: .78rem; }
    .mini-card .v { color: var(--sc-text); font-size: 1.05rem; font-weight: 750; margin-top: .2rem; }

    .footer-note {
        color: #6f817a;
        font-size: .78rem;
        text-align: center;
        padding-top: 1rem;
    }

    /* Streamlit widget styling */
    [data-testid="stNumberInput"] input {
        background: #0c1315 !important;
        color: #edf8f1 !important;
        border: 1px solid #26383c !important;
        border-radius: 12px !important;
    }
    [data-testid="stNumberInput"] button {
        background: #14201f !important;
        color: #bce7c7 !important;
        border: 0 !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid #2b5744 !important;
        background: linear-gradient(135deg, #6fd787, #4ebc7b) !important;
        color: #07110a !important;
        font-weight: 800 !important;
        min-height: 46px;
        box-shadow: 0 8px 20px rgba(79,190,124,.15);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #7ce697 !important;
        filter: brightness(1.04);
    }
    .stFormSubmitButton > button {
        border-radius: 13px !important;
        min-height: 52px;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #5ed37c, #7be5dc) !important;
    }
    .stProgress > div > div {
        background: #152224 !important;
    }
    [data-testid="stMetricValue"] { color: var(--sc-text) !important; }
    [data-testid="stMetricLabel"] { color: var(--sc-muted) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--sc-border); border-radius: 14px; overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid var(--sc-border); border-radius: 15px; background: #0c1315; }
    .stCaption, .stMarkdown p { color: #9bb0a8; }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">🌱 AI · SOIL · CLIMATE</div>
        <h1>SmartCrop <span>AI</span></h1>
        <p>Decision-support prototype that analyzes soil and climate conditions to recommend suitable crop classes from a trained machine-learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Top information row ----------
a, b, c = st.columns(3)
with a:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">MODEL ACCURACY</div><div class="metric-value">{accuracy * 100:.1f}%</div><div class="metric-note">Held-out benchmark split</div></div>',
        unsafe_allow_html=True,
    )
with b:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">CROP CLASSES</div><div class="metric-value">{df["label"].nunique()}</div><div class="metric-note">Learned from benchmark labels</div></div>',
        unsafe_allow_html=True,
    )
with c:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">INPUT SIGNALS</div><div class="metric-value">7</div><div class="metric-note">Soil + climate parameters</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title"><h2>Crop recommendation</h2><span>Enter values from a soil test or current observations</span></div>', unsafe_allow_html=True)

# ---------- Inputs ----------
if "use_demo" not in st.session_state:
    st.session_state.use_demo = False
if "has_prediction" not in st.session_state:
    st.session_state.has_prediction = False

with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    top_actions = st.columns([1, 3])
    with top_actions[0]:
        if st.button("Load demo values", use_container_width=True):
            st.session_state.use_demo = True
            st.rerun()
    with top_actions[1]:
        st.caption("Demo values are for presentation only. Replace them with measured values for an actual test.")

    def value_for(name):
        if st.session_state.use_demo:
            return clamp(DEMO_VALUES[name], *ranges[name])
        return sum(ranges[name]) / 2

    with st.form("crop_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("Nitrogen (N)", min_value=ranges["N"][0], max_value=ranges["N"][1], value=value_for("N"), step=1.0)
            p = st.number_input("Phosphorus (P)", min_value=ranges["P"][0], max_value=ranges["P"][1], value=value_for("P"), step=1.0)
            k = st.number_input("Potassium (K)", min_value=ranges["K"][0], max_value=ranges["K"][1], value=value_for("K"), step=1.0)
        with c2:
            temp = st.number_input("Temperature (°C)", min_value=ranges["temperature"][0], max_value=ranges["temperature"][1], value=value_for("temperature"), step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=ranges["humidity"][0], max_value=ranges["humidity"][1], value=value_for("humidity"), step=0.1)
        with c3:
            ph = st.number_input("Soil pH", min_value=ranges["ph"][0], max_value=ranges["ph"][1], value=value_for("ph"), step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=ranges["rainfall"][0], max_value=ranges["rainfall"][1], value=value_for("rainfall"), step=1.0)

        submitted = st.form_submit_button("🌾  ANALYZE & RECOMMEND CROP", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Result ----------
if submitted:
    values = {
        "N": n,
        "P": p,
        "K": k,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    input_df = pd.DataFrame([values], columns=FEATURES)
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1][:3]
    top3 = [(classes[i], float(probabilities[i])) for i in order]
    crop, score = top3[0]
    st.session_state.has_prediction = True
    st.session_state.last_result = {
        "values": values,
        "crop": crop,
        "score": score,
        "top3": top3,
    }

result = st.session_state.get("last_result")
if result:
    values = result["values"]
    crop = result["crop"]
    score = result["score"]
    top3 = result["top3"]

    st.markdown('<div class="section-title"><h2>Recommendation</h2><span>Model output</span></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="recommendation">
            <div class="label">Recommended crop</div>
            <div class="crop">🌾 {crop.title()}</div>
            <div class="score">Model score: <strong>{score * 100:.1f}%</strong> · Based on the benchmark model output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-title"><h2>Top model outputs</h2><span>Alternatives</span></div>', unsafe_allow_html=True)
        for i, (name, sc) in enumerate(top3, start=1):
            cols = st.columns([0.12, 0.53, 0.2, 0.15])
            with cols[0]:
                st.markdown(f"**{i}.**")
            with cols[1]:
                st.markdown(f"**{name.title()}**")
                st.progress(min(max(sc, 0.0), 1.0))
            with cols[2]:
                st.markdown(f"`{sc * 100:.1f}%`")
            with cols[3]:
                if i == 1:
                    st.markdown("🌱")
                else:
                    st.markdown("•")

    with right:
        st.markdown('<div class="section-title"><h2>Why this crop?</h2><span>Model context</span></div>', unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        top_features = [FEATURE_LABELS[x] for x in importances.head(3).index]
        explanation = gemini_explanation(crop, values, [(c, round(s, 3)) for c, s in top3])
        if explanation:
            st.markdown(f'<div class="reason"><div class="reason-title">AI explanation</div>{explanation.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="reason"><div class="reason-title">How the model arrived here</div>{crop_explanation(crop, values, top_features)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title"><h2>Input profile</h2><span>Values used for this prediction</span></div>', unsafe_allow_html=True)
    metric_cols = st.columns(7)
    for i, key in enumerate(FEATURES):
        val = values[key]
        label = FEATURE_LABELS[key]
        with metric_cols[i]:
            unit = {
                "N": "", "P": "", "K": "", "temperature": "°C", "humidity": "%", "ph": "", "rainfall": " mm"
            }[key]
            shown = f"{val:.1f}{unit}"
            st.markdown(f'<div class="mini-card"><div class="k">{label}</div><div class="v">{shown}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><h2>Benchmark position</h2><span>Where each input sits within the training-data range</span></div>', unsafe_allow_html=True)
    pos_cols = st.columns(4)
    for i, key in enumerate(FEATURES):
        pos = relative_position(values[key], *ranges[key])
        with pos_cols[i % 4]:
            st.caption(FEATURE_LABELS[key])
            st.progress(int(round(max(0, min(100, pos)))))
            st.write(f"{pos:.0f}% of benchmark range")

    st.markdown('<div class="section-title"><h2>Farmer-ready summary</h2><span>Simple export for the demo</span></div>', unsafe_allow_html=True)
    report = f"""SMARTCROP AI — CROP RECOMMENDATION\n\nRecommended crop: {crop.title()}\nModel score: {score * 100:.1f}%\n\nInputs\nN: {values['N']:.1f}\nP: {values['P']:.1f}\nK: {values['K']:.1f}\nTemperature: {values['temperature']:.1f} C\nHumidity: {values['humidity']:.1f}%\nSoil pH: {values['ph']:.1f}\nRainfall: {values['rainfall']:.1f} mm\n\nTop alternatives\n""" + "\n".join([f"{i+1}. {c.title()} — {s*100:.1f}%" for i, (c, s) in enumerate(top3)]) + "\n\nDisclaimer\nThis is a machine-learning prototype trained on a benchmark dataset. It is not a substitute for local agronomic advice, laboratory soil testing, current weather forecasts, or field trials."
    st.download_button("⬇ Download recommendation report", data=report, file_name="smartcrop_recommendation.txt", mime="text/plain", use_container_width=True)

# ---------- Sidebar / Details ----------
with st.sidebar:
    st.markdown("## 🌱 SmartCrop AI")
    st.write("A machine-learning decision-support prototype for crop recommendation.")
    st.metric("Benchmark accuracy", f"{accuracy * 100:.1f}%")
    st.caption("Measured on a held-out split of the benchmark dataset; not a real-world farm accuracy guarantee.")
    st.divider()
    st.markdown("**Prototype stack**")
    st.write("Python · Streamlit · Random Forest · optional Gemini")
    st.markdown("**Inputs**")
    st.write("N · P · K · temperature · humidity · pH · rainfall")

with st.expander("How the prototype works"):
    st.write("1. The farmer provides seven soil and climate measurements.")
    st.write("2. A Random Forest classifier trained on a public crop-recommendation benchmark predicts a crop class.")
    st.write("3. The app shows the top three model outputs and a simple explanation.")
    st.write("4. An optional Gemini key can generate a more natural-language explanation, but the core predictor does not depend on Gemini.")

with st.expander("Model information"):
    st.write(f"Dataset rows used: {len(df):,}")
    st.write(f"Crop classes: {df['label'].nunique()}")
    st.write("Features: N, P, K, temperature, humidity, pH, rainfall")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp_df = pd.DataFrame({"Feature": [FEATURE_LABELS[x] for x in imp.index], "Importance": imp.values})
    st.dataframe(imp_df, hide_index=True, use_container_width=True)

st.markdown('<div class="footer-note">Prototype only · Use measured soil data and local agricultural guidance before making real planting decisions.</div>', unsafe_allow_html=True)
