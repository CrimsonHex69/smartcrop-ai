import os
from io import BytesIO
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
    page_title="SmartCrop AI · Alpha Z",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEAM_NAME = "Alpha Z"
TEAM_MEMBERS = [
    ("Harshit Singh", "Team Leader · ML & Product Integration"),
    ("Anant Choudhary", "Backend & ML Data Pipeline"),
    ("Nitin Thakur", "Frontend & UI/UX"),
    ("Tanisha", "Research · Testing & Pitch"),
]

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
FEATURE_UNITS = {
    "N": "",
    "P": "",
    "K": "",
    "temperature": "°C",
    "humidity": "%",
    "ph": "",
    "rainfall": " mm",
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
CROP_ICONS = {
    "rice": "🌾", "maize": "🌽", "cotton": "🌿", "coffee": "☕", "banana": "🍌",
    "apple": "🍎", "mango": "🥭", "grapes": "🍇", "orange": "🍊", "papaya": "🍈",
    "watermelon": "🍉", "muskmelon": "🍈", "coconut": "🥥", "pomegranate": "🍎",
    "chickpea": "🫘", "kidneybeans": "🫘", "pigeonpeas": "🫘", "lentil": "🫘",
    "blackgram": "🫘", "mungbean": "🫘", "mothbeans": "🫘", "jute": "🌿",
}
# Explorer library is intentionally broader than the ML label set.
# It is for browsing/context only, not extra model predictions.
CROP_LIBRARY = {
    "Cereals & grains": [
        ("Rice", "🌾", "Cereal", "High-rainfall staple crop"),
        ("Maize", "🌽", "Cereal", "Versatile grain crop"),
        ("Wheat", "🌾", "Cereal", "Cool-season grain crop"),
        ("Bajra", "🌾", "Cereal", "Pearl millet"),
        ("Jowar", "🌾", "Cereal", "Sorghum"),
        ("Ragi", "🌾", "Cereal", "Finger millet"),
    ],
    "Pulses": [
        ("Chickpea", "🫘", "Pulse", "Protein-rich pulse"),
        ("Kidney beans", "🫘", "Pulse", "Common bean crop"),
        ("Pigeon pea", "🫘", "Pulse", "Long-duration pulse"),
        ("Lentil", "🫘", "Pulse", "Cool-season pulse"),
        ("Black gram", "🫘", "Pulse", "Urad dal crop"),
        ("Green gram", "🫘", "Pulse", "Mung bean"),
    ],
    "Oilseeds & commercial": [
        ("Mustard", "🌼", "Oilseed", "Cool-season oilseed"),
        ("Sunflower", "🌻", "Oilseed", "Oilseed crop"),
        ("Groundnut", "🥜", "Oilseed", "Groundnut / peanut"),
        ("Soybean", "🌱", "Oilseed", "Protein and oil crop"),
        ("Cotton", "🌿", "Commercial", "Fiber crop"),
        ("Sugarcane", "🎋", "Commercial", "Major sugar crop"),
        ("Jute", "🌿", "Commercial", "Natural fiber crop"),
    ],
    "Fruits & horticulture": [
        ("Banana", "🍌", "Fruit", "Tropical fruit crop"),
        ("Mango", "🥭", "Fruit", "Major tropical fruit"),
        ("Apple", "🍎", "Fruit", "Temperate fruit crop"),
        ("Grapes", "🍇", "Fruit", "Vine crop"),
        ("Orange", "🍊", "Fruit", "Citrus fruit"),
        ("Pomegranate", "🍎", "Fruit", "Arid/semi-arid fruit"),
        ("Tomato", "🍅", "Vegetable", "Warm-season horticulture crop"),
        ("Onion", "🧅", "Vegetable", "Bulb vegetable"),
    ],
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
    needed = FEATURES + ["label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=needed).copy()


@st.cache_resource(show_spinner=False)
def train_model(df):
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


@st.cache_data(show_spinner=False)
def get_class_profiles(df):
    return df.groupby("label")[FEATURES].mean()


def crop_icon(crop):
    return CROP_ICONS.get(str(crop).lower(), "🌱")


def clean_crop_name(crop):
    return str(crop).replace("_", " ").title()


def relative_position(value, lo, hi):
    if hi == lo:
        return 50.0
    return float((value - lo) / (hi - lo) * 100)


def predict(model, values, top_n=6):
    row = pd.DataFrame([values], columns=FEATURES)
    probs = model.predict_proba(row)[0]
    order = np.argsort(probs)[::-1][:top_n]
    return [(str(model.classes_[i]), float(probs[i])) for i in order]


def scenario_change(base_values, scenario_values):
    changes = []
    for key in FEATURES:
        delta = scenario_values[key] - base_values[key]
        if abs(delta) > 1e-9:
            sign = "+" if delta > 0 else "−"
            changes.append(f"{FEATURE_LABELS[key]} {sign}{abs(delta):.1f}{FEATURE_UNITS[key]}")
    return changes


def grounded_explanation(crop, values, class_profiles, feature_importance):
    class_mean = class_profiles.loc[crop]
    distances = []
    for key in FEATURES:
        spread = max(float(class_profiles[key].std()), 1e-6)
        normalized = abs(float(values[key]) - float(class_mean[key])) / spread
        distances.append((key, normalized))
    closest = [FEATURE_LABELS[k] for k, _ in sorted(distances, key=lambda x: x[1])[:3]]
    important = [FEATURE_LABELS[k] for k in feature_importance.head(3).index]
    return (
        f"The model's top prediction is {clean_crop_name(crop)}. "
        f"Within the benchmark data, the entered profile is relatively close to this crop class on {', '.join(closest)}. "
        f"The model also learned {', '.join(important)} as globally influential features. "
        "This is a benchmark-based decision-support result, not a guarantee of field performance."
    )


def optional_ai_explanation(crop, values, top3):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
Explain this machine-learning crop recommendation in simple language.
Predicted crop: {clean_crop_name(crop)}
Top model outputs: {[(clean_crop_name(c), round(s*100,1)) for c,s in top3]}
Inputs: {values}
Return 3 concise bullets and one caution sentence. Do not claim guaranteed yield, profit, or suitability.
"""
        response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        text = (response.text or "").strip()
        return text if text else None
    except Exception:
        return None


def build_report(crop, score, values, top6, accuracy):
    lines = [
        "SMARTCROP AI — CROP RECOMMENDATION",
        f"Team: {TEAM_NAME}",
        "",
        f"Recommended crop: {clean_crop_name(crop)}",
        f"Model score: {score * 100:.1f}%",
        f"Benchmark test accuracy: {accuracy * 100:.1f}%",
        "",
        "Field inputs",
    ]
    lines.extend(f"{FEATURE_LABELS[k]}: {values[k]:.1f}{FEATURE_UNITS[k]}" for k in FEATURES)
    lines += ["", "Top alternatives"]
    lines.extend(f"{i+1}. {clean_crop_name(c)} — {s*100:.1f}%" for i, (c, s) in enumerate(top6[:5]))
    lines += [
        "",
        "Disclaimer",
        "This is a machine-learning prototype trained on a benchmark dataset. It is not a substitute for local agronomic advice, current weather information, soil testing, or field trials.",
    ]
    return "\n".join(lines)


try:
    df = load_data()
    model, accuracy = train_model(df)
    ranges = {k: (float(df[k].min()), float(df[k].max())) for k in FEATURES}
    class_profiles = get_class_profiles(df)
    feature_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
except Exception as exc:
    st.error(f"Could not load the crop dataset: {exc}")
    st.info("Place Crop_recommendation.csv next to app.py or check your internet connection.")
    st.stop()

st.markdown(
    """
    <style>
    :root{--bg:#060b09;--panel:#0d1511;--panel2:#101b15;--line:#203a2a;--text:#eff7f1;--muted:#91a69a;--green:#6fe08b;--green2:#b9f3c3;--cyan:#83e9df;--warn:#e8c56f}
    .stApp{background:radial-gradient(circle at 8% 0%,rgba(80,210,119,.10),transparent 25%),radial-gradient(circle at 92% 10%,rgba(71,204,183,.06),transparent 24%),var(--bg);color:var(--text)}
    [data-testid="stHeader"]{background:rgba(6,11,9,.88)}
    .block-container{max-width:1240px;padding:1.7rem 1.2rem 3rem}
    .hero{padding:.3rem 0 1.15rem}.eyebrow{display:inline-flex;gap:7px;align-items:center;padding:7px 11px;border:1px solid #28503a;border-radius:999px;background:#0a1710;color:var(--green2);font-size:.76rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
    .hero h1{margin:.7rem 0 .25rem;font-size:clamp(2.4rem,5vw,4.2rem);line-height:.95;letter-spacing:-.05em;color:var(--text)}.hero h1 span{color:var(--green)}.hero p{max-width:800px;margin:0;color:var(--muted);font-size:1.03rem;line-height:1.7}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#0c2116;border:1px solid #244e35;color:var(--green2);font-size:.74rem;font-weight:800}
    .section{display:flex;justify-content:space-between;align-items:end;gap:1rem;margin:1.5rem 0 .75rem}.section h2{margin:0;font-size:1.15rem;color:var(--text)}.section span{font-size:.8rem;color:var(--muted)}
    .panel{background:linear-gradient(180deg,rgba(16,27,21,.98),rgba(11,18,15,.98));border:1px solid var(--line);border-radius:20px;padding:1.05rem}
    .metric{background:var(--panel2);border:1px solid var(--line);border-radius:16px;padding:1rem;min-height:108px}.metric .lab{font-size:.76rem;color:var(--muted)}.metric .val{font-size:1.45rem;font-weight:850;margin-top:.35rem;color:var(--text)}.metric .note{font-size:.73rem;color:#6f8477;margin-top:.25rem}
    .reco{background:linear-gradient(135deg,#0e2618,#10221c 55%,#112229);border:1px solid #2e674b;border-radius:22px;padding:1.4rem;box-shadow:0 18px 45px rgba(0,0,0,.22)}.reco .lab{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;color:var(--green2);font-weight:850}.reco .crop{font-size:clamp(2rem,4vw,3.2rem);font-weight:900;letter-spacing:-.04em;color:#f5fff6;margin:.2rem 0}.reco .score{color:#b5c9bd}.reco .score strong{color:var(--green2)}
    .why{background:#0b1210;border:1px solid var(--line);border-radius:17px;padding:1rem 1.05rem;color:#cbd8d1;line-height:1.55}.why b{color:var(--text)}
    .smallcard{background:#0b1410;border:1px solid var(--line);border-radius:15px;padding:.8rem}.smallcard .k{font-size:.74rem;color:var(--muted)}.smallcard .v{font-size:1.02rem;font-weight:800;margin-top:.2rem;color:var(--text)}
    .compare{background:#0b1410;border:1px solid var(--line);border-radius:16px;padding:.85rem 1rem;margin-bottom:.5rem}.compare .title{font-weight:800;color:var(--text)}
    .scenario{background:linear-gradient(180deg,#0e1913,#0b1210);border:1px solid #2c4d39;border-radius:18px;padding:1rem}
    .warn{background:#211a0c;border:1px solid #5e4d25;color:#f1d88e;border-radius:13px;padding:.8rem 1rem;font-size:.84rem}.good{background:#0b1f14;border:1px solid #265b3e;color:#aeeec0;border-radius:13px;padding:.8rem 1rem;font-size:.84rem}
    .library-card{background:#0b1410;border:1px solid var(--line);border-radius:16px;padding:1rem;height:100%}.library-card .name{font-size:1.02rem;font-weight:850;color:var(--text)}.library-card .meta{font-size:.72rem;color:var(--green2);margin:.18rem 0 .45rem;text-transform:uppercase;letter-spacing:.08em}.library-card .desc{font-size:.82rem;color:var(--muted);line-height:1.45}
    .stButton>button,.stDownloadButton>button{border-radius:12px!important;border:1px solid #2d6648!important;background:linear-gradient(135deg,#71df8d,#4cbd79)!important;color:#061108!important;font-weight:850!important;min-height:46px;box-shadow:0 8px 20px rgba(74,187,115,.12)}
    .stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.05)}
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#0b1411!important;color:#eef7f0!important;border:1px solid #294338!important;border-radius:11px!important}
    [data-testid="stNumberInput"] button{background:#14251c!important;color:#bcebc8!important;border:0!important}
    .stProgress>div>div>div>div{background:linear-gradient(90deg,#5dd37e,#83e9df)!important}.stProgress>div>div{background:#17251f!important}
    [data-testid="stExpander"]{border:1px solid var(--line);border-radius:15px;background:#0a120f}
    .stCaption,.stMarkdown p{color:#97aba0}.footer{color:#6d8075;font-size:.75rem;text-align:center;padding-top:1.2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">🌱 AI · SOIL · CLIMATE · DECISION SUPPORT · TEAM ALPHA Z</div>
      <h1>SmartCrop <span>AI</span></h1>
      <p>Seven field measurements in. A crop recommendation out. Then explore why the model chose it, compare alternatives, and test what happens when conditions change.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
for col, label, value, note in [
    (m1, "BENCHMARK ACCURACY", f"{accuracy*100:.1f}%", "held-out benchmark split"),
    (m2, "CROP CLASSES", f"{df['label'].nunique()}", "model-backed classes"),
    (m3, "INPUT SIGNALS", "7", "soil + climate"),
    (m4, "TRAINING ROWS", f"{len(df):,}", "benchmark samples"),
]:
    with col:
        st.markdown(f'<div class="metric"><div class="lab">{label}</div><div class="val">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

if "use_demo" not in st.session_state:
    st.session_state.use_demo = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.markdown('<div class="section"><h2>1 · Field conditions</h2><span>Use measured soil values when available</span></div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1.2, 2, 1.2])
    with b1:
        if st.button("Load demo scenario", use_container_width=True):
            st.session_state.use_demo = True
            st.rerun()
    with b2:
        st.caption("Use the demo scenario for a fast presentation; replace it with measured values for actual testing.")
    with b3:
        if st.button("Reset inputs", use_container_width=True):
            st.session_state.use_demo = False
            st.session_state.last_result = None
            st.rerun()

    with st.form("crop_form"):
        def initial_value(name):
            if st.session_state.use_demo:
                return DEMO_VALUES[name]
            return (ranges[name][0] + ranges[name][1]) / 2
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("Nitrogen (N)", min_value=ranges["N"][0], max_value=ranges["N"][1], value=initial_value("N"), step=1.0)
            p = st.number_input("Phosphorus (P)", min_value=ranges["P"][0], max_value=ranges["P"][1], value=initial_value("P"), step=1.0)
            k = st.number_input("Potassium (K)", min_value=ranges["K"][0], max_value=ranges["K"][1], value=initial_value("K"), step=1.0)
        with c2:
            temp = st.number_input("Temperature (°C)", min_value=ranges["temperature"][0], max_value=ranges["temperature"][1], value=initial_value("temperature"), step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=ranges["humidity"][0], max_value=ranges["humidity"][1], value=initial_value("humidity"), step=0.1)
        with c3:
            ph = st.number_input("Soil pH", min_value=ranges["ph"][0], max_value=ranges["ph"][1], value=initial_value("ph"), step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=ranges["rainfall"][0], max_value=ranges["rainfall"][1], value=initial_value("rainfall"), step=1.0)
        submitted = st.form_submit_button("🌾 Analyze field", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    values = {"N": n, "P": p, "K": k, "temperature": temp, "humidity": humidity, "ph": ph, "rainfall": rainfall}
    top6 = predict(model, values, top_n=6)
    st.session_state.last_result = {"values": values, "top6": top6}

result = st.session_state.last_result
if result:
    values = result["values"]
    top6 = result["top6"]
    crop, score = top6[0]

    st.markdown('<div class="section"><h2>2 · Smart recommendation</h2><span>Model-backed result</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="reco"><div class="lab">Recommended crop</div><div class="crop">{crop_icon(crop)} {clean_crop_name(crop)}</div><div class="score">Model score: <strong>{score*100:.1f}%</strong> · benchmark model output</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section"><h2>Top recommendations</h2><span>Six highest model scores</span></div>', unsafe_allow_html=True)
        for idx, (name, sc) in enumerate(top6, start=1):
            st.markdown('<div class="compare">', unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns([.11, .57, .24])
            with cc1:
                st.markdown(f"**{idx}**")
            with cc2:
                st.markdown(f"<div class='title'>{crop_icon(name)} {clean_crop_name(name)}</div>", unsafe_allow_html=True)
                st.progress(min(max(sc, 0.0), 1.0))
            with cc3:
                st.markdown(f"**{sc*100:.1f}%**")
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section"><h2>Why this crop?</h2><span>Explainable context</span></div>', unsafe_allow_html=True)
        ai_text = optional_ai_explanation(crop, values, top6)
        text = ai_text if ai_text else grounded_explanation(crop, values, class_profiles, feature_importance)
        st.markdown(f'<div class="why"><b>{crop_icon(crop)} Why {clean_crop_name(crop)}?</b><br><br>{text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        all_in_range = all(ranges[k][0] <= values[k] <= ranges[k][1] for k in FEATURES)
        if all_in_range:
            st.markdown('<div class="good" style="margin-top:.75rem">✓ All seven inputs are within the benchmark dataset range.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn" style="margin-top:.75rem">⚠ One or more inputs are outside the benchmark range. Treat the model output cautiously.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>3 · Field snapshot</h2><span>Exactly what the model received</span></div>', unsafe_allow_html=True)
    cols = st.columns(7)
    for i, key in enumerate(FEATURES):
        with cols[i]:
            st.markdown(f'<div class="smallcard"><div class="k">{FEATURE_LABELS[key]}</div><div class="v">{values[key]:.1f}{FEATURE_UNITS[key]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>4 · Compare the leading crops</h2><span>Side-by-side benchmark context</span></div>', unsafe_allow_html=True)
    compare_rows = []
    for name, sc in top6[:3]:
        profile = class_profiles.loc[name]
        compare_rows.append({
            "Crop": f"{crop_icon(name)} {clean_crop_name(name)}",
            "Model score": f"{sc*100:.1f}%",
            "Avg. pH": f"{profile['ph']:.2f}",
            "Avg. temp": f"{profile['temperature']:.1f}°C",
            "Avg. rainfall": f"{profile['rainfall']:.0f} mm",
        })
    st.table(pd.DataFrame(compare_rows))

    st.markdown('<div class="section"><h2>5 · What-if simulator</h2><span>Change conditions and see how the model responds</span></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="scenario">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            s_temp = st.slider("Temperature (°C)", ranges["temperature"][0], ranges["temperature"][1], float(values["temperature"]), step=0.5)
            s_hum = st.slider("Humidity (%)", ranges["humidity"][0], ranges["humidity"][1], float(values["humidity"]), step=1.0)
        with s2:
            s_ph = st.slider("Soil pH", ranges["ph"][0], ranges["ph"][1], float(values["ph"]), step=0.1)
            s_rain = st.slider("Rainfall (mm)", ranges["rainfall"][0], ranges["rainfall"][1], float(values["rainfall"]), step=5.0)
        with s3:
            s_n = st.slider("Nitrogen (N)", ranges["N"][0], ranges["N"][1], float(values["N"]), step=1.0)
            s_p = st.slider("Phosphorus (P)", ranges["P"][0], ranges["P"][1], float(values["P"]), step=1.0)
            s_k = st.slider("Potassium (K)", ranges["K"][0], ranges["K"][1], float(values["K"]), step=1.0)
        scenario_values = {"N": s_n, "P": s_p, "K": s_k, "temperature": s_temp, "humidity": s_hum, "ph": s_ph, "rainfall": s_rain}
        scenario_top = predict(model, scenario_values, top_n=3)
        scenario_crop, scenario_score = scenario_top[0]
        changed = scenario_change(values, scenario_values)
        if changed:
            st.markdown(f"**Scenario recommendation:** {crop_icon(scenario_crop)} {clean_crop_name(scenario_crop)} · {scenario_score*100:.1f}%")
            st.caption("Changed: " + ", ".join(changed))
            if scenario_crop != crop:
                st.markdown(f'<div class="pill">Recommendation changed: {clean_crop_name(crop)} → {clean_crop_name(scenario_crop)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pill">The same top class remains under this scenario.</div>', unsafe_allow_html=True)
        else:
            st.caption("Move a slider to explore how different field conditions affect the model output.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>6 · Crop explorer</h2><span>Broader crop knowledge library · not extra model classes</span></div>', unsafe_allow_html=True)
    category = st.selectbox("Browse a crop category", list(CROP_LIBRARY.keys()))
    lib_cols = st.columns(4)
    for idx, (name, icon, kind, desc) in enumerate(CROP_LIBRARY[category]):
        with lib_cols[idx % 4]:
            st.markdown(f'<div class="library-card"><div style="font-size:1.8rem">{icon}</div><div class="name">{name}</div><div class="meta">{kind}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>7 · Benchmark position</h2><span>Where your inputs sit in the training data</span></div>', unsafe_allow_html=True)
    with st.expander("Open benchmark-range view"):
        pcols = st.columns(4)
        for i, key in enumerate(FEATURES):
            pos = relative_position(values[key], *ranges[key])
            with pcols[i % 4]:
                st.caption(FEATURE_LABELS[key])
                st.progress(int(round(max(0, min(100, pos)))))
                st.write(f"{pos:.0f}% through benchmark range")

    st.markdown('<div class="section"><h2>8 · Farmer-ready report</h2><span>Save or share the current recommendation</span></div>', unsafe_allow_html=True)
    report = build_report(crop, score, values, top6, accuracy)
    st.download_button("⬇ Download recommendation report", data=report, file_name="smartcrop_recommendation.txt", mime="text/plain", use_container_width=True)

else:
    st.markdown('<div class="section"><h2>How the MVP works</h2><span>One clean end-to-end flow</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, step, title, body in [
        (c1, "01", "Enter field conditions", "Soil nutrients, temperature, humidity, pH and rainfall."),
        (c2, "02", "Get a recommendation", "A Random Forest model ranks the crop classes in the benchmark dataset."),
        (c3, "03", "Explore the decision", "See why the model chose it, compare options and run a what-if scenario."),
    ]:
        with col:
            st.markdown(f'<div class="smallcard"><div class="k">STEP {step}</div><div class="v">{title}</div><p>{body}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section"><h2>9 · Team Alpha Z</h2><span>IDEAFORGE 2.0 · AI Technology & Digital Innovation</span></div>', unsafe_allow_html=True)
tcols = st.columns(4)
for col, (name, role) in zip(tcols, TEAM_MEMBERS):
    with col:
        role_short = role.split(" · ")[0]
        st.markdown(f'<div class="smallcard"><div class="k">{role_short.upper()}</div><div class="v">{name}</div><p style="margin:.35rem 0 0;color:#91a69a;font-size:.78rem">{role}</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌱 SmartCrop AI")
    st.caption("Team Alpha Z")
    st.write("AI/ML decision-support prototype for crop recommendation.")
    st.metric("Benchmark accuracy", f"{accuracy*100:.1f}%")
    st.caption("Held-out benchmark performance, not real-world farm accuracy.")
    st.divider()
    st.markdown("**Model**")
    st.write("Random Forest · 300 trees")
    st.markdown("**Inputs**")
    st.write("N · P · K · temperature · humidity · pH · rainfall")
    st.divider()
    st.markdown("**Team Alpha Z**")
    for name, role in TEAM_MEMBERS:
        st.write(f"**{name}** — {role}")
    st.markdown("**Optional AI layer**")
    st.write("Gemini explanation only when an API key is configured.")

with st.expander("Model information"):
    st.write(f"Training rows: {len(df):,}")
    st.write(f"Crop classes: {df['label'].nunique()}")
    imp_df = pd.DataFrame({"Feature": [FEATURE_LABELS[k] for k in feature_importance.index], "Importance": feature_importance.values})
    st.dataframe(imp_df, hide_index=True, use_container_width=True)

st.markdown('<div class="footer">SmartCrop AI · Team Alpha Z · IDEAFORGE 2.0 · Benchmark-based decision support prototype</div>', unsafe_allow_html=True)
