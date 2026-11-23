import io
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from doctor_module.ui import render_nearby_cardiologists
except Exception:  # the doctor feature is optional
    render_nearby_cardiologists = None

URL = "https://raw.githubusercontent.com/imteekay/heart-failure-prediction/main/heart.csv"

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

FRIENDLY = {
    "Age": "Age", "Sex": "Sex", "ChestPainType": "Chest pain type",
    "RestingBP": "Resting blood pressure", "Cholesterol": "Cholesterol",
    "FastingBS": "Fasting blood sugar", "RestingECG": "Resting ECG",
    "MaxHR": "Max heart rate", "ExerciseAngina": "Exercise-induced angina",
    "Oldpeak": "ST depression (Oldpeak)", "ST_Slope": "ST slope",
}
CHEST_PAIN = {"TA": "Typical angina", "ATA": "Atypical angina",
              "NAP": "Non-anginal pain", "ASY": "Asymptomatic"}


@st.cache_resource(show_spinner="Training and comparing the models...")
def train_models():
    """Train every candidate model, rank them with cross-validation, and pick the best one live."""
    df = pd.read_csv(URL).drop_duplicates()
    for col in ["Cholesterol", "RestingBP"]:
        df[col] = df[col].replace(0, np.nan)  # fake zeros -> missing

    X, y = df.drop("HeartDisease", axis=1), df["HeartDisease"]
    num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    rows, fitted = [], {}
    for name, clf in candidates.items():
        pipe = Pipeline([("preprocessor", clone(pre)), ("model", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()  # train data only
        pipe.fit(X_train, y_train)
        pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob), "CV F1": cv_f1})
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("CV F1", ascending=False).reset_index(drop=True)
    baseline = {c: (X_train[c].median() if c in num else X_train[c].mode()[0]) for c in X.columns}
    return {"models": fitted, "results": results, "best": results.loc[0, "Model"], "baseline": baseline,
            "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test)}


def explain(model, patient, baseline):
    """How much each value pushes the risk up or down compared with a typical patient.
    Each value is swapped for the typical one (median / most common) and we see how the risk changes."""
    p0 = model.predict_proba(patient)[0][1]
    out = {}
    for col in patient.columns:
        alt = patient.copy()
        alt[col] = baseline[col]
        out[col] = p0 - model.predict_proba(alt)[0][1]
    s = pd.Series(out).sort_values(key=abs, ascending=False)
    s.index = [FRIENDLY.get(i, i) for i in s.index]
    return s


def risk_card(p):
    if p < 0.35:
        label, color, bg = "LOW RISK", "#1b8a3c", "#e8f6ec"
    elif p < 0.65:
        label, color, bg = "MODERATE RISK", "#b36b00", "#fff4e0"
    else:
        label, color, bg = "HIGH RISK", "#c62828", "#fdeaea"
    pct = p * 100
    st.markdown(f"""
<div style="background:{bg};border-left:8px solid {color};border-radius:12px;padding:22px 26px;">
  <div style="color:{color};font-weight:700;letter-spacing:1px;">{label}</div>
  <div style="color:{color};font-size:56px;font-weight:800;line-height:1.1;">{pct:.1f}%</div>
  <div style="color:#444;">estimated probability of heart disease</div>
  <div style="position:relative;height:14px;margin-top:20px;border-radius:7px;
       background:linear-gradient(90deg,#2e9e4f 0%,#2e9e4f 35%,#f0a020 35%,#f0a020 65%,#d64040 65%,#d64040 100%);">
    <div style="position:absolute;left:calc({pct}% - 3px);top:-5px;width:6px;height:24px;
         background:#222;border-radius:3px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#666;margin-top:8px;">
    <span>0% (low)</span><span>100% (high)</span></div>
</div>""", unsafe_allow_html=True)


def explanation_panel(contrib):
    top = contrib.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.barh(top.index, top.values, color=["#ff4d6d" if v > 0 else "#2ee6c5" for v in top.values])
    ax.axvline(0, color="#9FB0D0", linewidth=0.8)
    ax.set_xlabel("lowers risk  \u2190   \u2192  raises risk", color="#C9D6F2")
    ax.set_title("Why this result? (vs. a typical patient)", color="#EAF2FF")
    ax.tick_params(colors="#C9D6F2")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width="stretch")
    for feat, v in contrib.head(3).items():
        st.write(("\U0001F53A " if v > 0 else "\U0001F53B ") + f"**{feat}** {'raised' if v > 0 else 'lowered'} the risk.")


def personal_tips(d):
    """Lifestyle tips chosen from this patient's own values (no cure / medicine advice)."""
    tips = []
    if d["RestingBP"] >= 130:
        tips.append(("🧂", "Keep blood pressure down",
            f"Resting BP is {d['RestingBP']} mm Hg, above the ideal range. Keep salt under about 5 g a day, "
            "limit packaged and salty snacks, eat more fruit and vegetables, and check your BP regularly."))
    if d["Cholesterol"] >= 200:
        tips.append(("🥗", "Improve your cholesterol",
            f"Cholesterol is {d['Cholesterol']} mg/dl (under 200 is desirable). Eat more oats, beans, nuts and "
            "vegetables, and cut down fried food, red or processed meat and full-fat dairy."))
    if d["FastingBS"] == 1:
        tips.append(("🍬", "Watch your blood sugar",
            "Fasting sugar is above 120 mg/dl. Limit sugary drinks and sweets, choose whole grains over refined "
            "flour, keep a healthy weight and get your sugar checked regularly."))
    if d["MaxHR"] < 0.8 * (220 - d["Age"]):
        tips.append(("🚶", "Build stamina slowly",
            "Your maximum heart rate is lower than typical for your age. Start with gentle walks, add a little "
            "each week towards about 150 minutes, and always warm up and cool down."))
    if d["ExerciseAngina"] == "Y":
        tips.append(("🛑", "Exercise gently",
            "Because exercise brings on chest pain, avoid sudden heavy exertion, stop and rest if you feel chest "
            "discomfort, and ask a doctor before starting any new workout plan."))
    if d["Oldpeak"] >= 2 or d["ST_Slope"] == "Down":
        tips.append(("⏱️", "Pace your activity",
            "Your exercise ECG readings suggest extra strain on the heart during effort. Avoid overexertion and "
            "heavy lifting, and increase activity gradually."))
    if d["ChestPainType"] == "ASY":
        tips.append(("🔎", "Don't wait for symptoms",
            "Many heart problems cause no obvious chest pain, so feeling fine does not mean zero risk. "
            "Keep up regular screening."))
    if d["Age"] >= 45:
        tips.append(("🩺", "Make check-ups a habit",
            "Heart risk rises with age, so check blood pressure, cholesterol and blood sugar at least once a year."))
    if not tips:
        tips.append(("✅", "Keep it up", "Nothing in your values stands out. Keep your healthy habits going."))
    return tips


def next_step(p):
    if p < 0.35:
        return st.success, "Estimated risk is **low**. Keep up healthy habits and have routine check-ups."
    if p < 0.65:
        return st.warning, ("Estimated risk is **moderate**. Consider booking a routine check-up with a doctor "
                            "soon, along with the lifestyle steps below.")
    return st.error, ("Estimated risk is **high**. Please see a doctor soon for a proper check-up. This tool "
                      "cannot diagnose, and lifestyle changes do not replace medical care.")


STYLE = """
<style>
.block-container {max-width: 1100px; padding-top: 1.5rem;}
.hero {background: linear-gradient(120deg,#e53935 0%,#ec407a 45%,#7e57c2 100%); border-radius: 22px;
  padding: 34px 38px; margin-bottom: 8px; box-shadow: 0 10px 30px rgba(229,57,53,.25);}
.hero h1 {color:#fff !important; margin:0 0 8px 0 !important; padding:0 !important; font-size:2.3rem !important;}
.hero p {color:#fff !important; margin:0 0 6px 0 !important; opacity:.93; font-size:1.05rem;}
.hero .chip {display:inline-block; background:rgba(255,255,255,.22); color:#fff; padding:5px 14px;
  border-radius:20px; margin:10px 8px 0 0; font-size:.85rem; font-weight:600;}
.pulse {display:inline-block; animation: beat 1.5s infinite;}
@keyframes beat {0%,100%{transform:scale(1)} 15%{transform:scale(1.22)} 30%{transform:scale(1)} 45%{transform:scale(1.16)}}
.sec-title {font-size:1.4rem; font-weight:800; margin:28px 0 12px 0;}
.grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px;}
.card {border-radius:16px; padding:16px 18px; transition:transform .2s, box-shadow .2s;}
.card:hover {transform:translateY(-4px); box-shadow:0 10px 22px rgba(0,0,0,.14);}
.card .ic {font-size:1.6rem; width:46px; height:46px; display:flex; align-items:center;
  justify-content:center; border-radius:14px; margin-bottom:10px;}
.card .tt {font-weight:700; margin-bottom:4px;}
.card .tx {font-size:.92rem; opacity:.9; line-height:1.45;}
.alert {border-radius:16px; padding:14px 18px; margin-top:14px; border:1px solid rgba(229,57,53,.5);
  background:rgba(229,57,53,.12);}
[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#e53935,#ec407a) !important;
  border:none !important; color:#fff !important; font-weight:700; border-radius:12px; padding:.55rem 1.2rem;}
.card {backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);}
</style>
"""

PALETTE = [(30, 136, 229), (67, 160, 71), (251, 140, 0), (229, 57, 53),
           (126, 87, 194), (0, 172, 193), (236, 64, 122), (121, 85, 72)]


def tip_cards(items):
    """items = [(icon, title, text, (r, g, b))] -> a responsive grid of coloured cards."""
    cards = ""
    for icon, title, text, (r, g, b) in items:
        cards += (f'<div class="card" style="background:rgba({r},{g},{b},.11);border:1px solid rgba({r},{g},{b},.38);">'
                  f'<div class="ic" style="background:rgba({r},{g},{b},.25);">{icon}</div>'
                  f'<div class="tt">{title}</div><div class="tx">{text}</div></div>')
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)


def general_tips():
    section_title("🌍 Healthy habits for everyone")
    tip_cards([
        ("🏃", "Stay active", "Aim for about 150 minutes of brisk walking or similar activity each week.", PALETTE[0]),
        ("🥗", "Eat well", "More fruit, vegetables, whole grains, nuts and fish; less fried, sugary and processed food.", PALETTE[1]),
        ("🧂", "Go easy on salt", "Keep it under about 5 g (one teaspoon) a day.", PALETTE[2]),
        ("🚭", "Avoid tobacco", "Don't smoke, and keep alcohol low.", PALETTE[3]),
        ("😴", "Sleep and calm", "Sleep 7 to 8 hours and manage stress with walks, breathing exercises or hobbies.", PALETTE[4]),
        ("⚖️", "Healthy weight", "Small, steady changes to food and activity keep your weight in a healthy range.", PALETTE[5]),
        ("🩺", "Regular check-ups", "Check blood pressure, cholesterol and blood sugar regularly, even if you feel fine.", PALETTE[6]),
    ])
    st.markdown('<div class="alert">🚨 <b>Get emergency help immediately</b> for chest pain or pressure, '
                'trouble breathing, pain spreading to the arm or jaw, or fainting.</div>', unsafe_allow_html=True)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
AI_SYSTEM = ("You are a friendly health-education assistant inside a student machine-learning project. "
             "Give LIFESTYLE advice only (food, exercise, sleep, stress, habits, monitoring). Do NOT diagnose, "
             "do NOT mention medicines, supplements or doses, never suggest home cures, and never say a doctor "
             "is unnecessary. If risk is moderate or high, remind the person to see a doctor. Use simple "
             "language, about 150 words, 5 short bullet points, and end with one line saying this is "
             "educational and not medical advice.")


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def ask_groq(key, user_msg):
    r = requests.post(
        GROQ_URL, timeout=45,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": get_secret("GROQ_MODEL", "openai/gpt-oss-120b"), "temperature": 0.4,
              "max_tokens": 2000,
              "messages": [{"role": "system", "content": AI_SYSTEM},
                           {"role": "user", "content": user_msg}]})
    if r.status_code != 200:
        raise RuntimeError(f"Groq returned status {r.status_code}. Check your API key and model name.")
    return r.json()["choices"][0]["message"]["content"].strip()


def ai_section(d, p, contrib):
    section_title("🤖 AI-written lifestyle plan")
    key = get_secret("GROQ_API_KEY")
    if not key:
        st.info("Add your free Groq key as GROQ_API_KEY to see an AI-written plan here.")
        return
    # Runs once per result, automatically. The answer is kept in session_state so
    # re-runs of the page do not call the API again.
    if "ai_text" not in st.session_state and "ai_error" not in st.session_state:
        top = ", ".join(contrib.head(3).index)
        msg = (f"Patient: age {d['Age']}, sex {d['Sex']}, resting BP {d['RestingBP']}, cholesterol "
               f"{d['Cholesterol']}, fasting sugar >120: {'yes' if d['FastingBS'] else 'no'}, max heart rate "
               f"{d['MaxHR']}, exercise angina: {d['ExerciseAngina']}. Model-estimated heart disease risk: "
               f"{p * 100:.0f}%. Main factors: {top}. Give lifestyle suggestions.")
        with st.spinner("Writing your personalised plan..."):
            try:
                st.session_state["ai_text"] = ask_groq(key, msg)
            except Exception as e:
                st.session_state["ai_error"] = type(e).__name__
    if "ai_text" in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state["ai_text"])
            st.caption("AI-generated, educational only. Not medical advice.")
    else:
        st.error(f"Could not get AI advice right now ({st.session_state['ai_error']}). The tips above still apply.")
        if st.button("Try again"):
            st.session_state.pop("ai_error", None)
            st.rerun()


# ----------------------------------------------------------------------------
# Look & feel: midnight-navy background with soft rose / teal / violet glows, serif headings
# ----------------------------------------------------------------------------
THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp {background:
  radial-gradient(rgba(255,255,255,.05) 1px, transparent 1.3px) 0 0/26px 26px,
  radial-gradient(1100px 620px at 10% -8%, rgba(229,84,108,.20), transparent 60%),
  radial-gradient(900px 520px at 94% 2%, rgba(92,207,192,.13), transparent 62%),
  radial-gradient(1000px 640px at 50% 118%, rgba(120,92,200,.18), transparent 60%),
  linear-gradient(180deg,#0A0F1E 0%,#0C1226 100%) !important;
  background-attachment: fixed !important; font-family: 'Inter', system-ui, sans-serif;}
[data-testid="stHeader"] {background: transparent !important;}
.block-container {max-width: 1080px; padding-top: 2rem !important;}
.block-container p, .block-container li, .block-container label, .block-container h1, .block-container h2,
.block-container h3, [data-testid="stMarkdownContainer"] {color: #E9EEF9 !important;}
h1, h2, h3, .sec-title {font-family: 'Playfair Display', Georgia, serif !important; letter-spacing: .2px;}
[data-testid="stSidebar"] {background: rgba(10,15,30,.94) !important; border-right: 1px solid rgba(255,255,255,.07);}
[data-testid="stSidebar"] * {color: #E9EEF9 !important;}
[data-testid="stAlert"] {background: rgba(25,35,65,.70) !important;}
[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg,#E5546C,#C93B6B) !important; border: none !important; color: #fff !important;
  font-weight: 600; border-radius: 999px; padding: .6rem 1.6rem; box-shadow: 0 8px 24px rgba(229,84,108,.35);}
.hero {background: linear-gradient(135deg, rgba(255,255,255,.07), rgba(255,255,255,.02)) !important;
  border: 1px solid rgba(255,255,255,.12); box-shadow: none !important;}
.hero h1 {font-family: 'Playfair Display', Georgia, serif !important; font-weight: 700;}
.hero .chip {background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.14);}
.card {border-radius: 18px;}
.w-hero {text-align: center; padding: 52px 20px 18px;}
.w-kicker {letter-spacing: .32em; font-size: .78rem; color: #E6C27A; font-weight: 600; text-transform: uppercase;}
.w-title {font-family: 'Playfair Display', Georgia, serif; font-size: 3.6rem; font-weight: 800; line-height: 1.08;
  margin: 14px 0 12px; color: #fff;}
.w-title em {font-style: italic; background: linear-gradient(90deg,#ff7a93,#E6C27A); -webkit-background-clip: text;
  background-clip: text; color: transparent !important;}
.w-sub {max-width: 680px; margin: 0 auto; font-size: 1.12rem; line-height: 1.65; opacity: .85;}
.ecg {width: min(640px, 90%); height: 80px; display: block; margin: 26px auto 4px;}
.ecg path {fill: none; stroke-linecap: round; stroke-linejoin: round;}
.ecg .base {stroke: rgba(255,255,255,.14); stroke-width: 2;}
.ecg .run {stroke: #ff7a93; stroke-width: 2.6; stroke-dasharray: .16 1.2; animation: ecgrun 3.4s linear infinite;
  filter: drop-shadow(0 0 6px rgba(255,90,120,.9));}
@keyframes ecgrun {from {stroke-dashoffset: 1.36} to {stroke-dashoffset: 0}}
.facts {display: flex; flex-wrap: wrap; justify-content: center; gap: 14px; margin: 22px 0 6px;}
.fact {min-width: 150px; padding: 14px 22px; border-radius: 18px; text-align: center;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10);}
.fact b {display: block; font-family: 'Playfair Display', Georgia, serif; font-size: 1.9rem; color: #E6C27A;}
.fact span {font-size: .82rem; opacity: .75; letter-spacing: .04em;}
.lb-row {display: flex; align-items: center; gap: 14px; padding: 14px 18px; border-radius: 16px; margin-bottom: 10px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); flex-wrap: wrap;}
.lb-row.best {background: linear-gradient(90deg, rgba(230,194,122,.16), rgba(230,194,122,.04));
  border: 1px solid rgba(230,194,122,.55); box-shadow: 0 10px 30px rgba(230,194,122,.10);}
.lb-rank {width: 34px; text-align: center; font-size: 1.3rem;}
.lb-name {min-width: 170px; font-weight: 600; font-size: 1.05rem;}
.lb-bar {flex: 1; min-width: 120px; height: 8px; border-radius: 6px; background: rgba(255,255,255,.08); overflow: hidden;}
.lb-bar span {display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg,#5CCFC0,#E6C27A);}
.lb-num {font-size: .86rem; opacity: .9; min-width: 92px;}
.lb-note {font-size: .92rem; opacity: .8; margin: 4px 0 0;}
.step {display: flex; gap: 14px; align-items: flex-start;}
.step .n {flex: 0 0 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; background: rgba(229,84,108,.18); border: 1px solid rgba(229,84,108,.5); color: #ff9bb0;}
.step .tt {font-weight: 700; margin-bottom: 2px;}
.step .tx {font-size: .92rem; opacity: .85; line-height: 1.45;}
</style>"""

ECG_SVG = ('<svg class="ecg" viewBox="0 0 640 80" preserveAspectRatio="none">'
           '<path class="base" pathLength="1" d="M0 42 H70 L84 42 L92 30 L100 42 H130 L140 46 L150 6 L162 74 L174 42 H214 '
           'L226 38 L240 42 H320 L334 42 L342 30 L350 42 H380 L390 46 L400 6 L412 74 L424 42 H464 L476 38 L490 42 H640"/>'
           '<path class="run" pathLength="1" d="M0 42 H70 L84 42 L92 30 L100 42 H130 L140 46 L150 6 L162 74 L174 42 H214 '
           'L226 38 L240 42 H320 L334 42 L342 30 L350 42 H380 L390 46 L400 6 L412 74 L424 42 H464 L476 38 L490 42 H640"/></svg>')


def leaderboard_html(results):
    rows = ""
    for i, r in results.iterrows():
        best = i == 0
        rows += (f'<div class="lb-row{" best" if best else ""}"><div class="lb-rank">{"🏆" if best else i + 1}</div>'
                 f'<div class="lb-name">{r["Model"]}</div>'
                 f'<div class="lb-bar"><span style="width:{r["CV F1"] * 100:.1f}%"></span></div>'
                 f'<div class="lb-num">CV F1 <b>{r["CV F1"]:.3f}</b></div>'
                 f'<div class="lb-num">Accuracy <b>{r["Accuracy"]:.1%}</b></div>'
                 f'<div class="lb-num">ROC-AUC <b>{r["ROC-AUC"]:.3f}</b></div></div>')
    return rows


def steps_html():
    steps = [("1", "Enter the details", "Age, blood pressure, cholesterol, ECG results and a few more routine measurements."),
             ("2", "The model estimates risk", "The best-performing model returns a risk percentage and a Low / Moderate / High level."),
             ("3", "Understand and act", "See which values drove the result, then get lifestyle suggestions for that patient.")]
    items = "".join(f'<div class="step"><div class="n">{n}</div><div><div class="tt">{t}</div><div class="tx">{x}</div></div></div>'
                    for n, t, x in steps)
    return f'<div class="grid">{items}</div>'


def welcome_page(data):
    res, best = data["results"], data["best"]
    top = res.iloc[0]
    gap = top["CV F1"] - res.iloc[1]["CV F1"]
    close = (f"A close race: the runner-up, {res.iloc[1]['Model']}, is only {gap:.3f} behind, so the ranking could flip on different data."
             if gap < 0.01 else f"It leads the runner-up, {res.iloc[1]['Model']}, by {gap:.3f}.")
    st.markdown(f"""<div class="w-hero"><div class="w-kicker">Statistical Machine Learning Project</div>
<div class="w-title">Heart Disease <em>Risk Predictor</em></div>
<div class="w-sub">A machine-learning tool that estimates a person's chance of heart disease from 11 routine clinical
measurements, explains the result in plain words, and suggests healthy next steps.</div>{ECG_SVG}</div>
<div class="facts"><div class="fact"><b>{data['n_rows']}</b><span>REAL PATIENTS</span></div>
<div class="fact"><b>11</b><span>CLINICAL FEATURES</span></div><div class="fact"><b>{len(res)}</b><span>MODELS COMPARED</span></div>
<div class="fact"><b>{top['Accuracy']:.0%}</b><span>BEST TEST ACCURACY</span></div></div>""", unsafe_allow_html=True)

    section_title("🏆 Today's best model: " + best)
    st.markdown(f'{leaderboard_html(res)}<div class="lb-note">Found live when the app started: every model is trained and tested '
                f'on patients it has never seen, then ranked by 5-fold cross-validated F1 score, which balances catching '
                f'real cases against false alarms, instead of accuracy alone. {close}</div>', unsafe_allow_html=True)

    section_title("About the project")
    tip_cards([
        ("🩺", "The problem", "Heart disease is a leading cause of death and early warning signs are easy to miss. This project shows how routine data can flag risk early.", PALETTE[3]),
        ("🗂️", "The data", f"{data['n_rows']} real patient records (UCI / Kaggle heart dataset). Impossible zero values, like 0 cholesterol, are detected and repaired.", PALETTE[0]),
        ("🧠", "The method", "Logistic Regression, Decision Tree and Random Forest compete. The winner is chosen automatically, never hard-coded.", PALETTE[4]),
        ("🎯", "The outcome", "A risk percentage, a plain-English explanation of what drove it, and lifestyle tips tailored to the patient.", PALETTE[1]),
    ])
    section_title("How it works")
    st.markdown(steps_html(), unsafe_allow_html=True)
    st.write("")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("Begin assessment  →", type="primary"):
            st.session_state["page"] = "predict"
            st.rerun()
    st.caption("Supports UN Sustainable Development Goal 3: Good Health and Well-Being. Educational project, not a medical diagnosis.")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
data = train_models()
best, results, model = data["best"], data["results"], data["models"][data["best"]]
best_row = results.iloc[0]
page = st.session_state.get("page", "welcome")

with st.sidebar:
    st.header("About")
    st.write(f"🏆 **Best model:** {best}")
    st.write(f"**Chosen by:** 5-fold CV F1 ({best_row['CV F1']:.3f})")
    st.write(f"**Test accuracy:** {best_row['Accuracy']:.1%}")
    st.write(f"**Trained on:** {data['n_train']} patients, tested on {data['n_test']}")
    st.caption("Educational project. Not a medical diagnosis.")
    if page != "welcome" and st.button("🏠 Back to home"):
        st.session_state["page"] = "welcome"
        st.rerun()

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(THEME_CSS, unsafe_allow_html=True)

if page == "welcome":
    welcome_page(data)
    st.stop()

st.markdown(f"""<div class="hero">
<h1><span class="pulse">❤️</span> Heart Disease Risk Predictor</h1>
<p>Enter a patient's details to estimate heart disease risk, see why, and get lifestyle suggestions.</p>
<span class="chip">🏆 Best model: {best}</span><span class="chip">👥 {data['n_rows']} real patients</span><span class="chip">🎯 {best_row['Accuracy']:.0%} test accuracy</span>
</div>""", unsafe_allow_html=True)

general_tips()
section_title("🧑‍⚕️ Enter patient details")

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.slider("Age", 1, 120, 54)
    sex = c1.radio("Sex", ["M", "F"], horizontal=True,
                   format_func=lambda s: "Male" if s == "M" else "Female")
    cp = c1.selectbox("Chest pain type", list(CHEST_PAIN), index=3, format_func=CHEST_PAIN.get)
    bp = c1.slider("Resting blood pressure (mm Hg)", 50, 250, 130)
    chol = c1.slider("Cholesterol (mg/dl)", 50, 700, 220)
    fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl?", [0, 1],
                       format_func=lambda v: "Yes" if v else "No")
    ecg = c2.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    hr = c2.slider("Maximum heart rate", 40, 250, 138)
    ang = c2.radio("Exercise-induced angina", ["N", "Y"], horizontal=True,
                   format_func=lambda v: "Yes" if v == "Y" else "No")
    old = c2.slider("Oldpeak (ST depression)", -3.0, 10.0, 0.6, 0.1)
    slope = c2.selectbox("ST slope", ["Up", "Flat", "Down"], index=1)
    submitted = st.form_submit_button("Predict risk", type="primary")

if submitted:
    st.session_state["case"] = {"patient": {
        "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp, "Cholesterol": chol,
        "FastingBS": fbs, "RestingECG": ecg, "MaxHR": hr, "ExerciseAngina": ang,
        "Oldpeak": old, "ST_Slope": slope}}
    st.session_state.pop("ai_text", None)
    st.session_state.pop("ai_error", None)

case = st.session_state.get("case")
if case:
    d = case["patient"]
    patient = pd.DataFrame([d])
    p = float(model.predict_proba(patient)[0][1])
    contrib = explain(model, patient, data["baseline"])

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        risk_card(p)
        st.caption(f"🏆 Prediction made by the best model: **{best}**")
    with right:
        explanation_panel(contrib)

    section_title("🧭 Lifestyle suggestions for this patient")
    show, text = next_step(p)
    show(text)
    tip_cards([(i, t, b, PALETTE[n % len(PALETTE)]) for n, (i, t, b) in enumerate(personal_tips(d))])

    if p >= 0.35 and render_nearby_cardiologists:   # moderate or high risk
        render_nearby_cardiologists(p)

    ai_section(d, p, contrib)
    st.caption("This tool is for learning purposes only and is not a medical diagnosis.")