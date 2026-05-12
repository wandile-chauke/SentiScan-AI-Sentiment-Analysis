"""
dashboard.py
============
PURPOSE : Professional Streamlit web dashboard for the Sentiment Analysis project.

HOW TO RUN:
    streamlit run dashboard.py

WHAT THIS FILE DOES:
  1. Loads the pre-trained model from /models/
  2. Provides a live text-input sentiment predictor with confidence scores
  3. Displays all pre-generated charts and model evaluation metrics
  4. Shows AI-generated insights and a dataset overview
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
# ── No NLTK required — using inline stop-words ────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_PATH  = os.path.join(BASE_DIR, "data", "reviews.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the FIRST Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "SentiScan AI",
    page_icon   = "",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS – dark professional theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0B0E17;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111520;
    border-right: 1px solid #1F2333;
}

/* ── Cards ── */
.metric-card {
    background: #151922;
    border: 1px solid #1F2B40;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

/* ── Sentiment badge ── */
.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
}
.badge-positive { background:#1A3D2B; color:#2ECC71; border:1px solid #2ECC71; }
.badge-negative { background:#3D1A1A; color:#E74C3C; border:1px solid #E74C3C; }
.badge-neutral  { background:#1A2A3D; color:#3498DB; border:1px solid #3498DB; }

/* ── Section headers ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E0E6F0;
    border-left: 4px solid #3498DB;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* ── Insights box ── */
.insight-box {
    background: #111828;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #BDD3EE;
    font-size: 0.93rem;
    line-height: 1.6;
}

/* ── Confidence bar ── */
.conf-bar-wrap { background:#1C2030; border-radius:8px; height:12px; margin:4px 0 12px 0; }
.conf-bar      { height:12px; border-radius:8px; transition: width 0.4s ease; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_STOPS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were be been being have
has had having do does did doing a an the and but if or because as until while
of at by for with about against between into through during before after above
below to from up down in out on off over under again further then once here
there when where why how all both each few more most other some such no nor not
only own same so than too very can will just should now ll m o re ve y ain
aren couldn didn doesn don hadn hasn haven isn mightn mustn needn shan
shouldn wasn weren won wouldn
""".split())

def _simple_lemmatize(word):
    for suffix in ("ness","ment","tion","ing","ers","ed","er","ly","es","s"):
        if word.endswith(suffix) and len(word)-len(suffix) > 3:
            return word[:-len(suffix)]
    return word

def clean_text(text: str) -> str:
    """Mirror of the cleaning function in train_model.py."""
    text   = text.lower()
    text   = re.sub(r"http\S+|www\.\S+", " ", text)
    text   = re.sub(r"<[^>]+>", " ", text)
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = [_simple_lemmatize(w) for w in text.split()
              if w not in _STOPS and len(w) > 2]
    return " ".join(tokens)


@st.cache_resource(show_spinner="Loading model …")
def load_models():
    """
    Load pre-trained artefacts from disk.
    @st.cache_resource means Streamlit loads them ONCE and reuses them —
    otherwise the model would reload on every button click!
    """
    model      = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    le         = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "results.json")) as f:
        results = json.load(f)
    return model, vectorizer, le, results


@st.cache_data
def load_data():
    """Load the review dataset (cached so it reads from disk only once)."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col=0)
    return None


def predict_sentiment(text: str, model, vectorizer, le):
    """
    Run the full prediction pipeline for a single review.

    Returns
    -------
    label       : str    – 'positive', 'negative', or 'neutral'
    confidence  : float  – probability of the predicted class (0–1)
    proba_dict  : dict   – probability for every class
    """
    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    proba     = model.predict_proba(vec)[0]            # array of 3 probabilities
    class_idx = np.argmax(proba)
    label     = le.inverse_transform([class_idx])[0]
    proba_dict = {cls: float(p) for cls, p in zip(le.classes_, proba)}
    return label, float(proba[class_idx]), proba_dict


def emoji_for(sentiment: str) -> str:
    return {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")

def color_for(sentiment: str) -> str:
    return {"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#3498DB"}.get(sentiment, "#AAAAAA")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## SentiScan AI")
    st.markdown("*Sentiment Analysis Dashboard*")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio(
        "",
        ["🏠 Home & Predictor", "📊 Model Performance", "📈 Data Insights", "💡 About the Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit")
    st.caption("Week 3 — AI for Data Analysis & Insights")


# ══════════════════════════════════════════════════════════════════════════════
# GUARD: check model files exist
# ══════════════════════════════════════════════════════════════════════════════

model_files = ["sentiment_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl", "results.json"]
missing     = [f for f in model_files if not os.path.exists(os.path.join(MODEL_DIR, f))]

if missing:
    st.error("⚠️ Model not trained yet!")
    st.info("Run the following command in your terminal first:")
    st.code("python train_model.py", language="bash")
    st.stop()

# ── Load resources ─────────────────────────────────────────────────────────
model, vectorizer, le, results = load_models()
df = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

if "Home" in page:
    # --- Hero header ---
    st.markdown("""
    <div style='text-align:center; padding: 32px 0 24px 0;'>
        <h1 style='font-size:2.8rem; font-weight:700; color:#E0E6F0; letter-spacing:-1px;'>
            🔍 SentiScan <span style='color:#3498DB;'>AI</span>
        </h1>
        <p style='color:#7A8BA0; font-size:1.1rem; margin-top:-8px;'>
            Analyse the sentiment behind any text — powered by Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- KPI row ---
    acc     = results["accuracy"]
    n_total = len(df) if df is not None else "—"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, colour in [
        (c1, "Model Accuracy",    f"{acc*100:.1f}%",  "#2ECC71"),
        (c2, "Reviews Analysed",  f"{n_total:,}",     "#3498DB"),
        (c3, "Sentiment Classes", "3",                "#F39C12"),
        (c4, "Algorithm",         "Logistic\nReg.",   "#9B59B6"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <p style='color:#7A8BA0; font-size:0.8rem; margin:0; text-transform:uppercase; letter-spacing:1px;'>{label}</p>
            <p style='color:{colour}; font-size:1.9rem; font-weight:700; margin:4px 0 0 0;'>{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Predictor ---
    st.markdown("<div class='section-title'>✍️ Live Sentiment Predictor</div>", unsafe_allow_html=True)
    st.markdown("Type or paste any review below and the AI will predict its sentiment.")

    # Quick-fill example buttons
    st.markdown("**Try an example:**")
    ex1, ex2, ex3, _ = st.columns([1,1,1,3])
    ex_text = ""
    if ex1.button("😊 Positive"):
        ex_text = "This movie was absolutely amazing! The storyline was brilliant and the acting was superb."
    if ex2.button("😞 Negative"):
        ex_text = "Terrible film. Boring and predictable. A complete waste of time."
    if ex3.button("😐 Neutral"):
        ex_text = "The movie was okay. Not great but not terrible either. Average experience."

    # Text area
    user_input = st.text_area(
        "",
        value=ex_text,
        height=130,
        placeholder="e.g. 'This product exceeded all my expectations! Absolutely love it.'",
        label_visibility="collapsed",
    )

    analyse_btn = st.button("🔍  Analyse Sentiment", type="primary", use_container_width=False)

    if analyse_btn:
        if not user_input.strip():
            st.warning("Please enter some text before analysing.")
        else:
            label, confidence, proba_dict = predict_sentiment(user_input, model, vectorizer, le)
            em  = emoji_for(label)
            col = color_for(label)

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1])

            with r1:
                st.markdown(f"""
                <div class='metric-card' style='text-align:left;'>
                    <p style='color:#7A8BA0; font-size:0.78rem; margin:0; text-transform:uppercase; letter-spacing:1px;'>Detected Sentiment</p>
                    <div style='margin-top:10px;'>
                        <span class='badge badge-{label}'>{em} {label.upper()}</span>
                    </div>
                    <p style='color:#7A8BA0; font-size:0.88rem; margin-top:14px;'>Confidence Score</p>
                    <p style='color:{col}; font-size:2rem; font-weight:700; margin:0;'>{confidence*100:.1f}%</p>
                    <div class='conf-bar-wrap'>
                        <div class='conf-bar' style='width:{confidence*100:.1f}%; background:{col};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown("<div class='metric-card' style='text-align:left;'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#7A8BA0; font-size:0.78rem; margin:0 0 10px 0; text-transform:uppercase; letter-spacing:1px;'>Class Probabilities</p>", unsafe_allow_html=True)

                fig_prob = go.Figure()
                for cls, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
                    fig_prob.add_trace(go.Bar(
                        x=[prob],
                        y=[cls.capitalize()],
                        orientation="h",
                        marker_color=color_for(cls),
                        text=f"{prob*100:.1f}%",
                        textposition="outside",
                        textfont=dict(color="#E0E6F0", size=13),
                        name=cls,
                    ))
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(range=[0, 1.15], visible=False),
                    yaxis=dict(color="#E0E6F0"),
                    margin=dict(l=10, r=30, t=10, b=10),
                    height=160,
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Interpretation
            interp = {
                "positive": "✅ The text expresses a **positive** opinion — satisfaction, praise, or enjoyment.",
                "negative": "❌ The text expresses a **negative** opinion — dissatisfaction, criticism, or disappointment.",
                "neutral":  "➖ The text is **neutral** — balanced, factual, or lacking strong emotion.",
            }
            st.info(interp[label])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif "Performance" in page:
    st.markdown("## 📊 Model Performance")
    st.markdown("How well does our Logistic Regression model perform on unseen reviews?")

    acc = results["accuracy"]
    report = results["classification_report"]
    classes = results["classes"]

    # --- Accuracy hero ---
    st.markdown(f"""
    <div style='text-align:center; padding:20px; background:#151922; border-radius:14px;
                border:1px solid #1F2B40; margin-bottom:24px;'>
        <p style='color:#7A8BA0; font-size:0.9rem; margin:0;'>OVERALL ACCURACY</p>
        <p style='color:#2ECC71; font-size:4rem; font-weight:700; margin:0;'>{acc*100:.1f}%</p>
        <p style='color:#7A8BA0; font-size:0.85rem;'>Tested on 20% hold-out data</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Per-class metrics ---
    st.markdown("<div class='section-title'>Per-Class Metrics</div>", unsafe_allow_html=True)
    metric_cols = st.columns(3)
    for col, cls in zip(metric_cols, classes):
        p = report[cls]["precision"]
        r = report[cls]["recall"]
        f = report[cls]["f1-score"]
        s = int(report[cls]["support"])
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-top: 3px solid {color_for(cls)};'>
                <h3 style='color:{color_for(cls)}; margin:0 0 14px 0;'>{emoji_for(cls)} {cls.capitalize()}</h3>
                <table style='width:100%; font-size:0.9rem; color:#BDD3EE;'>
                    <tr><td>Precision</td><td style='text-align:right; font-weight:600;'>{p:.3f}</td></tr>
                    <tr><td>Recall</td>   <td style='text-align:right; font-weight:600;'>{r:.3f}</td></tr>
                    <tr><td>F1-Score</td> <td style='text-align:right; font-weight:600;'>{f:.3f}</td></tr>
                    <tr><td>Support</td>  <td style='text-align:right; font-weight:600;'>{s}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts ---
    img_cm = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    img_mm = os.path.join(ASSETS_DIR, "model_metrics.png")
    img_tw = os.path.join(ASSETS_DIR, "top_words.png")

    if os.path.exists(img_cm):
        ch1, ch2 = st.columns(2)
        ch1.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        ch1.image(img_cm, use_container_width=True)
        if os.path.exists(img_mm):
            ch2.markdown("<div class='section-title'>Precision / Recall / F1</div>", unsafe_allow_html=True)
            ch2.image(img_mm, use_container_width=True)

    if os.path.exists(img_tw):
        st.markdown("<div class='section-title'>Most Influential Words</div>", unsafe_allow_html=True)
        st.image(img_tw, use_container_width=True)

    # --- What these metrics mean ---
    st.markdown("<div class='section-title'>📚 Understanding the Metrics</div>", unsafe_allow_html=True)
    with st.expander("Click to learn what each metric means"):
        st.markdown("""
| Metric | What it measures | Ideal value |
|--------|-----------------|-------------|
| **Accuracy** | % of all predictions that are correct | As high as possible |
| **Precision** | Of predicted positives, how many are truly positive? | High = fewer false alarms |
| **Recall** | Of all actual positives, how many did we catch? | High = fewer misses |
| **F1-Score** | Balance between Precision and Recall | High = both are good |
| **Confusion Matrix** | Shows exactly which class gets confused with which | Diagonal = correct |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif "Insights" in page:
    st.markdown("## 📈 Data Insights")

    if df is None:
        st.warning("Dataset not found. Run `python train_model.py` first.")
        st.stop()

    # --- Stats row ---
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, colour in [
        (c1, "Total Reviews",     f"{len(df):,}",              "#3498DB"),
        (c2, "Avg Word Count",    f"{df['word_count'].mean():.0f}", "#F39C12"),
        (c3, "Positive Reviews",  f"{(df['sentiment']=='positive').sum():,}", "#2ECC71"),
        (c4, "Negative Reviews",  f"{(df['sentiment']=='negative').sum():,}", "#E74C3C"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <p style='color:#7A8BA0; font-size:0.78rem; margin:0; text-transform:uppercase;'>{label}</p>
            <p style='color:{colour}; font-size:1.9rem; font-weight:700; margin:4px 0 0 0;'>{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts from assets ---
    for img_file, title in [
        ("sentiment_distribution.png", "Sentiment Distribution"),
        ("word_count_distribution.png", "Word Count by Sentiment"),
    ]:
        path = os.path.join(ASSETS_DIR, img_file)
        if os.path.exists(path):
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.image(path, use_container_width=True)

    # --- Interactive Plotly chart ---
    st.markdown("<div class='section-title'>Interactive: Sentiment Breakdown</div>", unsafe_allow_html=True)
    counts  = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    colours = [color_for(s) for s in counts["Sentiment"]]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#3498DB"},
                 text="Count")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#151922",
        font_color="#E0E6F0", showlegend=False,
        xaxis=dict(gridcolor="#1F2333"), yaxis=dict(gridcolor="#1F2333"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Sample data ---
    st.markdown("<div class='section-title'>Sample Reviews</div>", unsafe_allow_html=True)
    sentiment_filter = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral"])
    sample = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]
    st.dataframe(sample[["review", "sentiment", "word_count"]].head(10),
                 use_container_width=True, height=300)

    # --- AI-Generated Insights ---
    st.markdown("<div class='section-title'>💡 AI-Generated Insights</div>", unsafe_allow_html=True)
    acc = results["accuracy"]
    n_pos = (df["sentiment"] == "positive").sum()
    n_neg = (df["sentiment"] == "negative").sum()
    n_neu = (df["sentiment"] == "neutral").sum()
    pct_pos = n_pos / len(df) * 100

    insights = [
        f"📌 The dataset is **well-balanced** across three sentiment classes, which helps the model learn each class equally well without bias.",
        f"📌 The model achieved **{acc*100:.1f}% accuracy**, meaning it correctly identifies the sentiment of approximately {acc*100:.0f} out of every 100 reviews.",
        f"📌 **{pct_pos:.0f}% of reviews are positive**, suggesting that when customers/viewers leave feedback, they lean slightly towards positive experiences.",
        f"📌 **Logistic Regression** with TF-IDF is an excellent baseline for text classification — fast to train, interpretable, and achieves competitive accuracy even on small datasets.",
        f"📌 **Neutral reviews** are typically the hardest to classify correctly because they share vocabulary with both positive and negative sentiment.",
        f"📌 **Bigrams** (two-word phrases like 'not good' or 'absolutely love') significantly improve accuracy because single words miss the context of negation.",
    ]
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif "About" in page:
    st.markdown("## 💡 About This Project")

    st.markdown("""
    <div class='insight-box'>
    <strong>SentiScan AI</strong> is a machine-learning-powered sentiment analysis tool built as
    a Week 3 portfolio project for the <em>AI for Data Analysis &amp; Insights</em> course module.
    It demonstrates the full ML pipeline from raw text data to a deployed interactive web application.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(3)
    techs = [
        ("Python 3.11", "Core programming language",                "#F39C12"),
        ("Pandas",      "Data loading & manipulation",              "#3498DB"),
        ("scikit-learn","TF-IDF vectorisation & Logistic Regression","#E74C3C"),
        ("NLTK",        "Text cleaning & lemmatisation",            "#9B59B6"),
        ("Matplotlib",  "Static chart generation",                  "#2ECC71"),
        ("Streamlit",   "Interactive web dashboard",                "#FF4B4B"),
        ("Plotly",      "Interactive charts in dashboard",          "#00CC96"),
        ("joblib",      "Model serialisation",                      "#7F8C8D"),
        ("NumPy",       "Numerical computations",                   "#4E9AF1"),
    ]
    for i, (name, desc, colour) in enumerate(techs):
        col = tech_cols[i % 3]
        col.markdown(f"""
        <div class='metric-card' style='text-align:left; margin-bottom:10px;'>
            <p style='color:{colour}; font-weight:700; margin:0;'>{name}</p>
            <p style='color:#7A8BA0; font-size:0.82rem; margin:2px 0 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔄 ML Pipeline Steps</div>", unsafe_allow_html=True)
    steps = [
        ("1. Data Loading",       "Import CSV reviews with Pandas"),
        ("2. Text Cleaning",      "Lower-case, remove HTML/URLs, strip punctuation, remove stop-words, lemmatise"),
        ("3. Train/Test Split",   "80% training data, 20% held-out test data (stratified)"),
        ("4. TF-IDF Vectorisation","Convert text to numbers — 5,000 features including bigrams"),
        ("5. Model Training",     "Logistic Regression with multinomial strategy and L2 regularisation"),
        ("6. Evaluation",         "Accuracy, precision, recall, F1-score, confusion matrix"),
        ("7. Visualisation",      "5 charts saved to /assets/"),
        ("8. Deployment",         "Streamlit dashboard with live prediction and interactive charts"),
    ]
    for step, desc in steps:
        st.markdown(f"""
        <div class='insight-box'>
            <strong style='color:#3498DB;'>{step}</strong><br>
            {desc}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📚 Key Concepts</div>", unsafe_allow_html=True)
    with st.expander("What is TF-IDF?"):
        st.markdown("""
**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.

- **TF** measures how often a word appears in a single document.
- **IDF** measures how rare that word is across all documents.
- **TF-IDF = TF × IDF** gives a high score to words that are frequent in one document but rare overall — these are the most *informative* words.

**Example:** The word "the" appears everywhere — low TF-IDF. The word "masterpiece" appears only in very positive reviews — high TF-IDF for positive class.
        """)

    with st.expander("Why Logistic Regression?"):
        st.markdown("""
**Logistic Regression** is a linear classifier that:
1. Assigns a weight to every word in the vocabulary.
2. Multiplies each word's TF-IDF score by its weight.
3. Sums all scores to get a raw value per class.
4. Applies the **softmax** function to convert raw values into probabilities.
5. Picks the class with the highest probability as the prediction.

**Why is it great for text?**
- Fast (trains in seconds on thousands of reviews)
- Interpretable (we can see which words push towards positive/negative)
- Works very well with sparse TF-IDF matrices
- Outputs calibrated probabilities (confidence scores)
        """)

    with st.expander("How does Sentiment Analysis work?"):
        st.markdown("""
**Sentiment Analysis** is a type of **text classification** task.

**The process:**
1. **Data** — Labelled examples: reviews tagged as positive, negative, neutral.
2. **Feature Extraction** — Convert raw text into numbers (TF-IDF).
3. **Training** — The model finds patterns: "amazing" → positive, "terrible" → negative.
4. **Prediction** — Given new unseen text, apply learned patterns to predict sentiment.
5. **Confidence** — Output a probability (how sure is the model?).

**Real-world applications:** Product reviews, social media monitoring, customer support, brand reputation tracking, political opinion mining.
        """)
