"""
train_model.py
==============
PURPOSE : Load data → clean text → vectorize with TF-IDF →
          train Logistic Regression → evaluate → save model artefacts.
"""

import os, re, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "reviews.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

PALETTE = {
    "positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#3498DB",
    "bg": "#0F1117", "card": "#1E2130", "text": "#EAEAEA",
}

# ── Minimal English stop-words (no NLTK needed) ────────────────────────────
STOP_WORDS = set("""
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


def simple_lemmatize(word: str) -> str:
    """Minimal rule-based lemmatisation (no NLTK required)."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    for suffix in ("ness", "ment", "tion", "ing", "ies", "ers", "ed", "er", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    return word


def clean_text(text: str) -> str:
    """
    Transform raw text into clean, normalised tokens.

    Pipeline:
      1. Lower-case
      2. Remove URLs
      3. Remove HTML tags
      4. Keep letters only
      5. Tokenise
      6. Remove stop-words
      7. Lemmatise
      8. Rejoin
    """
    text   = text.lower()
    text   = re.sub(r"http\S+|www\.\S+", " ", text)
    text   = re.sub(r"<[^>]+>", " ", text)
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = [simple_lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path):
    print("\n📂 STEP 1 — Loading data …")
    if not os.path.exists(path):
        print("   Dataset not found. Generating …")
        from generate_dataset import generate_dataset
        df = generate_dataset(900)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=True)
    df = pd.read_csv(path, index_col=0)
    print(f"   Loaded {len(df):,} rows\n{df['sentiment'].value_counts().to_string()}\n")
    return df


def preprocess(df):
    print("🧹 STEP 2 — Cleaning text …")
    df = df.copy()
    df["clean_review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["clean_review"].str.strip() != ""].reset_index(drop=True)
    print(f"   ✅ {len(df):,} clean reviews\n")
    return df


def split_data(df):
    print("✂️  STEP 3 — Train/Test split (80/20) …")
    le = LabelEncoder()
    y  = le.fit_transform(df["sentiment"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}\n")
    return X_train, X_test, y_train, y_test, le


def build_tfidf(X_train, X_test):
    print("🔢 STEP 4 — TF-IDF Vectorisation …")
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"   Vocab: {len(vec.vocabulary_):,} | Train: {Xtr.shape} | Test: {Xte.shape}\n")
    return vec, Xtr, Xte


def train_model(Xtr, ytr):
    print("🤖 STEP 5 — Training Logistic Regression …")
    m = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                           random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    print("   ✅ Done\n")
    return m


def evaluate(model, le, Xte, yte):
    print("📊 STEP 6 — Evaluating …")
    yp  = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    rep = classification_report(yte, yp, target_names=le.classes_, output_dict=True)
    cm  = confusion_matrix(yte, yp)
    print(f"   Accuracy: {acc*100:.1f}%")
    print(classification_report(yte, yp, target_names=le.classes_))
    return {"accuracy": acc, "classification_report": rep,
            "confusion_matrix": cm.tolist(), "classes": list(le.classes_)}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    p = os.path.join(ASSETS_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"   💾 {name}")

def plot_distribution(df):
    print("\n📈 Charts …")
    counts = df["sentiment"].value_counts()
    colors = [PALETTE[s] for s in counts.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Sentiment Distribution", color=PALETTE["text"], fontsize=15, fontweight="bold")
    # Bar
    bars = a1.bar(counts.index, counts.values, color=colors, width=0.5)
    a1.set_facecolor(PALETTE["card"]); a1.tick_params(colors=PALETTE["text"])
    a1.set_title("Review Count", color=PALETTE["text"])
    a1.spines[["top","right","left","bottom"]].set_visible(False)
    for b, v in zip(bars, counts.values):
        a1.text(b.get_x()+b.get_width()/2, b.get_height()+3, str(v),
                ha="center", color=PALETTE["text"], fontsize=11)
    # Pie
    a2.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
           textprops={"color": PALETTE["text"]})
    a2.set_facecolor(PALETTE["bg"]); a2.set_title("Share", color=PALETTE["text"])
    fig.patch.set_facecolor(PALETTE["bg"])
    _save(fig, "sentiment_distribution.png")

def plot_cm(results):
    cm = np.array(results["confusion_matrix"])
    classes = results["classes"]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=1, linecolor=PALETTE["bg"],
                annot_kws={"size": 14, "weight": "bold"}, ax=ax)
    ax.set_title("Confusion Matrix", color=PALETTE["text"], fontsize=13)
    ax.set_xlabel("Predicted", color=PALETTE["text"])
    ax.set_ylabel("True", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"]); ax.set_facecolor(PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])
    _save(fig, "confusion_matrix.png")

def plot_top_words(model, vectorizer, le):
    fn = np.array(vectorizer.get_feature_names_out())
    classes = list(le.classes_)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Most Influential Words", color=PALETTE["text"], fontsize=13)
    for ax, cls, i in zip(axes, classes, range(3)):
        idx = np.argsort(model.coef_[i])[-15:]
        ax.barh(fn[idx], model.coef_[i][idx], color=PALETTE[cls])
        ax.set_facecolor(PALETTE["card"]); ax.tick_params(colors=PALETTE["text"])
        ax.set_title(cls.capitalize(), color=PALETTE[cls], fontsize=11)
        ax.spines[["top","right","bottom"]].set_visible(False)
    fig.patch.set_facecolor(PALETTE["bg"])
    _save(fig, "top_words.png")

def plot_word_counts(df):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    for s in ["positive","negative","neutral"]:
        ax.hist(df[df["sentiment"]==s]["word_count"], bins=20,
                alpha=0.65, label=s.capitalize(), color=PALETTE[s])
    ax.set_title("Word Count by Sentiment", color=PALETTE["text"])
    ax.set_xlabel("Word Count", color=PALETTE["text"])
    ax.set_ylabel("Frequency", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor(PALETTE["bg"])
    _save(fig, "word_count_distribution.png")

def plot_metrics(results):
    report = results["classification_report"]; classes = results["classes"]
    metrics = ["precision","recall","f1-score"]
    x = np.arange(len(metrics)); w = 0.22
    fig, ax = plt.subplots(figsize=(9,5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    for i, cls in enumerate(classes):
        ax.bar(x+i*w, [report[cls][m] for m in metrics], width=w,
               label=cls.capitalize(), color=PALETTE[cls])
    ax.set_xticks(x+w); ax.set_xticklabels(metrics, color=PALETTE["text"])
    ax.set_ylim(0, 1.15); ax.set_title("Model Metrics per Class", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor(PALETTE["bg"])
    _save(fig, "model_metrics.png")


def save_artifacts(model, vectorizer, le, results):
    print("\n💾 Saving artefacts …")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model,      os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(le,         os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump({k:v for k,v in results.items()}, f, indent=2)
    print("   ✅ All artefacts saved")


def main():
    print("="*60)
    print("  SENTIMENT ANALYSIS — MODEL TRAINING PIPELINE")
    print("="*60)
    df        = load_data(DATA_PATH)
    df        = preprocess(df)
    Xtr,Xte,ytr,yte,le = split_data(df)
    vec,Xtrv,Xtev       = build_tfidf(Xtr, Xte)
    model     = train_model(Xtrv, ytr)
    results   = evaluate(model, le, Xtev, yte)
    plot_distribution(df)
    plot_cm(results)
    plot_top_words(model, vec, le)
    plot_word_counts(df)
    plot_metrics(results)
    save_artifacts(model, vec, le, results)
    print("\n" + "="*60)
    print(f"  🎉 COMPLETE! Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  ▶  streamlit run dashboard.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
