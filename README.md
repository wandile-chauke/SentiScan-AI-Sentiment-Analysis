# 🔍 SentiScan AI — Sentiment Analysis Dashboard

> A complete machine-learning-powered sentiment analysis tool that classifies customer and movie reviews as **Positive**, **Negative**, or **Neutral** using TF-IDF vectorisation and Logistic Regression.

---

## 📸 Project Overview

**SentiScan AI** is a portfolio-ready, end-to-end sentiment analysis project that covers the complete ML pipeline:

```
Raw Text → Data Cleaning → TF-IDF Vectorisation → Model Training → Evaluation → Web Dashboard
```

Built as a Week 3 individual project for the *AI for Data Analysis & Insights* module.

---

## ✨ Features

- ✅ **Live Sentiment Predictor** — type any review and get instant sentiment + confidence score
- ✅ **Full ML Pipeline** — data loading, cleaning, vectorisation, training, evaluation
- ✅ **5 Professional Visualisations** — distribution charts, confusion matrix, top words, metrics
- ✅ **Interactive Dashboard** — built with Streamlit + Plotly
- ✅ **AI-Generated Insights** — automated observations from the data
- ✅ **Beginner-Friendly Code** — every function is documented with explanations
- ✅ **Portfolio-Ready** — professional structure, clean code, full documentation

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Core language |
| Pandas | 2.1.4 | Data manipulation |
| scikit-learn | 1.4.0 | ML: TF-IDF + Logistic Regression |
| NLTK | 3.8.1 | Text cleaning & lemmatisation |
| Matplotlib | 3.8.2 | Static chart generation |
| Seaborn | 0.13.2 | Enhanced chart styling |
| Streamlit | 1.31.0 | Web dashboard framework |
| Plotly | 5.18.0 | Interactive charts |
| joblib | 1.3.2 | Model saving/loading |

---

## 📂 Project Structure

```
sentiment_analyzer/
│
├── dashboard.py          # Streamlit web dashboard (MAIN APP)
├── train_model.py        # ML pipeline: clean → train → evaluate → save
├── generate_dataset.py   # Synthetic review dataset generator
├── requirements.txt      # All Python dependencies
├── README.md             # This file
│
├── data/
│   └── reviews.csv       # Generated on first run
│
├── models/               # Saved after running train_model.py
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── results.json
│
├── assets/               # Generated charts
│   ├── sentiment_distribution.png
│   ├── confusion_matrix.png
│   ├── top_words.png
│   ├── word_count_distribution.png
│   └── model_metrics.png
│
└── reports/
    └── insights_report.md
```

---

## 🚀 Installation & Setup

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/YOUR_USERNAME/sentiscan-ai.git
cd sentiscan-ai
```

### Step 2 — Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Mac / Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model

```bash
python train_model.py
```

This will:
- Generate the dataset
- Clean the text
- Train the model
- Save all artefacts to `/models/`
- Generate all charts to `/assets/`

Expected output:
```
✅ PIPELINE COMPLETE!
  Accuracy : 94.4%
  Charts   : /assets/
  Models   : /models/
```

### Step 5 — Launch the Dashboard

```bash
streamlit run dashboard.py
```

Then open your browser at: **http://localhost:8501**

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | ~94% |
| Positive F1 | ~0.95 |
| Negative F1 | ~0.95 |
| Neutral F1  | ~0.92 |

---

## 🧠 How It Works

### 1. Text Cleaning
Every review goes through a 7-step pipeline:
```
lower-case → remove URLs → remove HTML → keep letters only → 
tokenise → remove stop-words → lemmatise → rejoin
```

### 2. TF-IDF Vectorisation
Converts text into a matrix of numbers where:
- **TF** = how often a word appears in this review
- **IDF** = how rare the word is across all reviews
- High score = word is frequent here AND rare overall → very informative

### 3. Logistic Regression
- Assigns weights to every word feature
- Applies softmax to produce class probabilities
- Picks the class with the highest probability

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home & Predictor | Live sentiment prediction with confidence score |
| 📊 Model Performance | Confusion matrix, per-class metrics, top words |
| 📈 Data Insights | Distribution charts, word counts, interactive graphs |
| 💡 About | Project explanation, tech stack, key concepts |

---

## 📸 Screenshots

*(Run the app and take screenshots for your portfolio)*

- Dashboard home with live predictor
- Sentiment distribution pie chart
- Confusion matrix heatmap
- Top influential words

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Data preprocessing** — cleaning and normalising raw text
- **Feature engineering** — TF-IDF vectorisation
- **Machine learning** — training and evaluating a classifier
- **Model evaluation** — accuracy, precision, recall, F1, confusion matrix
- **Data visualisation** — matplotlib, seaborn, plotly charts
- **Web deployment** — building an interactive Streamlit app
- **Software engineering** — project structure, modular code, documentation

---

## 🔭 Future Improvements

- [ ] Add BERT/transformer-based model for higher accuracy
- [ ] Connect to Twitter/X API for real-time social media analysis
- [ ] Add word cloud visualisation
- [ ] Support CSV file upload for batch analysis
- [ ] Add export-to-PDF report feature
- [ ] Deploy to Streamlit Cloud for public access

---

## 👤 Author

**[Your Name]**  
Week 3 — AI for Data Analysis & Insights  
[Your Institution / Programme]

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
