# Portfolio Documentation: SentiScan AI

## Short Technical Summary (for LinkedIn / Portfolio site)

> Built a complete end-to-end sentiment analysis system in Python that classifies
> text reviews as positive, negative, or neutral using TF-IDF vectorisation and
> Logistic Regression. Deployed as an interactive Streamlit web dashboard with
> live prediction, confidence scores, and data visualisations.

---

## Project Summary (3-Sentence Version)

SentiScan AI is a machine learning web application that automatically determines
the emotional tone of customer or movie reviews. It uses TF-IDF to convert raw text
into numerical features and Logistic Regression to classify sentiment with high accuracy.
The project includes a full data pipeline, five professional visualisations, model
evaluation metrics, and a Streamlit dashboard for live predictions.

---

## Skills Demonstrated

### Technical Skills
| Skill | How Demonstrated |
|-------|-----------------|
| Python Programming | 3 complete Python modules with clean, documented code |
| Data Manipulation | Pandas: loading, cleaning, filtering, aggregating review data |
| Natural Language Processing | Text preprocessing: lower-casing, stop-word removal, lemmatisation |
| Machine Learning | Full scikit-learn pipeline: split → vectorise → train → evaluate |
| Feature Engineering | TF-IDF with unigrams and bigrams, min_df filtering |
| Model Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |
| Data Visualisation | 5 charts: bar, pie, histogram, heatmap, grouped bar (matplotlib + seaborn + plotly) |
| Web Development | Streamlit dashboard with custom CSS, responsive layout, caching |
| Software Engineering | Modular code structure, separation of concerns, model serialisation |

### Soft Skills
- **Problem-solving** — Identified the challenge (manual review analysis) and built a scalable solution
- **Communication** — Code is fully documented; report explains every decision
- **Project ownership** — Completed independently from data to deployment
- **Analytical thinking** — Interpreted model metrics and generated insights

---

## Learning Outcomes

By completing this project, I learned to:

1. **Design an ML pipeline** — Understand the full workflow from raw data to deployed model
2. **Clean and preprocess text** — Apply NLP techniques to prepare data for machine learning
3. **Apply TF-IDF** — Convert text into numerical features that capture word importance
4. **Train and evaluate classifiers** — Use Logistic Regression and interpret accuracy, precision, recall, F1
5. **Interpret model behaviour** — Use coefficient weights to understand which words drive predictions
6. **Build a professional dashboard** — Design and deploy a Streamlit web app with custom styling
7. **Structure a data science project** — Organise code into modules, models, assets, and reports
8. **Document technical work** — Write code comments, README, and an insights report

---

## How to Present to Your Lecturer

### Opening (30 seconds)
> "My project, SentiScan AI, is a complete machine learning solution for sentiment analysis.
> It takes any text review and classifies it as positive, negative, or neutral.
> I built everything from scratch — data generation, model training, evaluation, and a web dashboard."

### Demo Walkthrough (2 minutes)
1. Open the dashboard → show the Home page with the KPI cards
2. Type a strongly positive review → show the prediction + confidence score
3. Type a negative review → show the class probabilities bar chart
4. Navigate to Model Performance → explain the confusion matrix
5. Navigate to Data Insights → show the distribution chart + AI insights
6. Navigate to About → explain the ML pipeline steps

### Technical Explanation (1 minute)
> "The pipeline works in 4 steps:
> First, I clean the raw text — removing HTML, stop-words, and normalising words.
> Second, I convert text to numbers using TF-IDF, which gives higher scores to
> rare but informative words. Third, I train a Logistic Regression classifier on
> 80% of the data. Finally, I evaluate on the held-out 20% and get strong accuracy."

### Closing (15 seconds)
> "The project is production-ready: the model is saved to disk, the dashboard
> loads it instantly, and the code is fully documented for anyone to run.
> I've also written an insights report and a professional README for my GitHub."

---

## Questions You Might Be Asked

**Q: Why Logistic Regression instead of a neural network?**
> "Logistic Regression is the industry-standard baseline for text classification.
> It's fast, interpretable, and achieves competitive accuracy with TF-IDF features.
> A neural network would be the next step, but adds complexity without necessarily
> improving accuracy on a dataset of this size."

**Q: What is TF-IDF?**
> "TF-IDF stands for Term Frequency – Inverse Document Frequency. It scores words
> based on how often they appear in one document versus how common they are across
> all documents. Words like 'amazing' that appear frequently in positive reviews
> but rarely overall get a high score. Common words like 'the' get a low score."

**Q: How would you improve this in the future?**
> "I'd replace the synthetic dataset with a real one like IMDB or Amazon reviews,
> and explore transformer-based models like BERT for higher accuracy on ambiguous
> or sarcastic text. I'd also add a CSV upload feature and deploy to Streamlit Cloud."

**Q: How do you know the model is good?**
> "I evaluate it on 20% of data the model has never seen during training.
> This gives an unbiased estimate of real-world performance.
> The confusion matrix shows exactly which classes get confused with each other."
