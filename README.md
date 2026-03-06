# Hybrid Review Authenticity Detection System

This project implements a Hybrid Review Authenticity Detection System designed for e-commerce platforms to detect statistically unusual product reviews.

---

## Live Demo

Try the deployed application here:

https://hybrid-detection-system-3z55rnheuv.streamlit.app/

---

## Project Overview 

**The system analyses multiple signals including:**

- rating–sentiment consistency

- review length patterns

- emotional exaggeration

- reviewer activity behaviour

**Using these signals, the system assigns an anomaly score to each review and classifies it as:**

- Normal Review

- Anomalous Review

**The system mimics moderation systems used in real e-commerce platforms where suspicious reviews are flagged before being published.**

---

## Key Features

- Hybrid detection architecture

- Rule-based anomaly scoring

- Unsupervised machine learning model

- Sentiment vs Rating Consistency Analysis

- Detects mismatch between review text sentiment and star rating

- Behavioural Signals

- Reviewer activity patterns

- Review length deviation

- Explainable Decisions

- Provides reasons for flagged reviews

- Interactive Web Interface uilt using Streamlit

- Visualization

- Displays anomaly score distribution across the dataset

---

## Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- VADER Sentiment Analysis

- Streamlit

- Docker

---

## System Architecture section

The system follows a hybrid pipeline combining rule-based anomaly scoring with machine learning based anomaly detection.

User Review Input
        │
        ▼
Feature Engineering
(sentiment analysis, review length,
rating normalization, reviewer behavior)
        │
        ▼
Anomaly Scoring System
(rule-based anomaly signals)
        │
        ▼
Isolation Forest Model
(unsupervised anomaly detection)
        │
        ▼
Hybrid Decision Engine
(combines rule-based score + ML prediction)
        │
        ▼
Explainable Output
(anomaly score, decision, and flagged reasons)
        │
        ▼
Streamlit Web Interface
(interactive dashboard for review analysis)

---

## Machine Learning Model

The system uses Isolation Forest, an unsupervised anomaly detection algorithm designed to identify rare patterns in data.

Isolation Forest works by isolating observations that behave differently from the majority of the dataset.

This approach allows the system to detect statistically unusual reviews without requiring labeled fake review data.

---

## Dataset

The system was trained and tested using review datasets derived from the Amazon SNAP Review Dataset.

The dataset includes:

- review text

- star ratings

- reviewer identifiers

- verified purchase information

These attributes were used to derive behavioural and linguistic features for anomaly detection.

---

## 📁 Project Structure

Hybrid-Detection-System
Hybrid-Detection-System
│
├── app.py                     # Streamlit application interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── README.md                  # Project documentation
│
├── data
│   ├── anomaly_scores.csv     # Dataset anomaly score distribution
│   └── threshold.pkl          # Learned anomaly threshold for inference
│
└── src
    ├── feature_engineering.py # Feature extraction pipeline
    ├── anomaly_scoring.py     # Rule-based anomaly score calculation
    ├── isolation_model.py     # Isolation Forest anomaly detection model
    ├── decision_engine.py     # Hybrid decision logic
    └── visualization.py       # Anomaly score visualizations


How the Detection Works

The anomaly score is computed using a combination of signals:

1. Rating-Sentiment Mismatch

Difference between normalized star rating and text sentiment score.

2. Review Length Deviation

Extremely short reviews compared to the dataset median.

3. Emotional Language

Detection of exaggerated emotional expressions.

4. Reviewer Activity Pattern

Users with unusually high review activity may indicate suspicious behaviour.

These features are combined to compute the final anomaly score.

---

## Running the Project Locally

Clone the repository:
```bash
git clone https://github.com/yourusername/hybrid-review-detection.git
cd hybrid-review-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
streamlit run app.py
```

Build the Docker image:
```bash
docker build -t review-detector .
```

Run the container:
```bash
docker run -p 8501:8501 review-detector
```

---

## Possible enhancements:

- Deep learning models (DistilBERT / RoBERTa)

- Reviewer network analysis

- Graph-based fraud detection

- Real-time moderation API

- Multi-product review aggregation

---

## 👩‍💻 Author

Arshia Sood
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: https://github.com/Arshia-Sood

⭐ If you like this project, give it a star!