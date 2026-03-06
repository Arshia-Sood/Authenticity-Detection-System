import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from src.feature_engineering import add_features
from src.anomaly_scoring import compute_anomaly_score
from src.isolation_model import run_isolation_forest
from src.decision_engine import apply_decision

st.set_page_config(page_title="E-Commerce Review Anomaly Detector", layout="centered")

# ---------- Header ----------
st.markdown("""
<h1 style='text-align:center;'>🛒 E-Commerce Review Anomaly Detector</h1>
<p style='text-align:center;'>Hybrid anomaly detection using linguistic signals and Isolation Forest</p>
""", unsafe_allow_html=True)

st.divider()

scores = pd.read_csv("data/anomaly_scores.csv")

try:
    with open("data/threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

except FileNotFoundError:
    st.error("Threshold file not found. Please run main.py first.")
    st.stop()

# ---------- Input Section ----------
with st.container():

    st.subheader("Try Example Reviews")

    # initialize session state
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    if "star_rating" not in st.session_state:
        st.session_state.star_rating = 5

    colA, colB, colC = st.columns(3)

    with colA:
        if st.button("Normal Review"):
            st.session_state.review_text = (
                "I have been using this shampoo for two weeks and it works great. "
                "My hair feels softer and the fragrance is pleasant."
            )
            st.session_state.star_rating = 5

    with colB:
        if st.button("Rating Mismatch"):
            st.session_state.review_text = (
                "This product is terrible. It damaged my hair and made my scalp itch badly."
            )
            st.session_state.star_rating = 5

    with colC:
        if st.button("Spam-like Review"):
            st.session_state.review_text = (
                "THIS PRODUCT IS AMAZING!!!!! BEST THING EVER!!!! YOU MUST BUY IT RIGHT NOW!!!!"
            )
            st.session_state.star_rating = 5

    review_text = st.text_area(
        "Enter product review",
        height=200,
        key="review_text"
    )

    star_rating = st.slider(
        "Star rating",
        1,
        5,
        key="star_rating"
    )

    analyze = st.button("Analyze Review")

st.divider()

# ---------- Analysis ----------
if analyze:

    if not review_text.strip():
        st.warning("Please enter some review text before analysing.")

    else:

        df = pd.DataFrame([
            {
                "reviewText": review_text,
                "overall": star_rating,
                "reviewerID": "test_user",
            }
        ])

        # pipeline
        df = add_features(df)
        df = compute_anomaly_score(df)
        df = run_isolation_forest(df)
        df = apply_decision(df, threshold)

        score = df.loc[0, "anomaly_score"]
        decision = df.loc[0, "decision"]

        # ---------- Metrics Dashboard ----------
        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Anomaly Score", f"{score:.3f}")

        with col2:
            st.metric("Threshold", f"{threshold:.3f}")

        with col3:
            st.metric("Star Rating", star_rating)

        # ---------- Risk Indicator ----------
        risk_ratio = min(score / threshold, 1.0)

        st.progress(risk_ratio)
        st.caption("Relative anomaly risk compared to dataset threshold")

        # ---------- Distribution Visualization ----------
        st.subheader("Score Position in Dataset Distribution")

        fig, ax = plt.subplots()

        ax.hist(scores["anomaly_score"], bins=40, alpha=0.7)

        ax.axvline(score, linestyle="--", linewidth=2, label="Current Review Score")

        ax.axvline(threshold, linestyle=":", linewidth=2, label="Anomaly Threshold")

        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)

        # ---------- Decision ----------
        if decision == "Normal":
            st.success("This review is classified as **Normal**.")

        else:
            st.error("This review is classified as **Anomalous**.")

        # ---------- Explainability ----------
        reasons = []

        if df.loc[0, "mismatch_norm"] > 0.3:
            reasons.append("Rating–sentiment mismatch detected")

        if df.loc[0, "length_deviation"] > 0.3:
            reasons.append("Review is unusually short compared to typical reviews")

        if df.loc[0, "emotion_score"] > 0.5:
            reasons.append("Emotionally exaggerated language detected")

        if reasons:
            st.subheader("Possible Reasons")

            for r in reasons:
                st.write(f"- {r}")

        # ---------- Feature Table ----------
        with st.expander("View extracted features"):
            st.dataframe(df.T)

st.divider()
st.caption("Hybrid Review Authenticity Detection System | Final Year Project")