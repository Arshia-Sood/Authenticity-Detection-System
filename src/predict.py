import pandas as pd
import pickle

from src.feature_engineering import add_features
from src.anomaly_scoring import compute_anomaly_score
from src.isolation_model import run_isolation_forest
from src.decision_engine import apply_decision


with open("data/threshold.pkl","rb") as f:
    threshold=pickle.load(f)
with open("data/isolation_model.pkl","rb") as f:
    model = pickle.load(f)
with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/score_range.pkl", "rb") as f:
    min_score, max_score = pickle.load(f)

def predict_review(review_text,star_rating):
    df=pd.DataFrame([{
        "reviewText":review_text,
        "overall":star_rating,
        "reviewerID":"user"
    }])

    df = add_features(df)
    df = compute_anomaly_score(df)
    df = run_isolation_forest(df,model=model,scaler=scaler)
    df = apply_decision(df, threshold)

    raw_score = float(df.loc[0, "anomaly_score"])
    normalized_score = (raw_score - min_score) / (max_score - min_score)
    normalized_score = max(0, min(normalized_score, 1))

    result = {
        "score": float(df.loc[0, "anomaly_score"]),
        "risk": normalized_score, 
        "decision": df.loc[0, "decision"],
        "mismatch": float(df.loc[0, "mismatch"]),
        "length_dev": float(df.loc[0, "length_deviation"]),
        "emotion": float(df.loc[0, "emotion_score"])
    }

    return result, threshold