import pandas as pd
from src.feature_engineering import add_features
from src.anomaly_scoring import compute_anomaly_score
from src.decision_engine import apply_decision
from src.isolation_model import run_isolation_forest
from src.visualization import anomaly_plot_distribution,feature_plot_relationships

df=pd.read_csv("data\preprocessed\preprocessed_dataset.csv")

df=add_features(df)
df=compute_anomaly_score(df)

threshold=df["anomaly_score"].quantile(0.95)

import pickle
with open("data/threshold.pkl","wb") as f:
    pickle.dump(threshold,f)

df=run_isolation_forest(df)
df=apply_decision(df,threshold)

df["anomaly_score"].to_csv("data/anomaly_scores.csv", index=False)

df.to_csv("data/preprocessed/final_validated_reviews.csv",index=False)

print("Anomaly Distribution:",threshold)

print("\nDecision Distribution:")
print(df["decision"].value_counts(normalize=True))

print("\nSample Anomalous Reviews:")
print(df[df["decision"]=="Anomalous"][["reviewText","anomaly_score","word_count"]].head(10))

anomaly_plot_distribution(df)
feature_plot_relationships(df)
