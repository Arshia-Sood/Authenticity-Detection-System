import pandas as pd
from src.feature_engineering import add_features
from src.anomaly_scoring import compute_anomaly_score
from src.decision_engine import apply_decision
from src.isolation_model import run_isolation_forest
from src.visualization import anomaly_plot_distribution,feature_plot_relationships
import pickle

df=pd.read_csv("data\preprocessed\preprocessed_dataset.csv")

df=add_features(df)
df=compute_anomaly_score(df)

threshold=df["anomaly_score"].quantile(0.95)

with open("data/threshold.pkl","wb") as f:
    pickle.dump(threshold,f)

df,model,scaler=run_isolation_forest(df,return_model=True)
df=apply_decision(df,threshold)

with open("data/isolation_model.pkl","wb") as f:
    pickle.dump(model, f)
with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

min_score=df["anomaly_score"].min()
max_score=df["anomaly_score"].max()

with open("data/score_range.pkl", "wb") as f:
    pickle.dump((min_score, max_score), f)

df["anomaly_score"].to_csv("data/anomaly_scores.csv", index=False)
df.to_csv("data/preprocessed/final_validated_reviews.csv",index=False)

print("Anomaly Distribution:",threshold)

print("\nDecision Distribution:")
print(df["decision"].value_counts(normalize=True))

print("\nSample Anomalous Reviews:")
print(df[df["decision"]=="Anomalous"][["reviewText","anomaly_score","word_count"]].head(10))

anomaly_plot_distribution(df)
feature_plot_relationships(df)

model=run_isolation_forest(df,return_model=True)
