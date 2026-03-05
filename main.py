import pandas as pd
from src.feature_engineering import add_features
from src.anomaly_scoring import compute_anomaly_score
from src.decision_engine import apply_decision
from src.visualization import anomaly_plot_distribution,feature_plot_relationships

df=pd.read_csv("data\preprocessed\preprocessed_dataset.csv")

df=add_features(df)
df=compute_anomaly_score(df)
df,threshold=apply_decision(df)

df.to_csv("data/preprocessed/final_validated_reviews.csv",index=False)

print("Anomaly Distribution:",threshold)

print("\nDecision Distribution:")
print(df["decision"].value_counts(normalize=True))

print("\nSample Anomalous Reviews:")
print(df[df["decision"]=="Anomalous"][["reviewText","anomaly_score","word_count"]].head(10))

anomaly_plot_distribution(df)
feature_plot_relationships(df)
