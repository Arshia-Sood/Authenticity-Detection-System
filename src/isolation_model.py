from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def run_isolation_forest(df):
    features=["mismatch_norm","length_deviation","emotion_score","reviewer_activity_score","word_count"]

    X=df[features]

    scalar=StandardScaler()
    X_scaled=scalar.fit_transform(X)

    model=IsolationForest(n_estimators=200,contamination=0.05,random_state=42)

    predictions=model.fit_predict(X_scaled)

    df["iforest_prediction"]=predictions

    df["iforest_anomaly"]=df["iforest_prediction"].apply(lambda x: 1 if x==-1 else 0)

    return df