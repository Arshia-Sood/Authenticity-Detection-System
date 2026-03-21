from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def run_isolation_forest(df,model=None,scaler=None,return_model=False):
    features=["mismatch_norm","length_deviation","emotion_score","reviewer_activity_score","word_count"]

    X=df[features]

    if scaler is None:
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X)
    else:
        X_scaled=scaler.transform(X)

    if model==None:
        model=IsolationForest(n_estimators=200,contamination=0.05,random_state=42)
        predictions=model.fit_predict(X_scaled)
    else:
        predictions=model.predict(X_scaled)

    df["iforest_prediction"]=predictions
    df["iforest_anomaly"]=df["iforest_prediction"].apply(lambda x: 1 if x==-1 else 0)

    if return_model:
        return df,model,scaler
    return df