def apply_decision(df):
    threshold=df["anomaly_score"].quantile(0.95)
    
    df["decision"]=df["anomaly_score"].apply(lambda x: "Anomalous" if x>=threshold else "Normal")

    return df,threshold