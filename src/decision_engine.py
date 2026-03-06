def apply_decision(df,threshold):
    df["rule_anomaly"]=df["anomaly_score"].apply(lambda x: 1 if x> threshold else 0)

    df["final_anomaly"]=(df["rule_anomaly"]+df["iforest_anomaly"])

    df["decision"]=df["final_anomaly"].apply(lambda x: "Anomalous" if x>=1 else "Normal")

    return df