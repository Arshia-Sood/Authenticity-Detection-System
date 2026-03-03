import numpy as np

def compute_thresholds(df):
    thresholds={}

    thresholds["mismatch_threshold"]=np.percentile(df["mismatch"],95)
    thresholds["wordcount_threshold"]=np.percentile(df["word_count"],5)

    return thresholds

def apply_validation(df,thresholds):
    df["high_mismatch_flag"]=(df["mismatch"]>thresholds["mismatch_threshold"]).astype(int)

    df["short_review_flag"]=(df["word_count"]<thresholds["wordcount_threshold"]).astype(int)

    df["emotional_flag"]=((df["exclamation_count"]>3)|(df["extreme_sentiment"]==1)).astype(int)

    #risk-score
    df["risk_score"]=(df["high_mismatch_flag"]+df["short_review_flag"]+df["emotional_flag"])

    #verified purchase reduces risk slightly
    df.loc[df["verified"]==1,"risk_score"]-=0.5

    #final decision
    df["decision"]=df["risk_score"].apply(lambda x: "Reject" if x>=2 else "Accept")

    return df