import numpy as np

def compute_anomaly_score(df):
    mismatch_max=df["mismatch"].max() or 1
    df["mismatch_norm"]=df["mismatch"]/mismatch_max

    median_words=df["word_count"].median()

    df["length_deviation"]=df["word_count"].apply(lambda x: max(0,(median_words-x)/median_words))

    df["emotion_score"]=((df["exclamation_count"]/5)+df["extreme_sentiment"])

    df["emotion_score"]=df["emotion_score"].clip(0,1)

    max_reviews=df["reviewer_review_count"].max() or 1

    df["reviewer_activity_score"]=(df["reviewer_review_count"]/max_reviews)

    df["reviewer_activity_score"]=df["reviewer_activity_score"].clip(0,1)

    df["detail_bonus"]=df["word_count"].apply(lambda x:0.3 if x>40 else 0)

    df["anomaly_score"]=(0.5*df["mismatch_norm"]+0.2*df["length_deviation"]+0.15*df["emotion_score"]+0.15*df["reviewer_activity_score"]-df["detail_bonus"])

    df.loc[(df["sentiment_score"]>0.8) & (df["star_norm"]>0.8) & (df["word_count"]<10), "anomaly_score"]-=0.25

    return df