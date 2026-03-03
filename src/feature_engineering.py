import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer=SentimentIntensityAnalyzer()

def compute_sentiment(text):
    score=analyzer.polarity_scores(str(text))
    return score["compound"]

def add_features(df):
    df["sentiment_raw"]=df["reviewText"].apply(compute_sentiment)
    df["sentiment_score"]=(df["sentiment_raw"]+1)/2

    df["star_norm"]=(df["overall"]-1)/4

    df["mismatch"]=abs(df["star_norm"]-df["sentiment_score"])

    df["word_count"]=df["reviewText"].apply(lambda x: len(str(x).split()))

    df["exclamation_count"]=df["reviewText"].apply(lambda x:str(x).count("!"))
    df["extreme_sentiment"]=df["sentiment_raw"].apply(lambda x:1 if abs(x)>0.95 else 0)

    return df
