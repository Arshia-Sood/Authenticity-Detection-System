import matplotlib.pyplot as plt
import seaborn as sns

def anomaly_plot_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df["anomaly_score"],bins=50,kde=True)

    plt.title("Distribution of Review Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of Reviews")

    plt.show()

def feature_plot_relationships(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["word_count"],y=df["anomaly_score"],hue=df["decision"],alpha=0.6)

    plt.title("Review Length vs Anomaly Score")
    plt.xlabel("Word Count")
    plt.ylabel("Anomaly Score")

    plt.show()