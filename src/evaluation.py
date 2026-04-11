import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Secure resources
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def run_evaluation():
    # 1. Load data
    if not os.path.exists("data/processed_en.csv"):
        print("Error: processed file not found.")
        return
    
    df = pd.read_csv("data/processed_en.csv")
    vader = SentimentIntensityAnalyzer()

    # 2. Sentiment Functions
    def get_textblob_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    # 3. Metrics calculation
    df['pros_vader'] = df['pros'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
    df['cons_vader'] = df['cons'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
    df['cons_textblob'] = df['cons'].apply(get_textblob_sentiment)

    # Simplified labels
    df['sentiment_final'] = df['cons_vader'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))

    # --- COMPLETE REPORT ---
    print("\n" + "="*40)
    print("SUMMARY OF COMPANY METRICS")
    print("="*40)
    print(f"Total reviews analyzed: {len(df)}")
    print(f"Average sentiment (Pros): {df['pros_vader'].mean():.2f}")
    print(f"Average sentiment (Cons): {df['cons_vader'].mean():.2f}")
    print("\nSentiment distribution in Cons:")
    print(df['sentiment_final'].value_counts(normalize=True) * 100)
    print("="*40 + "\n")

    # --- VISUALIZATIONS ---
    plt.figure(figsize=(12, 5))

    # Graph 1: Sentiment Distribution (Bars)
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sentiment_final', palette='viridis')
    plt.title('Sentiment Distribution (Cons)')
    plt.ylabel('Number of reviews')

    # Graph 2: Comparison of Pros vs Cons (Boxplot)
    plt.subplot(1, 2, 2)
    sentiment_data = df[['pros_vader', 'cons_vader']].melt(var_name='Type', value_name='Score')
    sns.boxplot(data=sentiment_data, x='Type', y='Score', palette='Set2')
    plt.title('Sentiment Range: Pros vs Cons')
    
    plt.tight_layout()
    plt.savefig('data/visual_report.png') # Guarda la imagen para el MLOps
    print("Graph saved in 'data/visual_report.png'")
    plt.show()

    # 4. Save results
    df.to_csv("data/final_sentiment_report.csv", index=False)
    print("Complete Evaluation. Results saved.")

if __name__ == "__main__":
    run_evaluation()