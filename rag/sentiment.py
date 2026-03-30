from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return 1.01 if result['label'] == 'POSITIVE' else 0.99