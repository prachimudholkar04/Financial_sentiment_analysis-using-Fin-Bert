# Sentiment Analysis on Financial News using FinBERT

# Cell 1: Install necessary packages (if needed)
#pip install transformers torch yfinance beautifulsoup4 requests nltk --quiet

# Cell 2: Imports
import pandas as pd

import requests
import plotly.express as px
import chart_studio.plotly as py
import chart_studio.tools as tls
from transformers import pipeline
import chart_studio.tools as tls
import yfinance as yf
import matplotlib.pyplot as plt
import nltk
from datetime import datetime, timedelta
import os
from wordcloud import WordCloud
import ssl


# SSL workaround for NLTK on macOS
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Cell 3: Load financial news using NewsAPI
NEWS_API_KEY = "0dfa8ba0c8aa47de8dcb11e883173f0a"  # Replace with your actual API key
def get_newsapi_headlines(query="finance", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles if article.get("title")]
    return pd.DataFrame({"headline": headlines})

# Get financial news
news_df = get_newsapi_headlines("Apple stock")
print(news_df.head())

# Cell 4: Load FinBERT model
sentiment_pipeline = pipeline("sentiment-analysis",
                               model="ProsusAI/finbert")

# Cell 5: Run sentiment prediction
def classify_sentiment(text_list):
    sentiments = sentiment_pipeline(text_list)
    return [x['label'] for x in sentiments]

news_df['sentiment'] = classify_sentiment(news_df['headline'].tolist())
print(news_df.head())


# Set your Plotly credentials (sign up at https://chart-studio.plotly.com)
tls.set_credentials_file(username='prachimudholkar', api_key='0dfa8ba0c8aa47de8dcb11e883173f0a')

# Cell 6: Interactive sentiment breakdown with Plotly
sentiment_counts = news_df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
             title='Interactive Sentiment Breakdown', text='Count')
fig.update_layout(template='plotly_white')
fig.write_html("sentiment_breakdown.html")
# Cell 7: Word Cloud for Each Sentiment
for sentiment in news_df['sentiment'].unique():
    subset = news_df[news_df['sentiment'] == sentiment]
    text = ' '.join(subset['headline'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Headlines')
    plt.show()

# Cell 8: Interactive stock price plot

ticker = "AAPL"
data = yf.download(ticker, period="7d", auto_adjust=False)
data.reset_index(inplace=True)
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]  # flatten multi-index

fig2 = px.line(data, x='Date', y='Close', title=f'{ticker} Closing Price - Last 7 Days')
fig2.update_layout(template='plotly_white', yaxis_title='Price ($)')
fig2.write_html("stock_price_trend.html")


# Cell 9: Save data to CSV (optional)
os.makedirs("data", exist_ok=True)
news_df.to_csv("data/news_with_sentiment.csv", index=False)

# Cell 10: Summary
print("✅ Sentiment analysis complete. Visuals + labeled data saved.")



