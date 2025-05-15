# 📈 Sentiment Analysis on Financial News using FinBERT

This project analyzes the **sentiment of real-world financial news** using **FinBERT**, a finance-specific transformer model. It scrapes the latest news headlines with NewsAPI, classifies sentiment (positive, negative, neutral), and visualizes the results interactively using **Plotly**. It also correlates the sentiment with actual stock price data from Yahoo Finance.

---

## 🚀 Features

- 🔍 **Real-time Financial News** from [NewsAPI](https://newsapi.org)
- 🧠 **FinBERT Transformer Model** for sentiment classification
- 📊 **Interactive Charts** with Plotly (sentiment breakdown, stock trends)
- ☁️ **Word Clouds** for sentiment-driven keyword insight
- 💹 **Stock Correlation** with 7-day price trends (via yFinance)
- 🧾 Exports results to CSV for further analysis

---

## 🛠️ Tech Stack

| Tool/Lib         | Purpose                             |
|------------------|-------------------------------------|
| `FinBERT`        | Sentiment analysis on financial text |
| `NewsAPI`        | Fetch real-time financial headlines  |
| `Plotly`         | Interactive visualizations          |
| `yFinance`       | Historical stock price data         |
| `Pandas/Matplotlib` | Data wrangling & word clouds    |

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-finance.git
   cd sentiment-analysis-finance
