# 📱 App Store Review Insights Analyzer

> Turn raw user reviews into actionable product intelligence.

## What it does

| Feature | Description |
|---|---|
| Sentiment Analysis | Classifies each review as Positive / Neutral / Negative using TextBlob NLP |
| Complaint Detection | Flags reviews containing known complaint vocabulary |
| Feature Request Mining | Identifies reviews asking for new features |
| Visualizations | Pie chart, bar chart, trend line, rating histogram |
| Interactive Dashboard | Streamlit dashboard with filters and Plotly charts |

---

## Quick Start

```bash
# 1. Clone / download the project
cd app_review_analyzer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the CLI analyzer (uses built-in sample data)
python analyzer.py

# 5. Run with your own CSV
python analyzer.py --csv path/to/reviews.csv

# 6. Launch the interactive dashboard
streamlit run dashboard.py
```

---

## CSV Format

Your CSV must have at least these three columns:

| Column | Type | Example |
|---|---|---|
| `review` | string | "App crashes on login" |
| `rating` | int 1–5 | 2 |
| `date` | YYYY-MM-DD | 2024-03-15 |

---

## Free Datasets (Kaggle)

| Dataset | Link |
|---|---|
| Google Play Store Reviews | https://www.kaggle.com/datasets/lava18/google-play-store-apps |
| Apple App Store Reviews | https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews |
| Twitter US Airline Sentiment | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| Amazon Product Reviews | https://www.kaggle.com/datasets/bittlingmayer/amazonreviews |

---

## Project Structure

```
app_review_analyzer/
├── analyzer.py          # Main CLI pipeline
├── dashboard.py         # Streamlit interactive dashboard
├── requirements.txt     # Python dependencies
├── README.md
├── data/                # Put your CSV files here
└── outputs/             # Generated charts saved here
    ├── sentiment_pie.png
    ├── keyword_bar.png
    ├── sentiment_trend.png
    ├── rating_distribution.png
    └── enriched_reviews.csv
```

---

## Portfolio Upgrade Ideas

- **Advanced NLP**: Replace TextBlob with a fine-tuned BERT model (e.g. `cardiffnlp/twitter-roberta-base-sentiment`)
- **Topic Modelling**: Use LDA (Latent Dirichlet Allocation) to auto-discover complaint themes
- **Automated Alerts**: Email/Slack alert when negative sentiment spikes above a threshold
- **Multi-app Comparison**: Compare sentiment across competitor apps side-by-side
- **Database Storage**: Store reviews in SQLite or PostgreSQL instead of CSV
- **Scheduled Scraping**: Use `google-play-scraper` to auto-fetch new reviews daily
- **Word Cloud**: Add a visual word cloud using the `wordcloud` library
- **CI/CD**: Add GitHub Actions to run analysis on every new data push

---

## Tech Stack

- **Python 3.10+**
- **Pandas** — data loading & manipulation
- **NLTK + TextBlob** — NLP preprocessing & sentiment
- **Matplotlib + Seaborn** — static charts
- **Plotly + Streamlit** — interactive dashboard

---

*Built as a portfolio data analytics project. MIT License.*
