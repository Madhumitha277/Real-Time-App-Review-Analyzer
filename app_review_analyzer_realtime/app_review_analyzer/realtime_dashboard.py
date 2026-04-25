"""
Approach 2: Real-Time Review Polling via REST API
==================================================
Polls a REST API (App Store Connect or any review API) for new reviews,
stores them in PostgreSQL (or SQLite), and serves a live Streamlit
dashboard that auto-refreshes every N seconds.

Install:
    pip install requests sqlalchemy streamlit plotly pandas textblob nltk

Run dashboard:
    streamlit run realtime_dashboard.py

Supported sources (set SOURCE in config):
  - "mock"        : built-in random review generator (no API key needed)
  - "appstore"    : Apple App Store RSS feed (free, no key)
  - "twitter"     : Twitter/X filtered stream (needs bearer token)
"""

import random
import re
import time
from datetime import datetime, timezone

import nltk
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from textblob import TextBlob

for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SOURCE        = "appstore"           # "mock" | "appstore" | "twitter"
REFRESH_SECS  = 10               # dashboard auto-refresh interval
DB_URL        = "sqlite:///realtime_reviews.db"
APP_STORE_ID  = "324684580"      # Spotify's iTunes ID (example)
TWITTER_TOKEN = "YOUR_BEARER_TOKEN"

# ── DB ────────────────────────────────────────────────────────────────────────
engine = create_engine(DB_URL)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS live_reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            fetched_at  TEXT,
            rating      REAL,
            content     TEXT,
            clean       TEXT,
            sentiment   TEXT,
            polarity    REAL
        )
    """))


# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)


def get_sentiment(text: str):
    p = TextBlob(str(text)).sentiment.polarity
    label = "Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral")
    return label, round(p, 4)


def store_reviews(rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_sql("live_reviews", engine, if_exists="append", index=False)


# ── DATA SOURCES ──────────────────────────────────────────────────────────────

MOCK_TEMPLATES = [
    ("App crashes every time on startup!", 1),
    ("Love the new update, works great!", 5),
    ("Battery drain is too high.", 2),
    ("Please add dark mode.", 3),
    ("Best app I have used all year.", 5),
    ("Login keeps failing, very frustrating.", 1),
    ("Pretty good but could be faster.", 3),
    ("Amazing features, highly recommend.", 5),
    ("Constant bugs after the update.", 2),
    ("Would love offline mode support.", 4),
]


def fetch_mock() -> list[dict]:
    """Generate 1-3 fake reviews (for demo/testing — no API key needed)."""
    rows = []
    for _ in range(random.randint(1, 3)):
        content, rating = random.choice(MOCK_TEMPLATES)
        clean   = clean_text(content)
        sentiment, polarity = get_sentiment(content)
        rows.append({
            "source":     "mock",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rating":     rating,
            "content":    content,
            "clean":      clean,
            "sentiment":  sentiment,
            "polarity":   polarity,
        })
    return rows


def fetch_appstore(app_id: str = APP_STORE_ID) -> list[dict]:
    """
    Pull reviews from Apple App Store RSS feed (free, no auth).
    Replace APP_STORE_ID with your app's iTunes ID.
    Find it at: https://apps.apple.com — the number in the URL.
    """
    url = f"https://itunes.apple.com/us/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("feed", {}).get("entry", [])
    except Exception as exc:
        print(f"App Store fetch error: {exc}")
        return []

    rows = []
    for entry in entries:
        content = entry.get("content", {}).get("label", "")
        rating  = float(entry.get("im:rating", {}).get("label", 3))
        clean   = clean_text(content)
        sentiment, polarity = get_sentiment(content)
        rows.append({
            "source":     "appstore",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rating":     rating,
            "content":    content,
            "clean":      clean,
            "sentiment":  sentiment,
            "polarity":   polarity,
        })
    return rows


def fetch_twitter(query: str = "app review -is:retweet lang:en") -> list[dict]:
    """
    Stream tweets matching a query (requires Twitter Developer account).
    Sign up at: https://developer.twitter.com
    Set TWITTER_TOKEN = your bearer token above.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_TOKEN}"}
    params  = {"query": query, "max_results": 10, "tweet.fields": "created_at"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        tweets = resp.json().get("data", [])
    except Exception as exc:
        print(f"Twitter fetch error: {exc}")
        return []

    rows = []
    for tw in tweets:
        content = tw.get("text", "")
        clean   = clean_text(content)
        sentiment, polarity = get_sentiment(content)
        rows.append({
            "source":     "twitter",
            "fetched_at": tw.get("created_at", datetime.now(timezone.utc).isoformat()),
            "rating":     None,
            "content":    content,
            "clean":      clean,
            "sentiment":  sentiment,
            "polarity":   polarity,
        })
    return rows


def poll_once():
    """Fetch from the configured source and store."""
    if SOURCE == "appstore":
        rows = fetch_appstore()
    elif SOURCE == "twitter":
        rows = fetch_twitter()
    else:
        rows = fetch_mock()
    store_reviews(rows)
    return len(rows)


# ── STREAMLIT LIVE DASHBOARD ───────────────────────────────────────────────────

def load_df() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM live_reviews ORDER BY fetched_at DESC", engine)


st.set_page_config(page_title="Live Review Monitor", layout="wide", page_icon="📡")
st.title("Live Review Monitor")
st.caption(f"Source: {SOURCE.upper()} · Auto-refreshes every {REFRESH_SECS}s")

# Poll for new data on each page load
new = poll_once()
if new:
    st.toast(f"+{new} new review(s) added", icon="📬")

df = load_df()

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", len(df))
c2.metric("Positive",  int((df["sentiment"] == "Positive").sum()))
c3.metric("Negative",  int((df["sentiment"] == "Negative").sum()))
c4.metric("Avg Polarity", f"{df['polarity'].mean():.2f}" if len(df) else "—")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment over time")
    if len(df):
        df["fetched_at"] = pd.to_datetime(df["fetched_at"])
        trend = df.set_index("fetched_at").resample("1min")["sentiment"].value_counts().unstack(fill_value=0)
        fig = px.line(trend, color_discrete_map={
            "Positive": "#1D9E75", "Neutral": "#888780", "Negative": "#D85A30"
        })
        fig.update_layout(xaxis_title="Time", yaxis_title="Count", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Sentiment split")
    if len(df):
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig2 = px.pie(counts, names="Sentiment", values="Count", hole=0.4,
                      color="Sentiment", color_discrete_map={
                          "Positive": "#1D9E75", "Neutral": "#888780", "Negative": "#D85A30"
                      })
        fig2.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Latest reviews")
if len(df):
    st.dataframe(
        df[["fetched_at", "rating", "sentiment", "content"]].head(20),
        use_container_width=True,
    )

# Auto-refresh
time.sleep(REFRESH_SECS)
st.rerun()