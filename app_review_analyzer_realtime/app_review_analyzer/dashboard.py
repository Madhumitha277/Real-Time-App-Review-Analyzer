"""
Streamlit Dashboard — App Store Review Insights Analyzer
=========================================================
Run with:
    streamlit run dashboard.py

Then open http://localhost:8501 in your browser.
"""

import re
import sys
from collections import Counter

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob

# ── NLTK ─────────────────────────────────────────────────────────────────────
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="App Review Insights",
    page_icon="📱",
    layout="wide",
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

SAMPLE_REVIEWS = [
    {"date": "2024-01-05", "rating": 1, "review": "App crashes every time I open it. Very frustrating!"},
    {"date": "2024-01-08", "rating": 2, "review": "Keeps crashing on login. Please fix the login bug."},
    {"date": "2024-01-10", "rating": 5, "review": "Love this app! Super smooth and fast. Highly recommend."},
    {"date": "2024-01-12", "rating": 4, "review": "Great features overall. Would love a dark mode option."},
    {"date": "2024-01-15", "rating": 3, "review": "Battery drains very quickly when the app is running in background."},
    {"date": "2024-01-18", "rating": 1, "review": "Terrible update broke everything. Performance is awful now."},
    {"date": "2024-01-20", "rating": 5, "review": "Best app I've used. The UI is beautiful and intuitive."},
    {"date": "2024-02-01", "rating": 2, "review": "Notifications not working. Please add notification customization."},
    {"date": "2024-02-03", "rating": 4, "review": "Really useful app! Please add export to PDF feature."},
    {"date": "2024-02-05", "rating": 1, "review": "App is slow and laggy. Too many bugs. Uninstalling."},
    {"date": "2024-02-08", "rating": 5, "review": "Amazing! Works flawlessly. Best purchase ever."},
    {"date": "2024-02-10", "rating": 3, "review": "Average app. Would be better with offline mode support."},
    {"date": "2024-02-12", "rating": 2, "review": "Constant crashes and freezes. Login is broken again."},
    {"date": "2024-02-15", "rating": 4, "review": "Good app but needs a widget for home screen."},
    {"date": "2024-02-18", "rating": 1, "review": "Data loss after update. Lost all my saved data. Horrible bug."},
    {"date": "2024-03-01", "rating": 5, "review": "Fantastic design and performance. Love the new update!"},
    {"date": "2024-03-04", "rating": 3, "review": "OK app. Sometimes slow. Would appreciate multi-language support."},
    {"date": "2024-03-07", "rating": 2, "review": "Poor battery life, app drains phone quickly. Needs fixing."},
    {"date": "2024-03-10", "rating": 5, "review": "Excellent! The sync feature works perfectly. Very happy!"},
    {"date": "2024-03-15", "rating": 1, "review": "Broken after latest update. Crashes immediately on startup."},
    {"date": "2024-03-18", "rating": 4, "review": "Nice app. Please add calendar integration feature soon."},
    {"date": "2024-03-22", "rating": 3, "review": "Decent but the search function is really slow and buggy."},
    {"date": "2024-04-01", "rating": 5, "review": "Perfect! No bugs, super fast. Best app in this category."},
    {"date": "2024-04-05", "rating": 2, "review": "App crashes when I try to upload photos. Very frustrating bug."},
    {"date": "2024-04-10", "rating": 1, "review": "Absolutely terrible. Slow, buggy, crashes constantly."},
]

COMPLAINT_WORDS = {
    "crash", "crashes", "crashing", "bug", "bugs", "buggy", "slow", "lag",
    "laggy", "broken", "freezes", "freeze", "error", "issue", "problem",
    "terrible", "horrible", "awful", "drain", "battery", "loss",
}
FEATURE_WORDS = {
    "add", "feature", "want", "need", "wish", "request", "please", "option",
    "support", "integrate", "integration", "mode", "widget", "export",
}


def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)


def get_sentiment(text):
    p = TextBlob(str(text)).sentiment.polarity
    if p > 0.1:
        return "Positive"
    if p < -0.1:
        return "Negative"
    return "Neutral"


def load_and_process(df_raw):
    df = df_raw.copy()
    df.dropna(subset=["review"], inplace=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["clean"] = df["review"].apply(clean_text)
    df["sentiment"] = df["review"].apply(get_sentiment)
    df["is_complaint"] = df["clean"].apply(lambda t: bool(COMPLAINT_WORDS & set(t.split())))
    df["is_feature"] = df["clean"].apply(lambda t: bool(FEATURE_WORDS & set(t.split())))
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("📱 Review Insights")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(raw_df):,} rows")
else:
    raw_df = pd.DataFrame(SAMPLE_REVIEWS)
    st.sidebar.info("Using built-in sample data")

df = load_and_process(raw_df)

sentiments = st.sidebar.multiselect(
    "Filter by sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
)
df = df[df["sentiment"].isin(sentiments)]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("App Store Review Insights Analyzer")
st.caption("Turn raw user feedback into actionable product insights.")
st.markdown("---")

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Reviews", len(df))
col2.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
col3.metric("Positive", f"{(df['sentiment']=='Positive').sum()}")
col4.metric("Complaints", f"{df['is_complaint'].sum()}")
col5.metric("Feature Requests", f"{df['is_feature'].sum()}")

st.markdown("---")

# ── ROW 1: Pie + Trend ────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Sentiment Split")
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig_pie = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#1D9E75", "Neutral": "#888780", "Negative": "#D85A30"},
        hole=0.35,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.subheader("Sentiment Trend Over Time")
    trend = df.groupby(["month", "sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(
        trend,
        x="month",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={"Positive": "#1D9E75", "Neutral": "#888780", "Negative": "#D85A30"},
    )
    fig_trend.update_layout(xaxis_title="Month", yaxis_title="Reviews", margin=dict(t=10))
    st.plotly_chart(fig_trend, use_container_width=True)

# ── ROW 2: Keywords + Rating ──────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("Top Keywords")
    all_words = " ".join(df["clean"]).split()
    freq = Counter(all_words).most_common(12)
    kw_df = pd.DataFrame(freq, columns=["Word", "Frequency"])
    fig_kw = px.bar(
        kw_df,
        x="Frequency",
        y="Word",
        orientation="h",
        color="Frequency",
        color_continuous_scale="Blues",
    )
    fig_kw.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig_kw, use_container_width=True)

with c4:
    st.subheader("Rating Distribution")
    rating_df = df["rating"].astype(int).value_counts().sort_index().reset_index()
    rating_df.columns = ["Stars", "Count"]
    fig_rating = px.bar(
        rating_df,
        x="Stars",
        y="Count",
        color_discrete_sequence=["#378ADD"],
        text="Count",
    )
    fig_rating.update_traces(textposition="outside")
    fig_rating.update_layout(xaxis=dict(tickmode="linear"), margin=dict(t=10))
    st.plotly_chart(fig_rating, use_container_width=True)

# ── ROW 3: Complaints & Feature Requests ──────────────────────────────────────
st.markdown("---")
c5, c6 = st.columns(2)

with c5:
    st.subheader("⚠️ Complaint Reviews")
    complaints = df[df["is_complaint"]][["date", "rating", "review"]].head(8)
    st.dataframe(complaints.reset_index(drop=True), use_container_width=True)

with c6:
    st.subheader("💡 Feature Requests")
    features = df[df["is_feature"]][["date", "rating", "review"]].head(8)
    st.dataframe(features.reset_index(drop=True), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python · TextBlob · Plotly · Streamlit")
