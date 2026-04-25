"""
App Store Review Insights Analyzer
====================================
Analyzes mobile app reviews to extract:
- Sentiment trends (positive / negative / neutral)
- Common user complaints
- Feature requests

Dependencies:
    pip install pandas textblob nltk matplotlib seaborn wordcloud

Usage:
    python analyzer.py                         # uses built-in sample data
    python analyzer.py --csv your_data.csv     # use your own CSV
"""

import argparse
import os
import re
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nltk
import pandas as pd
import seaborn as sns
from collections import Counter
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ── NLTK downloads (runs once) ────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ── 1. SAMPLE DATA ────────────────────────────────────────────────────────────

SAMPLE_REVIEWS = [
    {"review_id": 1,  "date": "2024-01-05", "rating": 1, "review": "App crashes every time I open it. Very frustrating!"},
    {"review_id": 2,  "date": "2024-01-08", "rating": 2, "review": "Keeps crashing on login. Please fix the login bug."},
    {"review_id": 3,  "date": "2024-01-10", "rating": 5, "review": "Love this app! Super smooth and fast. Highly recommend."},
    {"review_id": 4,  "date": "2024-01-12", "rating": 4, "review": "Great features overall. Would love a dark mode option."},
    {"review_id": 5,  "date": "2024-01-15", "rating": 3, "review": "Battery drains very quickly when the app is running in background."},
    {"review_id": 6,  "date": "2024-01-18", "rating": 1, "review": "Terrible update broke everything. Performance is awful now."},
    {"review_id": 7,  "date": "2024-01-20", "rating": 5, "review": "Best app I've used. The UI is beautiful and intuitive."},
    {"review_id": 8,  "date": "2024-02-01", "rating": 2, "review": "Notifications not working. Please add notification customization."},
    {"review_id": 9,  "date": "2024-02-03", "rating": 4, "review": "Really useful app! Please add export to PDF feature."},
    {"review_id": 10, "date": "2024-02-05", "rating": 1, "review": "App is slow and laggy. Too many bugs. Uninstalling."},
    {"review_id": 11, "date": "2024-02-08", "rating": 5, "review": "Amazing! Works flawlessly. Best purchase ever."},
    {"review_id": 12, "date": "2024-02-10", "rating": 3, "review": "Average app. Would be better with offline mode support."},
    {"review_id": 13, "date": "2024-02-12", "rating": 2, "review": "Constant crashes and freezes. Login is broken again."},
    {"review_id": 14, "date": "2024-02-15", "rating": 4, "review": "Good app but needs a widget for home screen."},
    {"review_id": 15, "date": "2024-02-18", "rating": 1, "review": "Data loss after update. Lost all my saved data. Horrible bug."},
    {"review_id": 16, "date": "2024-03-01", "rating": 5, "review": "Fantastic design and performance. Love the new update!"},
    {"review_id": 17, "date": "2024-03-04", "rating": 3, "review": "OK app. Sometimes slow. Would appreciate multi-language support."},
    {"review_id": 18, "date": "2024-03-07", "rating": 2, "review": "Poor battery life, app drains phone quickly. Needs fixing."},
    {"review_id": 19, "date": "2024-03-10", "rating": 5, "review": "Excellent! The sync feature works perfectly. Very happy!"},
    {"review_id": 20, "date": "2024-03-15", "rating": 1, "review": "Broken after latest update. Crashes immediately on startup."},
    {"review_id": 21, "date": "2024-03-18", "rating": 4, "review": "Nice app. Please add calendar integration feature soon."},
    {"review_id": 22, "date": "2024-03-22", "rating": 3, "review": "Decent but the search function is really slow and buggy."},
    {"review_id": 23, "date": "2024-04-01", "rating": 5, "review": "Perfect! No bugs, super fast. Best app in this category."},
    {"review_id": 24, "date": "2024-04-05", "rating": 2, "review": "App crashes when I try to upload photos. Very frustrating bug."},
    {"review_id": 25, "date": "2024-04-10", "rating": 1, "review": "Absolutely terrible. Slow, buggy, crashes constantly."},
]


# ── 2. TEXT CLEANING ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)           # remove non-alpha
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── 3. SENTIMENT ANALYSIS ─────────────────────────────────────────────────────

def get_sentiment(text: str) -> str:
    """
    TextBlob polarity:  > 0.1  → Positive
                       < -0.1  → Negative
                        else   → Neutral
    """
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"


# ── 4. COMPLAINT & FEATURE EXTRACTION ────────────────────────────────────────

COMPLAINT_KEYWORDS = {
    "crash", "crashes", "crashing", "bug", "bugs", "buggy",
    "slow", "lag", "laggy", "broken", "freezes", "freeze",
    "error", "issue", "problem", "terrible", "horrible", "awful",
    "drain", "battery", "loss", "lost",
}

FEATURE_KEYWORDS = {
    "add", "feature", "want", "need", "wish", "request",
    "please", "option", "support", "integrate", "integration",
    "mode", "widget", "export",
}


def tag_complaint(text: str) -> bool:
    return bool(COMPLAINT_KEYWORDS & set(text.split()))


def tag_feature_request(text: str) -> bool:
    return bool(FEATURE_KEYWORDS & set(text.split()))


# ── 5. PIPELINE ───────────────────────────────────────────────────────────────

def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅  Loaded {len(df):,} reviews from {csv_path}")
    else:
        df = pd.DataFrame(SAMPLE_REVIEWS)
        print("ℹ️   Using built-in sample dataset (25 reviews).")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Require these columns (rename if your CSV uses different names)
    required = {"review", "rating", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Handle missing values
    df.dropna(subset=["review"], inplace=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # Text cleaning
    df["clean_review"] = df["review"].apply(clean_text)

    # Sentiment
    df["sentiment"] = df["review"].apply(get_sentiment)

    # Flags
    df["is_complaint"] = df["clean_review"].apply(tag_complaint)
    df["is_feature_request"] = df["clean_review"].apply(tag_feature_request)

    # Month column for trend analysis
    df["month"] = df["date"].dt.to_period("M").astype(str)

    print(f"✅  Preprocessed {len(df):,} reviews.")
    return df


def extract_top_keywords(df: pd.DataFrame, n: int = 15) -> pd.Series:
    all_words = " ".join(df["clean_review"]).split()
    freq = Counter(all_words)
    return pd.Series(freq).nlargest(n)


# ── 6. VISUALIZATIONS ────────────────────────────────────────────────────────

PALETTE = {
    "Positive": "#1D9E75",
    "Neutral":  "#888780",
    "Negative": "#D85A30",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_sentiment_pie(df: pd.DataFrame, out_dir: str) -> None:
    counts = df["sentiment"].value_counts()
    colors = [PALETTE.get(s, "#ccc") for s in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", fontsize=16, fontweight="bold", pad=18)
    plt.tight_layout()
    path = os.path.join(out_dir, "sentiment_pie.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Saved {path}")


def plot_keyword_bar(df: pd.DataFrame, out_dir: str, n: int = 12) -> None:
    kw = extract_top_keywords(df, n)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=kw.values, y=kw.index, palette="Blues_r", ax=ax)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(f"Top {n} Keywords in Reviews", fontsize=16, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(out_dir, "keyword_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Saved {path}")


def plot_sentiment_trend(df: pd.DataFrame, out_dir: str) -> None:
    trend = (
        df.groupby(["month", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    pivot = trend.pivot(index="month", columns="sentiment", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5))
    for sentiment, color in PALETTE.items():
        if sentiment in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[sentiment],
                marker="o",
                label=sentiment,
                color=color,
                linewidth=2.2,
            )
    ax.set_title("Sentiment Trend Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(title="Sentiment")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "sentiment_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Saved {path}")


def plot_rating_distribution(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["rating"].astype(int).value_counts().sort_index().plot(
        kind="bar", ax=ax, color="#378ADD", edgecolor="white", width=0.6
    )
    ax.set_title("Rating Distribution (1–5 Stars)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, "rating_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Saved {path}")


# ── 7. REPORT ─────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    total = len(df)
    sentiment_counts = df["sentiment"].value_counts()
    complaints = df[df["is_complaint"]]
    features   = df[df["is_feature_request"]]
    avg_rating = df["rating"].mean()

    separator = "─" * 58

    print(f"\n{'='*58}")
    print("  APP STORE REVIEW INSIGHTS REPORT")
    print(f"{'='*58}")
    print(f"  Total reviews analysed : {total}")
    print(f"  Average star rating    : {avg_rating:.2f} / 5.00")
    print(separator)

    print("\n📊  SENTIMENT BREAKDOWN")
    for s, c in sentiment_counts.items():
        pct = c / total * 100
        bar = "█" * int(pct / 4)
        print(f"  {s:<10} {bar:<20} {c:>3} ({pct:.1f}%)")

    print(f"\n{separator}")
    print(f"\n⚠️   COMMON COMPLAINTS  ({len(complaints)} reviews)")
    complaint_kw = extract_top_keywords(complaints, 8)
    for word, freq in complaint_kw.items():
        print(f"  • {word} ({freq}x)")

    print(f"\n{separator}")
    print(f"\n💡  FEATURE REQUESTS  ({len(features)} reviews)")
    for _, row in features.head(5).iterrows():
        print(f"  • [{row['rating']}★] {row['review'][:80]}...")

    print(f"\n{separator}")
    print("\n✅  Visualizations saved to ./outputs/")
    print(f"{'='*58}\n")


# ── 8. ENTRY POINT ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="App Store Review Insights Analyzer")
    parser.add_argument("--csv", type=str, default=None, help="Path to your CSV file")
    parser.add_argument("--out", type=str, default="outputs",  help="Output directory for charts")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n🚀  Starting App Store Review Insights Analyzer...\n")

    df = load_data(args.csv)
    df = preprocess(df)

    print("\n📈  Generating visualizations...")
    plot_sentiment_pie(df, args.out)
    plot_keyword_bar(df, args.out)
    plot_sentiment_trend(df, args.out)
    plot_rating_distribution(df, args.out)

    print_report(df)

    # Optionally save the enriched dataset
    enriched_path = os.path.join(args.out, "enriched_reviews.csv")
    df.to_csv(enriched_path, index=False)
    print(f"💾  Enriched dataset saved to {enriched_path}\n")


if __name__ == "__main__":
    main()
