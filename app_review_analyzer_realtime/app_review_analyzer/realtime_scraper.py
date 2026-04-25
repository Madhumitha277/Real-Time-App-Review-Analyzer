"""
Approach 1: Real-Time Reviews via google-play-scraper + APScheduler
=====================================================================
Scrapes new reviews from Google Play Store on a schedule,
runs sentiment analysis, and stores results incrementally.

Install:
    pip install google-play-scraper apscheduler textblob nltk pandas

Usage:
    python realtime_scraper.py
    python realtime_scraper.py --app com.spotify.music --interval 30
"""

import argparse
import logging
import re
import sqlite3
import time
from datetime import datetime

import nltk
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from google_play_scraper import Sort, reviews
from textblob import TextBlob

# ── NLTK ──────────────────────────────────────────────────────────────────────
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_APP_ID    = "com.spotify.music"   # any Play Store app ID
DEFAULT_INTERVAL  = 60                    # seconds between scrapes
DB_PATH           = "reviews.db"


# ── DATABASE SETUP ────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id   TEXT PRIMARY KEY,
            app_id      TEXT,
            at          TEXT,
            score       INTEGER,
            content     TEXT,
            clean       TEXT,
            sentiment   TEXT,
            polarity    REAL,
            fetched_at  TEXT
        )
    """)
    conn.commit()
    return conn


# ── TEXT & SENTIMENT ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)


def get_sentiment(text: str) -> tuple[str, float]:
    polarity = TextBlob(str(text)).sentiment.polarity
    label = "Positive" if polarity > 0.1 else ("Negative" if polarity < -0.1 else "Neutral")
    return label, round(polarity, 4)


# ── SCRAPE JOB ────────────────────────────────────────────────────────────────

def scrape_and_store(app_id: str, count: int = 100):
    """Fetch latest reviews, analyse sentiment, upsert to SQLite."""
    log.info(f"Scraping {app_id} ...")
    try:
        result, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=count,
        )
    except Exception as exc:
        log.error(f"Scrape failed: {exc}")
        return

    conn = get_db()
    new_count = 0

    for r in result:
        review_id = r.get("reviewId", "")
        content   = r.get("content", "")
        score     = r.get("score", 3)
        at        = str(r.get("at", ""))

        clean     = clean_text(content)
        sentiment, polarity = get_sentiment(content)
        fetched   = datetime.utcnow().isoformat()

        try:
            conn.execute(
                """INSERT OR IGNORE INTO reviews
                   (review_id, app_id, at, score, content, clean, sentiment, polarity, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (review_id, app_id, at, score, content, clean, sentiment, polarity, fetched),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                new_count += 1
        except Exception as exc:
            log.warning(f"DB insert error: {exc}")

    conn.commit()
    conn.close()

    log.info(f"  → {new_count} new reviews stored (total scraped: {len(result)})")
    print_live_stats()


def print_live_stats():
    """Print a quick sentiment breakdown from the DB."""
    conn = get_db()
    df = pd.read_sql("SELECT sentiment, COUNT(*) as n FROM reviews GROUP BY sentiment", conn)
    conn.close()
    total = df["n"].sum()
    print(f"\n  ── Live stats (total: {total}) ──")
    for _, row in df.iterrows():
        bar = "█" * int(row["n"] / max(total, 1) * 30)
        print(f"  {row['sentiment']:<10} {bar} {row['n']}")
    print()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app",      default=DEFAULT_APP_ID,  help="Google Play app ID")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, type=int, help="Seconds between scrapes")
    parser.add_argument("--count",    default=100, type=int,    help="Reviews to fetch per run")
    args = parser.parse_args()

    get_db()   # initialise DB

    log.info(f"Starting real-time scraper — app: {args.app}, interval: {args.interval}s")
    log.info("Press Ctrl+C to stop.\n")

    # Run once immediately, then schedule
    scrape_and_store(args.app, args.count)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        scrape_and_store,
        "interval",
        seconds=args.interval,
        args=[args.app, args.count],
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")
