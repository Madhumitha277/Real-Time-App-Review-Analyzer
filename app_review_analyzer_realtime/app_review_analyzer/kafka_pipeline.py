"""
Approach 3: Kafka Streaming Pipeline (Production-Grade)
=======================================================
Producer scrapes reviews → publishes to Kafka topic.
Consumer subscribes → analyses sentiment → stores in DB.

Install:
    pip install confluent-kafka google-play-scraper textblob nltk sqlalchemy

Start Kafka locally (Docker):
    docker run -d -p 9092:9092 apache/kafka:3.7.0

Usage:
    # Terminal 1 — start consumer first
    python kafka_pipeline.py consumer

    # Terminal 2 — start producer
    python kafka_pipeline.py producer --app com.spotify.music
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone

import nltk
from confluent_kafka import Consumer, KafkaError, Producer
from google_play_scraper import Sort, reviews
from sqlalchemy import create_engine, text
from textblob import TextBlob

for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── KAFKA CONFIG ──────────────────────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
TOPIC        = "app-reviews"
GROUP_ID     = "review-analyzer"
DB_URL       = "sqlite:///kafka_reviews.db"

# ── DB SETUP ──────────────────────────────────────────────────────────────────
engine = create_engine(DB_URL)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS kafka_reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id   TEXT UNIQUE,
            app_id      TEXT,
            at          TEXT,
            score       REAL,
            content     TEXT,
            sentiment   TEXT,
            polarity    REAL,
            processed_at TEXT
        )
    """))

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_text(t: str) -> str:
    t = re.sub(r"[^a-z\s]", " ", str(t).lower())
    tokens = word_tokenize(t)
    return " ".join(x for x in tokens if x not in STOP_WORDS and len(x) > 2)


def get_sentiment(text: str):
    p = TextBlob(str(text)).sentiment.polarity
    label = "Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral")
    return label, round(p, 4)


# ── PRODUCER ──────────────────────────────────────────────────────────────────

def run_producer(app_id: str, interval: int = 60, count: int = 50):
    """
    Scrapes Google Play reviews and publishes each one
    as a JSON message to the Kafka topic.
    """
    producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    seen_ids: set = set()

    log.info(f"Producer started — app: {app_id}, interval: {interval}s")

    while True:
        log.info("Fetching reviews from Play Store...")
        try:
            result, _ = reviews(
                app_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=count,
            )
        except Exception as exc:
            log.error(f"Scrape error: {exc}")
            time.sleep(interval)
            continue

        published = 0
        for r in result:
            rid = r.get("reviewId", "")
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)

            payload = json.dumps({
                "review_id": rid,
                "app_id":    app_id,
                "at":        str(r.get("at", "")),
                "score":     r.get("score", 3),
                "content":   r.get("content", ""),
            })

            producer.produce(
                TOPIC,
                key=rid.encode(),
                value=payload.encode(),
                callback=lambda err, msg: (
                    log.warning(f"Delivery error: {err}") if err else None
                ),
            )
            published += 1

        producer.flush()
        log.info(f"  Published {published} new reviews to Kafka topic '{TOPIC}'")
        time.sleep(interval)


# ── CONSUMER ──────────────────────────────────────────────────────────────────

def run_consumer():
    """
    Subscribes to the Kafka topic, processes each message
    (sentiment analysis), and stores results in the DB.
    """
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id":           GROUP_ID,
        "auto.offset.reset":  "earliest",
    })
    consumer.subscribe([TOPIC])

    log.info(f"Consumer started — listening to topic '{TOPIC}'")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error(f"Kafka error: {msg.error()}")
                continue

            data = json.loads(msg.value().decode())
            content  = data.get("content", "")
            sentiment, polarity = get_sentiment(content)

            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT OR IGNORE INTO kafka_reviews
                    (review_id, app_id, at, score, content, sentiment, polarity, processed_at)
                    VALUES (:review_id, :app_id, :at, :score, :content, :sentiment, :polarity, :processed_at)
                """), {
                    **data,
                    "sentiment":    sentiment,
                    "polarity":     polarity,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                })

            log.info(
                f"  [{sentiment:>8}] score={data.get('score')}  "
                f"'{content[:60]}...'"
            )

    except KeyboardInterrupt:
        log.info("Consumer shutting down...")
    finally:
        consumer.close()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Review Pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prod = sub.add_parser("producer", help="Start the review producer")
    p_prod.add_argument("--app",      default="com.spotify.music")
    p_prod.add_argument("--interval", default=60, type=int)
    p_prod.add_argument("--count",    default=50,  type=int)

    sub.add_parser("consumer", help="Start the sentiment consumer")

    args = parser.parse_args()

    if args.mode == "producer":
        run_producer(args.app, args.interval, args.count)
    else:
        run_consumer()
