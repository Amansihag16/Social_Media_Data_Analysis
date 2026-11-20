# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # SocialPulse - Gold Layer Business Analytics Aggregations
# MAGIC 
# MAGIC **Purpose:** Aggregate Silver data into 7 business-ready analytical datasets tailored for the Power BI Executive Dashboard.
# MAGIC 
# MAGIC **Output Gold Analytical Tables:**
# MAGIC 1. `gold_sentiment_summary`: Cross-platform sentiment distribution, percentages, and counts.
# MAGIC 2. `gold_platform_performance`: Platform-wise total interactions, engagement, and averages.
# MAGIC 3. `gold_hashtag_trends`: Hashtag volume, engagement, and temporal trends.
# MAGIC 4. `gold_daily_trends`: Time-series sentiment and daily engagement tracking.
# MAGIC 5. `gold_top_posts`: Top engaging posts with direct URLs and sentiment labels.
# MAGIC 6. `gold_topic_analysis`: Key discussion topics and sentiment distribution.
# MAGIC 7. `gold_customer_concerns`: Recurring complaints, taxonomy categorization, and negative impact.

# COMMAND ----------
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from analytics.engagement_analysis import engagement_engine
from analytics.hashtag_analysis import hashtag_engine
from analytics.topic_analysis import topic_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Gold: %(message)s")
logger = logging.getLogger("GoldAggregations")

# COMMAND ----------
class GoldLayer:
    """Computes Gold analytical models and exports Delta/Parquet/CSV for Power BI."""

    def __init__(self):
        self.silver_dir = config.DELTA_SILVER_DIR
        self.gold_dir = config.DELTA_GOLD_DIR
        self.export_dir = config.DATA_PROCESSED_DIR / "gold"
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def read_silver_posts(self) -> List[Dict[str, Any]]:
        """Reads silver_social_posts table."""
        path = self.silver_dir / "silver_social_posts" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required Silver table not found at {path}. Run 02_silver_layer first.")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def build_sentiment_summary(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates platform-wise and overall sentiment distribution."""
        platforms = sorted(list(set(p.get("source_platform", "unknown") for p in posts)))
        summary_rows = []

        # 1. Platform-wise breakdown
        for plat in platforms:
            plat_posts = [p for p in posts if p.get("source_platform") == plat]
            tot = len(plat_posts)
            pos = sum(1 for p in plat_posts if p.get("sentiment_label") == "Positive")
            neg = sum(1 for p in plat_posts if p.get("sentiment_label") == "Negative")
            neu = sum(1 for p in plat_posts if p.get("sentiment_label") == "Neutral")

            pos_pct = round((pos / max(1, tot)) * 100, 2)
            neg_pct = round((neg / max(1, tot)) * 100, 2)
            neu_pct = round((neu / max(1, tot)) * 100, 2)
            nss = round(pos_pct - neg_pct, 2)  # Net Sentiment Score

            summary_rows.append({
                "platform": plat,
                "total_posts": tot,
                "positive_posts": pos,
                "negative_posts": neg,
                "neutral_posts": neu,
                "positive_pct": pos_pct,
                "negative_pct": neg_pct,
                "neutral_pct": neu_pct,
                "net_sentiment_score": nss,
            })

        # 2. Overall portfolio summary
        tot_all = len(posts)
        pos_all = sum(1 for p in posts if p.get("sentiment_label") == "Positive")
        neg_all = sum(1 for p in posts if p.get("sentiment_label") == "Negative")
        neu_all = sum(1 for p in posts if p.get("sentiment_label") == "Neutral")

        pos_all_pct = round((pos_all / max(1, tot_all)) * 100, 2)
        neg_all_pct = round((neg_all / max(1, tot_all)) * 100, 2)
        neu_all_pct = round((neu_all / max(1, tot_all)) * 100, 2)

        summary_rows.append({
            "platform": "ALL_PLATFORMS",
            "total_posts": tot_all,
            "positive_posts": pos_all,
            "negative_posts": neg_all,
            "neutral_posts": neu_all,
            "positive_pct": pos_all_pct,
            "negative_pct": neg_all_pct,
            "neutral_pct": neu_all_pct,
            "net_sentiment_score": round(pos_all_pct - neg_all_pct, 2),
        })

        return summary_rows

    def build_platform_performance(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates platform interaction metrics."""
        return engagement_engine.summarize_platform_performance(posts)

    def build_hashtag_trends(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates hashtag volume, date-wise velocity, and engagement."""
        return hashtag_engine.analyze_hashtags(posts)

    def build_daily_trends(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates daily post volume, sentiment breakdown, and engagement."""
        dates = sorted(list(set(p.get("created_date", "") for p in posts if p.get("created_date"))))
        trends = []

        for d in dates:
            d_posts = [p for p in posts if p.get("created_date") == d]
            tot = len(d_posts)
            pos = sum(1 for p in d_posts if p.get("sentiment_label") == "Positive")
            neg = sum(1 for p in d_posts if p.get("sentiment_label") == "Negative")
            neu = sum(1 for p in d_posts if p.get("sentiment_label") == "Neutral")
            eng = sum(int(p.get("engagement_score", 0)) for p in d_posts)

            trends.append({
                "date": d,
                "total_posts": tot,
                "positive_posts": pos,
                "negative_posts": neg,
                "neutral_posts": neu,
                "daily_engagement": eng,
                "avg_daily_engagement": round(eng / max(1, tot), 2),
            })

        return trends

    def build_top_posts(self, posts: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """Extracts highest engaging social media posts."""
        sorted_posts = sorted(posts, key=lambda x: int(x.get("engagement_score", 0)), reverse=True)[:top_n]
        top = []
        for p in sorted_posts:
            top.append({
                "post_id": p.get("post_id"),
                "platform": p.get("source_platform"),
                "author": p.get("author"),
                "post_title": p.get("content_title") or p.get("cleaned_text")[:80],
                "sentiment_label": p.get("sentiment_label"),
                "sentiment_compound": p.get("sentiment_compound", 0.0),
                "likes": p.get("likes", 0),
                "comments": p.get("comments", 0),
                "shares_reposts": p.get("shares_reposts", 0),
                "views": p.get("views", 0),
                "engagement_score": p.get("engagement_score", 0),
                "published_date": p.get("created_date"),
                "post_url": p.get("post_url", ""),
            })
        return top

    def build_topic_analysis(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mines top discussion topics and sentiment distribution."""
        return topic_engine.extract_topics(posts, top_n_per_platform=12)

    def build_customer_concerns(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts customer complaints, bugs, and friction categories."""
        return topic_engine.extract_customer_concerns(posts)

    def write_table(self, table_name: str, records: List[Dict[str, Any]]):
        """Persists Gold tables as Delta/Parquet and CSV for Power BI Desktop."""
        df = pd.DataFrame(records)

        # Delta/Parquet directory
        delta_path = self.gold_dir / table_name
        delta_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(delta_path / "data.parquet", index=False)

        # CSV Export directory for direct Power BI drag-and-drop
        csv_file = self.export_dir / f"{table_name}.csv"
        df.to_csv(csv_file, index=False)
        logger.info("Saved Gold table '%s': %d rows to %s and %s", table_name, len(records), delta_path.name, csv_file.name)

    def run_all(self) -> Dict[str, int]:
        """Executes generation of all 7 Gold analytical tables."""
        logger.info("Starting Gold Layer Business Analytics Pipeline...")
        posts = self.read_silver_posts()

        tables = {
            "gold_sentiment_summary": self.build_sentiment_summary(posts),
            "gold_platform_performance": self.build_platform_performance(posts),
            "gold_hashtag_trends": self.build_hashtag_trends(posts),
            "gold_daily_trends": self.build_daily_trends(posts),
            "gold_top_posts": self.build_top_posts(posts),
            "gold_topic_analysis": self.build_topic_analysis(posts),
            "gold_customer_concerns": self.build_customer_concerns(posts),
        }

        counts = {}
        for tbl_name, data in tables.items():
            self.write_table(tbl_name, data)
            counts[tbl_name] = len(data)

        summary = {
            "status": "SUCCESS",
            "gold_tables_generated": counts,
            "export_csv_directory": str(self.export_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Gold Layer Processing Completed: %s", summary)
        return summary

# COMMAND ----------
if __name__ == "__main__":
    gold = GoldLayer()
    res = gold.run_all()
    print("[SUCCESS] Gold Layer Pipeline Complete:", res)
