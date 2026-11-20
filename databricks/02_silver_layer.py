# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # SocialPulse - Silver Layer Transformation & Cleaning
# MAGIC 
# MAGIC **Purpose:** Clean, standardize, deduplicate, and validate raw Bronze records. Enrich records with NLP sentiment polarity and engagement scoring.
# MAGIC 
# MAGIC **Output Silver Delta Tables:**
# MAGIC 1. `silver_social_posts`: Standardized posts across all three platforms with sentiment and clean text.
# MAGIC 2. `silver_social_comments`: Standardized user comments.
# MAGIC 3. `silver_social_engagement`: Calculated engagement scores and performance metrics.
# MAGIC 4. `silver_social_hashtags`: Normalized exploded hashtags with sentiment and platform associations.

# COMMAND ----------
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from preprocessing.data_cleaning import cleaner
from preprocessing.data_validation import validator
from sentiment.sentiment_analysis import sentiment_engine
from analytics.engagement_analysis import engagement_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Silver: %(message)s")
logger = logging.getLogger("SilverTransformation")

# COMMAND ----------
class SilverLayer:
    """Manages Silver Layer transformation, validation, and Delta Lake storage."""

    def __init__(self):
        self.bronze_dir = config.DELTA_BRONZE_DIR
        self.silver_dir = config.DELTA_SILVER_DIR
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    def read_bronze_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Reads records from bronze table (Parquet/Delta)."""
        table_path = self.bronze_dir / table_name
        parquet_file = table_path / "data.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            return df.to_dict(orient="records")
        return []

    def transform_and_enrich(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Executes full cleaning, validation, sentiment analysis, and feature enrichment."""
        # 1. Read Bronze tables
        reddit_raw = self.read_bronze_table("bronze_reddit_posts")
        youtube_raw = self.read_bronze_table("bronze_youtube_videos")
        youtube_c_raw = self.read_bronze_table("bronze_youtube_comments")
        bluesky_raw = self.read_bronze_table("bronze_bluesky_posts")

        # 2. Clean and standardize each source
        reddit_posts, _ = cleaner.clean_reddit_records(reddit_raw)
        yt_posts, _ = cleaner.clean_youtube_records(youtube_raw)
        bsky_posts, _ = cleaner.clean_bluesky_records(bluesky_raw)

        # 3. Combine and Deduplicate posts
        combined_posts = cleaner.deduplicate(reddit_posts + yt_posts + bsky_posts, id_key="post_id")
        logger.info("Combined raw posts: %d. After deduplication: %d", len(reddit_posts) + len(yt_posts) + len(bsky_posts), len(combined_posts))

        # 4. Data Quality Validation
        valid_posts, dq_stats = validator.validate_posts(combined_posts)
        logger.info("Silver Data Quality Assertions: Passed %d, Failed %d (Pass Rate: %s%%)", dq_stats["passed"], dq_stats["failed"], dq_stats["dq_pass_rate_pct"])

        # 5. Sentiment Analysis (VADER)
        enriched_posts = sentiment_engine.enrich_records(valid_posts)

        # 6. Engagement Metrics
        final_posts = engagement_engine.enrich_records(enriched_posts)

        # 7. Build Silver Engagement Table
        engagement_records = []
        for p in final_posts:
            engagement_records.append({
                "post_id": p["post_id"],
                "source_platform": p["source_platform"],
                "likes": p["likes"],
                "comments": p["comments"],
                "shares_reposts": p["shares_reposts"],
                "views": p["views"],
                "raw_engagement": p["raw_engagement"],
                "weighted_engagement": p["weighted_engagement"],
                "engagement_score": p["engagement_score"],
                "created_date": p["created_date"],
            })

        # 8. Build Silver Hashtags Table
        hashtag_records = []
        for p in final_posts:
            for tag in p.get("hashtags", []):
                hashtag_records.append({
                    "post_id": p["post_id"],
                    "hashtag": f"#{tag.lower().strip()}",
                    "source_platform": p["source_platform"],
                    "sentiment_label": p["sentiment_label"],
                    "engagement_score": p["engagement_score"],
                    "created_date": p["created_date"],
                })

        # 9. Clean and enrich comments
        enriched_comments = []
        for c in youtube_c_raw:
            c_dict = dict(c)
            c_res = sentiment_engine.analyze_text(c_dict.get("comment_text", ""))
            c_dict["sentiment_compound"] = c_res["compound"]
            c_dict["sentiment_label"] = c_res["sentiment_label"]
            enriched_comments.append(c_dict)

        return final_posts, enriched_comments, engagement_records, hashtag_records

    def write_silver_tables(self, posts, comments, engagement, hashtags) -> Dict[str, int]:
        """Saves enriched records to Silver Delta/Parquet tables."""
        targets = {
            "silver_social_posts": posts,
            "silver_social_comments": comments,
            "silver_social_engagement": engagement,
            "silver_social_hashtags": hashtags,
        }

        counts = {}
        for tbl_name, records in targets.items():
            tbl_path = self.silver_dir / tbl_name
            tbl_path.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(records)
            df.to_parquet(tbl_path / "data.parquet", index=False)
            df.to_csv(tbl_path / "data.csv", index=False)
            counts[tbl_name] = len(records)
            logger.info("Successfully persisted Silver table '%s' with %d records.", tbl_name, len(records))

        return counts

    def run_all(self) -> Dict[str, Any]:
        """Runs the entire Silver layer transformation workflow."""
        logger.info("Starting Silver Layer Transformation & Cleaning Pipeline...")
        posts, comments, eng, tags = self.transform_and_enrich()
        counts = self.write_silver_tables(posts, comments, eng, tags)
        summary = {
            "status": "SUCCESS",
            "table_counts": counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Silver Layer Transformation complete: %s", summary)
        return summary

# COMMAND ----------
if __name__ == "__main__":
    silver = SilverLayer()
    res = silver.run_all()
    print("[SUCCESS] Silver Transformation Complete:", res)
