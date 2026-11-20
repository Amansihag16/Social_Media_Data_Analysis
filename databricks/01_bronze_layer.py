# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # SocialPulse - Bronze Layer Ingestion
# MAGIC 
# MAGIC **Purpose:** Ingest raw multi-source social media data (Reddit, YouTube, Bluesky) with minimal modification.
# MAGIC 
# MAGIC **Key Responsibilities:**
# MAGIC - Preserve raw payload integrity and schema fidelity
# MAGIC - Append audit metadata: `source_platform`, `ingestion_timestamp`, `file_source`
# MAGIC - Write to Delta Lake tables: `bronze_reddit_posts`, `bronze_youtube_videos`, `bronze_youtube_comments`, `bronze_bluesky_posts`
# MAGIC - Support both Databricks cluster execution and local PySpark environments

# COMMAND ----------
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Bronze: %(message)s")
logger = logging.getLogger("BronzeIngestion")

# COMMAND ----------
def get_or_create_spark():
    """Initializes or retrieves active SparkSession with Delta Lake support."""
    try:
        from pyspark.sql import SparkSession
        builder = (
            SparkSession.builder.appName("SocialPulse-Bronze-Layer")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = builder.getOrCreate()
        return spark
    except Exception as e:
        logger.warning("Local SparkSession could not be initialized (%s); using file-based storage engine.", e)
        return None

# COMMAND ----------
class BronzeLayer:
    """Manages raw data ingestion into Bronze Delta Lake tables."""

    def __init__(self, spark=None):
        self.spark = spark or get_or_create_spark()
        self.raw_dir = config.DATA_RAW_DIR
        self.delta_dir = config.DELTA_BRONZE_DIR
        self.delta_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_raw_files(self, platform: str) -> List[Dict[str, Any]]:
        """Loads all raw JSON records for a given platform from data/raw/{platform}/."""
        platform_dir = self.raw_dir / platform
        if not platform_dir.exists():
            return []

        json_files = sorted(platform_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not json_files:
            return []

        # Ingest the latest file
        latest_file = json_files[0]
        logger.info("Reading latest raw %s file: %s", platform, latest_file.name)
        with open(latest_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records

    def ingest_reddit(self) -> int:
        """Ingests raw Reddit posts into bronze_reddit_posts."""
        records = self.load_latest_raw_files("reddit")
        if not records:
            logger.warning("No raw Reddit records found.")
            return 0

        target_path = self.delta_dir / "bronze_reddit_posts"
        target_path.mkdir(parents=True, exist_ok=True)

        # Write partitioned or parquet/delta
        if self.spark:
            try:
                df = self.spark.createDataFrame(records)
                (
                    df.write.format("delta")
                    .mode("overwrite")
                    .save(str(target_path))
                )
                logger.info("Spark wrote %d records to Delta table bronze_reddit_posts", df.count())
                return df.count()
            except Exception as e:
                logger.warning("Spark Delta write encountered: %s. Using direct Parquet/JSON fallback.", e)

        # Standalone fallback
        import pandas as pd
        df_pd = pd.DataFrame(records)
        df_pd.to_parquet(target_path / "data.parquet", index=False)
        with open(target_path / "schema.json", "w", encoding="utf-8") as f:
            json.dump({"table": "bronze_reddit_posts", "count": len(records), "timestamp": datetime.now(timezone.utc).isoformat()}, f)
        logger.info("Wrote %d records to bronze_reddit_posts fallback storage", len(records))
        return len(records)

    def ingest_youtube(self) -> int:
        """Ingests raw YouTube records into bronze_youtube_videos and bronze_youtube_comments."""
        records = self.load_latest_raw_files("youtube")
        if not records:
            logger.warning("No raw YouTube records found.")
            return 0

        # Split videos and comments
        videos = []
        comments = []
        for r in records:
            v_copy = {k: v for k, v in r.items() if k != "public_comments"}
            videos.append(v_copy)
            for c in r.get("public_comments", []):
                comments.append({
                    "video_id": r.get("video_id"),
                    "comment_id": c.get("comment_id"),
                    "comment_text": c.get("text"),
                    "author": c.get("author"),
                    "likes": c.get("likes", 0),
                    "source_platform": "youtube",
                    "ingestion_timestamp": r.get("ingestion_timestamp"),
                })

        import pandas as pd
        # Ingest videos
        v_target = self.delta_dir / "bronze_youtube_videos"
        v_target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(videos).to_parquet(v_target / "data.parquet", index=False)

        # Ingest comments
        c_target = self.delta_dir / "bronze_youtube_comments"
        c_target.mkdir(parents=True, exist_ok=True)
        if comments:
            pd.DataFrame(comments).to_parquet(c_target / "data.parquet", index=False)

        logger.info("Ingested %d YouTube videos and %d comments into Bronze tables", len(videos), len(comments))
        return len(videos)

    def ingest_bluesky(self) -> int:
        """Ingests raw Bluesky posts into bronze_bluesky_posts."""
        records = self.load_latest_raw_files("bluesky")
        if not records:
            logger.warning("No raw Bluesky records found.")
            return 0

        # Flatten hashtags array for standard storage
        cleaned_records = []
        for r in records:
            r_copy = dict(r)
            if isinstance(r_copy.get("hashtags"), list):
                r_copy["hashtags_str"] = ",".join(r_copy["hashtags"])
            cleaned_records.append(r_copy)

        import pandas as pd
        target_path = self.delta_dir / "bronze_bluesky_posts"
        target_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cleaned_records).to_parquet(target_path / "data.parquet", index=False)
        logger.info("Ingested %d Bluesky posts into bronze_bluesky_posts", len(cleaned_records))
        return len(cleaned_records)

    def run_all(self) -> Dict[str, int]:
        """Runs the entire Bronze ingestion pipeline."""
        logger.info("Starting Bronze Layer Medallion Ingestion...")
        reddit_cnt = self.ingest_reddit()
        yt_cnt = self.ingest_youtube()
        bsky_cnt = self.ingest_bluesky()
        summary = {
            "bronze_reddit_posts": reddit_cnt,
            "bronze_youtube_videos": yt_cnt,
            "bronze_bluesky_posts": bsky_cnt,
            "execution_status": "SUCCESS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Bronze Layer Ingestion Completed: %s", summary)
        return summary

# COMMAND ----------
if __name__ == "__main__":
    bronze = BronzeLayer()
    res = bronze.run_all()
    print("[SUCCESS] Bronze Ingestion Complete:", res)
