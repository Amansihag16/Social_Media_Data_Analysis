# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # SocialPulse - Pipeline Validation and Health Check
# MAGIC 
# MAGIC **Purpose:** Perform comprehensive integrity verification across Bronze, Silver, and Gold layers.
# MAGIC Validates record counts, schema nullability, data quality compliance, and business metric sanity.

# COMMAND ----------
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from preprocessing.data_validation import validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Validation: %(message)s")
logger = logging.getLogger("PipelineValidation")

# COMMAND ----------
class PipelineValidator:
    """End-to-End Medallion Pipeline Auditor."""

    def __init__(self):
        self.bronze_dir = config.DELTA_BRONZE_DIR
        self.silver_dir = config.DELTA_SILVER_DIR
        self.gold_dir = config.DELTA_GOLD_DIR
        self.export_csv_dir = config.DATA_PROCESSED_DIR / "gold"

    def audit_layer(self, layer_name: str, base_path: Path, expected_tables: list) -> dict:
        """Audits table presence, row counts, and null constraints in a layer."""
        audit_results = {}
        for tbl in expected_tables:
            tbl_path = base_path / tbl
            parquet_file = tbl_path / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                null_counts = df.isnull().sum().to_dict()
                audit_results[tbl] = {
                    "status": "PASS",
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "null_counts": null_counts,
                }
            else:
                audit_results[tbl] = {
                    "status": "FAIL - NOT FOUND",
                    "row_count": 0,
                    "columns": [],
                    "null_counts": {},
                }
        return audit_results

    def run_audit(self) -> dict:
        """Runs complete multi-layer verification suite."""
        logger.info("Starting SocialPulse Pipeline Integrity Audit...")

        # 1. Bronze Audit
        bronze_tables = ["bronze_reddit_posts", "bronze_youtube_videos", "bronze_bluesky_posts"]
        bronze_audit = self.audit_layer("Bronze", self.bronze_dir, bronze_tables)

        # 2. Silver Audit
        silver_tables = ["silver_social_posts", "silver_social_comments", "silver_social_engagement", "silver_social_hashtags"]
        silver_audit = self.audit_layer("Silver", self.silver_dir, silver_tables)

        # 3. Gold Audit
        gold_tables = [
            "gold_sentiment_summary",
            "gold_platform_performance",
            "gold_hashtag_trends",
            "gold_daily_trends",
            "gold_top_posts",
            "gold_topic_analysis",
            "gold_customer_concerns",
        ]
        gold_audit = self.audit_layer("Gold", self.gold_dir, gold_tables)

        # 4. Check Business Logic Constraints on Gold
        sentiment_path = self.gold_dir / "gold_sentiment_summary" / "data.parquet"
        business_rules_passed = True
        if sentiment_path.exists():
            sent_df = pd.read_parquet(sentiment_path)
            business_rules_passed = validator.assert_gold_summary(sent_df.to_dict(orient="records"))

        # 5. Check CSV Exports for Power BI
        csv_files = list(self.export_csv_dir.glob("*.csv"))
        csv_export_count = len(csv_files)

        all_passed = (
            all(t["status"] == "PASS" for t in bronze_audit.values()) and
            all(t["status"] == "PASS" for t in silver_audit.values()) and
            all(t["status"] == "PASS" for t in gold_audit.values()) and
            business_rules_passed and
            csv_export_count >= 7
        )

        overall_status = "HEALTHY (ALL CHECKS PASSED)" if all_passed else "ATTENTION NEEDED"

        report = {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bronze_tables": bronze_audit,
            "silver_tables": silver_audit,
            "gold_tables": gold_audit,
            "business_rules_passed": business_rules_passed,
            "powerbi_ready_csv_exports": csv_export_count,
        }

        # Format ASCII Summary
        print("\n" + "=" * 70)
        print("          SOCIALPULSE PIPELINE HEALTH & INTEGRITY REPORT          ")
        print("=" * 70)
        print(f"Overall Status: {overall_status}")
        print(f"Audit Timestamp: {report['timestamp']}\n")

        print("--- BRONZE LAYER ---")
        for k, v in bronze_audit.items():
            print(f"  [x] {k:<28}: {v['row_count']:>5} records  ({v['status']})")

        print("\n--- SILVER LAYER ---")
        for k, v in silver_audit.items():
            print(f"  [x] {k:<28}: {v['row_count']:>5} records  ({v['status']})")

        print("\n--- GOLD LAYER (ANALYTICS & POWER BI READY) ---")
        for k, v in gold_audit.items():
            print(f"  [x] {k:<28}: {v['row_count']:>5} records  ({v['status']})")

        print(f"\nPower BI CSV Datasets Generated: {csv_export_count} files in {self.export_csv_dir}")
        print("=" * 70 + "\n")

        return report

# COMMAND ----------
if __name__ == "__main__":
    auditor = PipelineValidator()
    report = auditor.run_audit()
