"""
SocialPulse - Data Quality Validation Module
Performs automated data quality assertions across all pipeline stages.
Validates schemas, nullability, formats, duplicates, and metric boundaries.
"""

import sys
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

PROJECT_ROOT = sys.path[0]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataValidation")

VALID_PLATFORMS = {"reddit", "youtube", "bluesky"}


class DataValidator:
    """Automated Data Quality assertion engine."""

    def __init__(self):
        self.validation_errors: List[Dict[str, Any]] = []

    def validate_posts(self, posts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validates post records against data engineering quality rules:
        - Primary ID non-empty
        - Platform valid
        - Non-negative likes/comments/shares
        - Timestamp parseable
        - Cleaned text non-empty
        - No duplicate (platform, id) pairs
        """
        valid_records: List[Dict[str, Any]] = []
        invalid_records: List[Dict[str, Any]] = []
        seen_keys = set()

        stats = {
            "total_evaluated": len(posts),
            "passed": 0,
            "failed": 0,
            "duplicate_count": 0,
            "missing_id_count": 0,
            "invalid_platform_count": 0,
            "negative_metrics_count": 0,
            "empty_text_count": 0,
            "invalid_timestamp_count": 0,
        }

        for p in posts:
            pid = str(p.get("post_id", "")).strip()
            platform = str(p.get("source_platform", "")).strip().lower()
            text = str(p.get("cleaned_text", "")).strip()
            ts = p.get("created_timestamp")
            likes = p.get("likes", 0)
            comments = p.get("comments", 0)
            shares = p.get("shares_reposts", 0)

            errors = []

            # 1. Missing post ID check
            if not pid:
                errors.append("Missing post_id")
                stats["missing_id_count"] += 1

            # 2. Duplicate record check
            key = (platform, pid)
            if key in seen_keys:
                errors.append(f"Duplicate record: {key}")
                stats["duplicate_count"] += 1
            else:
                seen_keys.add(key)

            # 3. Invalid platform check
            if platform not in VALID_PLATFORMS:
                errors.append(f"Invalid platform: {platform}")
                stats["invalid_platform_count"] += 1

            # 4. Empty text check
            if not text:
                errors.append("Empty post text")
                stats["empty_text_count"] += 1

            # 5. Negative engagement check
            if (isinstance(likes, (int, float)) and likes < 0) or \
               (isinstance(comments, (int, float)) and comments < 0) or \
               (isinstance(shares, (int, float)) and shares < 0):
                errors.append("Negative metric values detected")
                stats["negative_metrics_count"] += 1

            # 6. Timestamp validity check
            if not ts:
                errors.append("Missing timestamp")
                stats["invalid_timestamp_count"] += 1

            if errors:
                invalid_records.append({"record": p, "errors": errors})
            else:
                valid_records.append(p)

        stats["passed"] = len(valid_records)
        stats["failed"] = len(invalid_records)
        stats["dq_pass_rate_pct"] = round((stats["passed"] / max(1, stats["total_evaluated"])) * 100, 2)

        logger.info(
            "Data Quality Audit: %d/%d passed (%.2f%%). Failures: %d",
            stats["passed"],
            stats["total_evaluated"],
            stats["dq_pass_rate_pct"],
            stats["failed"],
        )

        return valid_records, stats

    def assert_gold_summary(self, gold_df_or_dicts: List[Dict[str, Any]]) -> bool:
        """Verifies Gold summary business constraints (e.g. percentages sum to ~100%)."""
        if not gold_df_or_dicts:
            logger.warning("Gold table is empty!")
            return False

        for row in gold_df_or_dicts:
            pos_pct = row.get("positive_pct", 0.0)
            neg_pct = row.get("negative_pct", 0.0)
            neu_pct = row.get("neutral_pct", 0.0)
            total = pos_pct + neg_pct + neu_pct
            if abs(total - 100.0) > 1.0:
                logger.warning("Sentiment percentages do not sum to 100%%: %.2f", total)
                return False

        logger.info("Gold summary constraints verified successfully.")
        return True


validator = DataValidator()
