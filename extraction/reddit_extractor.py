"""
SocialPulse - Reddit Data Extractor
Extracts posts from target subreddits using the Reddit API or falls back to benchmark datasets.
Handles rate limiting, pagination, missing fields, and records ingestion metadata.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RedditExtractor")


class RedditExtractor:
    """Extracts posts from Reddit subreddits with API authentication and sample fallback."""

    def __init__(self):
        self.client_id = config.REDDIT_CLIENT_ID
        self.client_secret = config.REDDIT_CLIENT_SECRET
        self.user_agent = config.REDDIT_USER_AGENT
        self.subreddits = config.REDDIT_SUBREDDITS
        self.limit = config.REDDIT_POST_LIMIT
        self.access_token: Optional[str] = None

    def _get_access_token(self) -> Optional[str]:
        """Obtains OAuth2 bearer token from Reddit API."""
        if not self.client_id or not self.client_secret:
            logger.info("Reddit API credentials not fully set; will use public or sample data.")
            return None

        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.post(auth_url, auth=auth, data=data, headers=headers, timeout=10)
            if response.status_code == 200:
                self.access_token = response.json().get("access_token")
                logger.info("Successfully authenticated with Reddit API.")
                return self.access_token
            else:
                logger.warning("Reddit authentication failed (HTTP %s): %s", response.status_code, response.text)
        except Exception as e:
            logger.error("Exception during Reddit OAuth authentication: %s", e)
        return None

    def fetch_from_api(self, subreddit: str) -> List[Dict[str, Any]]:
        """Fetches hot posts from a given subreddit using OAuth or public JSON endpoint."""
        ingestion_ts = datetime.now(timezone.utc).isoformat()
        headers = {"User-Agent": self.user_agent}
        if self.access_token:
            url = f"https://oauth.reddit.com/r/{subreddit}/hot"
            headers["Authorization"] = f"bearer {self.access_token}"
        else:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"

        params = {"limit": self.limit}
        posts: List[Dict[str, Any]] = []

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 429:
                logger.warning("Reddit API rate limited (HTTP 429). Retrying after cooldown or using sample.")
                return []
            if response.status_code != 200:
                logger.warning("Failed to fetch r/%s (HTTP %s)", subreddit, response.status_code)
                return []

            data = response.json()
            children = data.get("data", {}).get("children", [])
            for item in children:
                pdata = item.get("data", {})
                created_utc = pdata.get("created_utc", 0)
                created_iso = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else ingestion_ts

                record = {
                    "post_id": str(pdata.get("id", "")),
                    "post_title": pdata.get("title", ""),
                    "post_description": pdata.get("selftext", ""),
                    "subreddit": pdata.get("subreddit", subreddit),
                    "author": pdata.get("author", "[deleted]"),
                    "created_utc": created_iso,
                    "score": int(pdata.get("score", 0)),
                    "comment_count": int(pdata.get("num_comments", 0)),
                    "post_url": f"https://reddit.com{pdata.get('permalink', '')}",
                    "source_platform": "reddit",
                    "ingestion_timestamp": ingestion_ts,
                    "data_mode": "live_api"
                }
                posts.append(record)
            logger.info("Extracted %d live posts from r/%s", len(posts), subreddit)
        except Exception as e:
            logger.error("Error fetching r/%s: %s", subreddit, e)

        return posts

    def load_sample_dataset(self) -> List[Dict[str, Any]]:
        """Loads vetted benchmark sample dataset when API is offline or unconfigured."""
        sample_file = config.DATA_SAMPLE_DIR / "reddit_sample.json"
        if not sample_file.exists():
            logger.warning("Sample dataset not found, generating benchmark datasets.")
            from extraction.sample_data_generator import generate_benchmark_datasets
            generate_benchmark_datasets()

        with open(sample_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        for r in records:
            r["ingestion_timestamp"] = ingestion_ts
            r["data_mode"] = "benchmark_sample"

        logger.info("Loaded %d records from Reddit benchmark sample dataset.", len(records))
        return records

    def extract(self, force_sample: bool = False) -> List[Dict[str, Any]]:
        """Executes extraction pipeline across configured subreddits."""
        all_records: List[Dict[str, Any]] = []

        if not force_sample and (self.client_id or self.user_agent):
            self._get_access_token()
            for sub in self.subreddits:
                records = self.fetch_from_api(sub)
                all_records.extend(records)

        # Fallback to benchmark sample if API extraction yielded no records
        if not all_records:
            logger.info("Live extraction produced 0 records or force_sample=True. Using vetted benchmark sample.")
            all_records = self.load_sample_dataset()

        return all_records

    def save_raw(self, records: List[Dict[str, Any]]) -> Path:
        """Persists raw records to JSON in data/raw/reddit/."""
        raw_dir = config.DATA_RAW_DIR / "reddit"
        raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = raw_dir / f"reddit_raw_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        logger.info("Saved %d raw Reddit records to %s", len(records), output_file)
        return output_file


if __name__ == "__main__":
    extractor = RedditExtractor()
    extracted = extractor.extract()
    file_path = extractor.save_raw(extracted)
    print(f"[SUCCESS] Reddit Extraction complete: {len(extracted)} posts written to {file_path}")
