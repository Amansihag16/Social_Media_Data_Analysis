"""
SocialPulse - YouTube Data Extractor
Extracts videos and top public comments using the YouTube Data API v3.
Includes robust quota management, error handling, and realistic benchmark dataset fallback.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("YouTubeExtractor")


class YouTubeExtractor:
    """Extracts videos and comments from YouTube Data API v3 with quota fallback."""

    def __init__(self):
        self.api_key = config.YOUTUBE_API_KEY
        self.queries = config.YOUTUBE_SEARCH_QUERIES
        self.max_results = config.YOUTUBE_MAX_RESULTS
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def fetch_from_api(self, query: str) -> List[Dict[str, Any]]:
        """Fetches videos matching query from YouTube Data API v3."""
        if not self.api_key:
            logger.info("YouTube API key not configured; skipping live API query for '%s'", query)
            return []

        search_url = f"{self.base_url}/search"
        search_params = {
            "key": self.api_key,
            "q": query,
            "part": "snippet",
            "type": "video",
            "maxResults": min(self.max_results, 25),
            "order": "relevance",
        }

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        try:
            res = requests.get(search_url, params=search_params, timeout=15)
            if res.status_code == 403:
                logger.warning("YouTube API quota exceeded or forbidden (HTTP 403). Falling back to sample.")
                return []
            if res.status_code != 200:
                logger.warning("YouTube API search failed (HTTP %s): %s", res.status_code, res.text)
                return []

            search_data = res.json()
            items = search_data.get("items", [])
            video_ids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]

            if not video_ids:
                return []

            # Retrieve video statistics
            stats_url = f"{self.base_url}/videos"
            stats_params = {
                "key": self.api_key,
                "id": ",".join(video_ids),
                "part": "snippet,statistics",
            }
            stats_res = requests.get(stats_url, params=stats_params, timeout=15)
            if stats_res.status_code != 200:
                logger.warning("Failed to retrieve YouTube video stats (HTTP %s)", stats_res.status_code)
                return []

            video_items = stats_res.json().get("items", [])
            for v in video_items:
                vid = v.get("id", "")
                snippet = v.get("snippet", {})
                statistics = v.get("statistics", {})

                # Fetch comments where available
                comments = self._fetch_video_comments(vid)

                record = {
                    "video_id": vid,
                    "video_title": snippet.get("title", ""),
                    "video_description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ingestion_ts),
                    "view_count": int(statistics.get("viewCount", 0)),
                    "like_count": int(statistics.get("likeCount", 0)),
                    "comment_count": int(statistics.get("commentCount", 0)),
                    "public_comments": comments,
                    "source_platform": "youtube",
                    "ingestion_timestamp": ingestion_ts,
                    "data_mode": "live_api"
                }
                records.append(record)

            logger.info("Extracted %d YouTube videos for query '%s'", len(records), query)
        except Exception as e:
            logger.error("Exception during YouTube extraction for '%s': %s", query, e)

        return records

    def _fetch_video_comments(self, video_id: str, max_comments: int = 5) -> List[Dict[str, Any]]:
        """Fetches top public comments for a specific video."""
        comments: List[Dict[str, Any]] = []
        if not self.api_key:
            return comments

        comments_url = f"{self.base_url}/commentThreads"
        params = {
            "key": self.api_key,
            "videoId": video_id,
            "part": "snippet",
            "maxResults": max_comments,
            "textFormat": "plainText",
        }
        try:
            res = requests.get(comments_url, params=params, timeout=10)
            if res.status_code == 200:
                for item in res.json().get("items", []):
                    top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    comments.append({
                        "comment_id": item.get("id", ""),
                        "text": top.get("textDisplay", ""),
                        "author": top.get("authorDisplayName", "Anonymous"),
                        "likes": int(top.get("likeCount", 0))
                    })
        except Exception:
            # Comments may be disabled on video
            pass
        return comments

    def load_sample_dataset(self) -> List[Dict[str, Any]]:
        """Loads vetted YouTube benchmark sample dataset."""
        sample_file = config.DATA_SAMPLE_DIR / "youtube_sample.json"
        if not sample_file.exists():
            from extraction.sample_data_generator import generate_benchmark_datasets
            generate_benchmark_datasets()

        with open(sample_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        for r in records:
            r["ingestion_timestamp"] = ingestion_ts
            r["data_mode"] = "benchmark_sample"

        logger.info("Loaded %d records from YouTube benchmark sample dataset.", len(records))
        return records

    def extract(self, force_sample: bool = False) -> List[Dict[str, Any]]:
        """Executes extraction pipeline across configured search queries."""
        all_records: List[Dict[str, Any]] = []

        if not force_sample and self.api_key:
            for q in self.queries:
                records = self.fetch_from_api(q)
                all_records.extend(records)

        if not all_records:
            logger.info("Live extraction yielded 0 records or force_sample=True. Using vetted benchmark sample.")
            all_records = self.load_sample_dataset()

        return all_records

    def save_raw(self, records: List[Dict[str, Any]]) -> Path:
        """Persists raw records to JSON in data/raw/youtube/."""
        raw_dir = config.DATA_RAW_DIR / "youtube"
        raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = raw_dir / f"youtube_raw_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        logger.info("Saved %d raw YouTube records to %s", len(records), output_file)
        return output_file


if __name__ == "__main__":
    extractor = YouTubeExtractor()
    extracted = extractor.extract()
    file_path = extractor.save_raw(extracted)
    print(f"[SUCCESS] YouTube Extraction complete: {len(extracted)} videos written to {file_path}")
