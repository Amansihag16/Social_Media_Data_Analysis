"""
SocialPulse - Data Cleaning and Standardization Module
Transforms heterogeneous raw records from Reddit, YouTube, and Bluesky
into standardized, deduplicated, and enriched Silver schema records.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import dateutil.parser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.text_preprocessing import preprocessor


class DataCleaner:
    """Standardizes, cleans, and deduplicates multi-source social media data."""

    @staticmethod
    def parse_timestamp(ts: Any) -> Tuple[str, str]:
        """
        Parses various timestamp representations into (iso_timestamp, date_str).
        Defaults to current UTC if invalid.
        """
        now = datetime.now(timezone.utc)
        default_iso = now.isoformat()
        default_date = now.strftime("%Y-%m-%d")

        if not ts:
            return default_iso, default_date

        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = dateutil.parser.parse(str(ts))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
            return dt.isoformat(), dt.strftime("%Y-%m-%d")
        except Exception:
            return default_iso, default_date

    def clean_reddit_records(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Transforms raw Reddit records into standardized posts and comments."""
        posts: List[Dict[str, Any]] = []
        comments: List[Dict[str, Any]] = []

        for r in raw_records:
            pid = str(r.get("post_id", "")).strip()
            if not pid:
                continue

            title = str(r.get("post_title", "")).strip()
            desc = str(r.get("post_description", "")).strip()
            combined_text = f"{title}. {desc}".strip() if desc else title

            clean_text = preprocessor.clean_text_for_sentiment(combined_text)
            nlp_text = preprocessor.clean_text_for_nlp(combined_text)
            hashtags = preprocessor.extract_hashtags(combined_text)

            created_iso, created_date = self.parse_timestamp(r.get("created_utc"))
            score = max(0, int(r.get("score", 0)))
            comment_count = max(0, int(r.get("comment_count", 0)))

            post_entry = {
                "post_id": pid,
                "source_platform": "reddit",
                "author": str(r.get("author", "[deleted]")),
                "content_title": title,
                "content_body": desc,
                "cleaned_text": clean_text,
                "nlp_tokens": nlp_text,
                "hashtags": hashtags,
                "created_timestamp": created_iso,
                "created_date": created_date,
                "likes": score,  # Upvotes mapped to likes
                "comments": comment_count,
                "shares_reposts": 0,  # Reddit does not supply native share counts via standard API
                "views": 0,
                "post_url": str(r.get("post_url", "")),
                "ingestion_timestamp": str(r.get("ingestion_timestamp", datetime.now(timezone.utc).isoformat())),
            }
            posts.append(post_entry)

        return posts, comments

    def clean_youtube_records(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Transforms raw YouTube records into standardized posts (videos) and comments."""
        posts: List[Dict[str, Any]] = []
        comments: List[Dict[str, Any]] = []

        for r in raw_records:
            vid = str(r.get("video_id", "")).strip()
            if not vid:
                continue

            title = str(r.get("video_title", "")).strip()
            desc = str(r.get("video_description", "")).strip()
            combined_text = f"{title}. {desc}".strip()

            clean_text = preprocessor.clean_text_for_sentiment(combined_text)
            nlp_text = preprocessor.clean_text_for_nlp(combined_text)
            hashtags = preprocessor.extract_hashtags(combined_text)

            created_iso, created_date = self.parse_timestamp(r.get("published_at"))
            views = max(0, int(r.get("view_count", 0)))
            likes = max(0, int(r.get("like_count", 0)))
            comment_count = max(0, int(r.get("comment_count", 0)))

            post_entry = {
                "post_id": vid,
                "source_platform": "youtube",
                "author": str(r.get("channel_title", "UnknownChannel")),
                "content_title": title,
                "content_body": desc,
                "cleaned_text": clean_text,
                "nlp_tokens": nlp_text,
                "hashtags": hashtags,
                "created_timestamp": created_iso,
                "created_date": created_date,
                "likes": likes,
                "comments": comment_count,
                "shares_reposts": 0,
                "views": views,
                "post_url": f"https://www.youtube.com/watch?v={vid}",
                "ingestion_timestamp": str(r.get("ingestion_timestamp", datetime.now(timezone.utc).isoformat())),
            }
            posts.append(post_entry)

            # Process public comments
            for c in r.get("public_comments", []):
                cid = str(c.get("comment_id", "")).strip()
                ctext = str(c.get("text", "")).strip()
                if not cid or not ctext:
                    continue

                c_clean = preprocessor.clean_text_for_sentiment(ctext)
                c_nlp = preprocessor.clean_text_for_nlp(ctext)
                comment_entry = {
                    "comment_id": cid,
                    "parent_post_id": vid,
                    "source_platform": "youtube",
                    "author": str(c.get("author", "Anonymous")),
                    "comment_text": ctext,
                    "cleaned_text": c_clean,
                    "nlp_tokens": c_nlp,
                    "likes": max(0, int(c.get("likes", 0))),
                    "created_timestamp": created_iso,
                    "created_date": created_date,
                }
                comments.append(comment_entry)

        return posts, comments

    def clean_bluesky_records(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Transforms raw Bluesky records into standardized posts."""
        posts: List[Dict[str, Any]] = []
        comments: List[Dict[str, Any]] = []

        for r in raw_records:
            pid = str(r.get("post_id", "")).strip()
            if not pid:
                continue

            text = str(r.get("post_text", "")).strip()
            clean_text = preprocessor.clean_text_for_sentiment(text)
            nlp_text = preprocessor.clean_text_for_nlp(text)

            # Combine tags from facets and regex
            explicit_tags = [t.lower().replace("#", "") for t in r.get("hashtags", []) if t]
            regex_tags = preprocessor.extract_hashtags(text)
            combined_tags = list(dict.fromkeys(explicit_tags + regex_tags))

            created_iso, created_date = self.parse_timestamp(r.get("created_at"))
            likes = max(0, int(r.get("like_count", 0)))
            reposts = max(0, int(r.get("repost_count", 0)))
            replies = max(0, int(r.get("reply_count", 0)))

            post_entry = {
                "post_id": pid,
                "source_platform": "bluesky",
                "author": str(r.get("author_handle", "anonymous")),
                "content_title": text[:60] + ("..." if len(text) > 60 else ""),
                "content_body": text,
                "cleaned_text": clean_text,
                "nlp_tokens": nlp_text,
                "hashtags": combined_tags,
                "created_timestamp": created_iso,
                "created_date": created_date,
                "likes": likes,
                "comments": replies,
                "shares_reposts": reposts,
                "views": 0,
                "post_url": f"https://bsky.app/profile/{r.get('author_handle', 'bsky.social')}/post/{pid.split('/')[-1]}",
                "ingestion_timestamp": str(r.get("ingestion_timestamp", datetime.now(timezone.utc).isoformat())),
            }
            posts.append(post_entry)

        return posts, comments

    def deduplicate(self, records: List[Dict[str, Any]], id_key: str = "post_id") -> List[Dict[str, Any]]:
        """Deduplicates records based on (source_platform, primary_id)."""
        seen = set()
        deduped = []
        for r in records:
            key = (r.get("source_platform", ""), r.get(id_key, ""))
            if key not in seen and r.get(id_key):
                seen.add(key)
                deduped.append(r)
        return deduped


cleaner = DataCleaner()
