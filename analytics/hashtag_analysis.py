"""
SocialPulse - Hashtag Analytics Module
Discovers trending hashtags, tracks volume over time,
measures engagement generated, and analyzes sentiment associations.
"""

from typing import Dict, List, Any
from collections import defaultdict


class HashtagAnalyzer:
    """Analyzes hashtag distribution, engagement impact, and sentiment affinity."""

    def analyze_hashtags(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregates hashtag metrics across posts:
        - Hashtag name
        - Platform
        - Usage frequency
        - Date
        - Total engagement generated
        - Sentiment distribution (positive, negative, neutral)
        """
        # Group key: (hashtag, platform, date)
        grouped = defaultdict(lambda: {
            "frequency": 0,
            "total_engagement": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        })

        for p in posts:
            tags = p.get("hashtags", [])
            plat = p.get("source_platform", "unknown")
            date_str = p.get("created_date", "")
            eng = int(p.get("engagement_score", 0))
            sentiment = p.get("sentiment_label", "Neutral")

            for tag in tags:
                tag_norm = tag.lower().strip()
                if not tag_norm:
                    continue

                key = (tag_norm, plat, date_str)
                grouped[key]["frequency"] += 1
                grouped[key]["total_engagement"] += eng
                if sentiment == "Positive":
                    grouped[key]["positive_count"] += 1
                elif sentiment == "Negative":
                    grouped[key]["negative_count"] += 1
                else:
                    grouped[key]["neutral_count"] += 1

        results = []
        for (tag, plat, date_str), data in grouped.items():
            results.append({
                "hashtag": f"#{tag}",
                "platform": plat,
                "date": date_str,
                "usage_frequency": data["frequency"],
                "total_engagement": data["total_engagement"],
                "positive_count": data["positive_count"],
                "negative_count": data["negative_count"],
                "neutral_count": data["neutral_count"],
            })

        # Sort by usage frequency and engagement descending
        return sorted(results, key=lambda x: (x["usage_frequency"], x["total_engagement"]), reverse=True)

    def get_top_n_hashtags(self, hashtag_summary: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top N hashtags aggregated across all platforms and dates."""
        agg = defaultdict(lambda: {"total_usage": 0, "total_engagement": 0, "pos": 0, "neg": 0, "neu": 0})
        for item in hashtag_summary:
            tag = item["hashtag"]
            agg[tag]["total_usage"] += item["usage_frequency"]
            agg[tag]["total_engagement"] += item["total_engagement"]
            agg[tag]["pos"] += item["positive_count"]
            agg[tag]["neg"] += item["negative_count"]
            agg[tag]["neu"] += item["neutral_count"]

        top = []
        for tag, data in agg.items():
            top.append({
                "hashtag": tag,
                "total_usage": data["total_usage"],
                "total_engagement": data["total_engagement"],
                "positive_count": data["pos"],
                "negative_count": data["neg"],
                "neutral_count": data["neu"],
            })

        return sorted(top, key=lambda x: x["total_usage"], reverse=True)[:n]


hashtag_engine = HashtagAnalyzer()
