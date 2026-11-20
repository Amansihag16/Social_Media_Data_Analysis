"""
SocialPulse - Engagement Analytics Module
Calculates composite engagement metrics, platform-weighted interactions,
and multi-dimensional aggregations (temporal, platform-wise, and sentiment-based).
"""

from typing import Dict, List, Any
from collections import defaultdict


class EngagementAnalyzer:
    """Calculates standardized and weighted cross-platform engagement."""

    @staticmethod
    def calculate_post_engagement(post: Dict[str, Any], weight_comments: float = 1.5, weight_shares: float = 2.0) -> Dict[str, Any]:
        """
        Computes composite engagement score.
        Weighted formula: likes + (comments * 1.5) + (shares * 2.0)
        Standard formula: likes + comments + shares
        """
        likes = max(0, int(post.get("likes", 0)))
        comments = max(0, int(post.get("comments", 0)))
        shares = max(0, int(post.get("shares_reposts", 0)))

        raw_engagement = likes + comments + shares
        weighted_engagement = round(likes + (comments * weight_comments) + (shares * weight_shares), 2)

        return {
            "raw_engagement": raw_engagement,
            "weighted_engagement": weighted_engagement,
            "likes": likes,
            "comments": comments,
            "shares": shares,
        }

    def enrich_records(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enriches each post dictionary with engagement metrics."""
        for p in posts:
            metrics = self.calculate_post_engagement(p)
            p["raw_engagement"] = metrics["raw_engagement"]
            p["weighted_engagement"] = metrics["weighted_engagement"]
            # Primary engagement score for Gold reporting
            p["engagement_score"] = metrics["raw_engagement"]
        return posts

    def summarize_platform_performance(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregates engagement by platform."""
        groups = defaultdict(lambda: {
            "total_posts": 0,
            "total_likes": 0,
            "total_comments": 0,
            "total_shares": 0,
            "total_engagement": 0,
            "total_views": 0,
        })

        for p in posts:
            plat = p.get("source_platform", "unknown")
            g = groups[plat]
            g["total_posts"] += 1
            g["total_likes"] += int(p.get("likes", 0))
            g["total_comments"] += int(p.get("comments", 0))
            g["total_shares"] += int(p.get("shares_reposts", 0))
            g["total_engagement"] += int(p.get("engagement_score", 0))
            g["total_views"] += int(p.get("views", 0))

        summary = []
        for plat, stats in groups.items():
            avg_eng = round(stats["total_engagement"] / max(1, stats["total_posts"]), 2)
            summary.append({
                "platform": plat,
                "total_posts": stats["total_posts"],
                "total_likes": stats["total_likes"],
                "total_comments": stats["total_comments"],
                "total_shares": stats["total_shares"],
                "total_views": stats["total_views"],
                "total_engagement": stats["total_engagement"],
                "avg_engagement_per_post": avg_eng,
            })

        return sorted(summary, key=lambda x: x["total_engagement"], reverse=True)


engagement_engine = EngagementAnalyzer()
