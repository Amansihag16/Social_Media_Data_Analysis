"""
SocialPulse - Topic & Customer Concerns Analysis Module
Mines key topics from cleaned tokens and identifies recurring customer concerns,
complaints, and product issues across social media platforms.
"""

from typing import Dict, List, Any
from collections import defaultdict, Counter

# Categorized complaint & concern taxonomy
COMPLAINT_TAXONOMY = {
    "Reliability": ["crash", "crashes", "crashing", "freeze", "freezes", "broken", "outage", "down", "fail", "failed"],
    "Performance": ["slow", "latency", "timeout", "throttling", "overheating", "hot", "lag", "battery"],
    "Billing & Pricing": ["price", "pricing", "hike", "expensive", "subscription", "refund", "billing", "contract", "charge"],
    "Customer Support": ["support", "ticket", "unhelpful", "refused", "service", "dispute", "ignored"],
    "Usability & Bugs": ["bug", "bugs", "glitch", "glitches", "error", "errors", "500", "unresponsive", "patch"]
}

# Reverse lookup dictionary: keyword -> category
KEYWORD_TO_CATEGORY = {}
for category, keywords in COMPLAINT_TAXONOMY.items():
    for kw in keywords:
        KEYWORD_TO_CATEGORY[kw] = category


class TopicAnalyzer:
    """Mines discussion topics and isolates customer dissatisfaction vectors."""

    def extract_topics(self, posts: List[Dict[str, Any]], top_n_per_platform: int = 15) -> List[Dict[str, Any]]:
        """
        Extracts top topics/keywords per platform with associated sentiment distribution:
        - Topic / keyword
        - Platform
        - Frequency
        - Positive count, Negative count, Neutral count
        """
        grouped = defaultdict(lambda: Counter())
        sentiment_tracker = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})

        for p in posts:
            plat = p.get("source_platform", "unknown")
            tokens = p.get("nlp_tokens", "").split()
            sentiment = p.get("sentiment_label", "Neutral")

            for token in tokens:
                if len(token) > 2:
                    grouped[plat][token] += 1
                    sentiment_tracker[(plat, token)][sentiment] += 1

        topic_rows = []
        for plat, counter in grouped.items():
            for kw, freq in counter.most_common(top_n_per_platform):
                sent_counts = sentiment_tracker[(plat, kw)]
                topic_rows.append({
                    "topic_keyword": kw,
                    "platform": plat,
                    "frequency": freq,
                    "positive_count": sent_counts["Positive"],
                    "negative_count": sent_counts["Negative"],
                    "neutral_count": sent_counts["Neutral"],
                })

        return sorted(topic_rows, key=lambda x: (x["frequency"]), reverse=True)

    def extract_customer_concerns(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifies negative complaints and concerns matching issue taxonomy:
        - Common complaint keyword
        - Issue category (Performance, Reliability, Billing, Support, Usability)
        - Frequency
        - Related platform
        - Primary sentiment category
        """
        # Group key: (keyword, platform)
        concern_counts = defaultdict(lambda: {
            "frequency": 0,
            "category": "General Issue",
            "negative_count": 0,
            "neutral_count": 0,
            "positive_count": 0,
            "sample_contexts": [],
        })

        for p in posts:
            plat = p.get("source_platform", "unknown")
            tokens = set(p.get("nlp_tokens", "").split())
            sentiment = p.get("sentiment_label", "Neutral")
            clean_text = p.get("cleaned_text", "")

            for kw, category in KEYWORD_TO_CATEGORY.items():
                if kw in tokens:
                    key = (kw, plat)
                    data = concern_counts[key]
                    data["frequency"] += 1
                    data["category"] = category
                    if sentiment == "Negative":
                        data["negative_count"] += 1
                    elif sentiment == "Positive":
                        data["positive_count"] += 1
                    else:
                        data["neutral_count"] += 1

                    if len(data["sample_contexts"]) < 2:
                        data["sample_contexts"].append(clean_text[:120])

        concerns = []
        for (kw, plat), data in concern_counts.items():
            # Determine dominant sentiment category for this complaint keyword
            if data["negative_count"] >= data["positive_count"] and data["negative_count"] > 0:
                dom_sentiment = "Negative"
            elif data["positive_count"] > data["negative_count"]:
                dom_sentiment = "Positive"
            else:
                dom_sentiment = "Neutral"

            concerns.append({
                "complaint_keyword": kw,
                "issue_category": data["category"],
                "frequency": data["frequency"],
                "platform": plat,
                "sentiment_category": dom_sentiment,
                "negative_count": data["negative_count"],
                "sample_context": data["sample_contexts"][0] if data["sample_contexts"] else "",
            })

        return sorted(concerns, key=lambda x: x["frequency"], reverse=True)


topic_engine = TopicAnalyzer()
