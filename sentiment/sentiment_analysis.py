"""
SocialPulse - NLP & Sentiment Analysis Module
Performs lexicon and rule-based sentiment classification using VADER (Valence Aware Dictionary
and sEntiment Reasoner) with compound threshold mapping.
Includes documentation of NLP limitations (sarcasm, slang, context-dependence).
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SentimentAnalysis")

# Attempt to import SentimentIntensityAnalyzer from vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment package not installed; falling back to lightweight lexicon analyzer.")


class RuleBasedFallbackAnalyzer:
    """Lightweight rule-based sentiment fallback when VADER library is absent."""

    POSITIVE_WORDS = {
        "great", "excellent", "best", "brilliant", "delighted", "love", "amazing",
        "good", "fast", "improved", "breakthrough", "exceptional", "happy", "recommend",
        "nice", "awesome", "loving", "clean", "outstanding", "super"
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "worst", "crash", "crashes", "crashing", "freeze", "freezes",
        "broken", "error", "errors", "timeout", "unresponsive", "refused", "complaint",
        "expensive", "waste", "horrible", "outage", "hate", "slow", "fail", "failed"
    }

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = [w.lower().strip(".,!?:;\"'()[]{}") for w in text.split()]
        pos_count = sum(1 for w in tokens if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in tokens if w in self.NEGATIVE_WORDS)
        total = max(1, len(tokens))

        pos_score = round(pos_count / total, 3)
        neg_score = round(neg_count / total, 3)
        neu_score = round(max(0.0, 1.0 - (pos_score + neg_score)), 3)

        compound_raw = (pos_count - neg_count) / max(1, (pos_count + neg_count + 1))
        compound = round(max(-1.0, min(1.0, compound_raw)), 4)

        return {"neg": neg_score, "neu": neu_score, "pos": pos_score, "compound": compound}


class SentimentAnalyzer:
    """
    Industry-grade Sentiment Analysis engine using VADER.
    Maps compound sentiment scores to Positive, Negative, or Neutral categories.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = RuleBasedFallbackAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single text string and returns scores and category.
        Rules:
        - compound >= 0.05  -> Positive
        - compound <= -0.05 -> Negative
        - else              -> Neutral
        """
        if not text or not isinstance(text, str):
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_label": "Neutral"
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)

        if compound >= self.pos_threshold:
            label = "Positive"
        elif compound <= self.neg_threshold:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": compound,
            "positive": scores.get("pos", 0.0),
            "negative": scores.get("neg", 0.0),
            "neutral": scores.get("neu", 0.0),
            "sentiment_label": label
        }

    def enrich_records(self, records: List[Dict[str, Any]], text_field: str = "cleaned_text") -> List[Dict[str, Any]]:
        """Enriches each record with sentiment polarity scores and label."""
        for r in records:
            text = r.get(text_field) or r.get("post_title") or r.get("post_text") or ""
            res = self.analyze_text(text)
            r["sentiment_compound"] = res["compound"]
            r["sentiment_positive"] = res["positive"]
            r["sentiment_negative"] = res["negative"]
            r["sentiment_neutral"] = res["neutral"]
            r["sentiment_label"] = res["sentiment_label"]
        return records


# Global sentiment analyzer instance
sentiment_engine = SentimentAnalyzer()
