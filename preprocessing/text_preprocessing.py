"""
SocialPulse - Text Preprocessing Module
Performs natural language preprocessing on social media text:
URL removal, mention normalization, hashtag extraction, emoji handling, and stopword filtering.
"""

import re
import html
import unicodedata
from typing import List, Tuple, Set

# Comprehensive standard stop words list (independent of external NLTK downloads for reliability)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#(\w+)", re.UNICODE)
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s#]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


class TextPreprocessor:
    """Preprocesses social media text for NLP, sentiment, and topic analysis."""

    def __init__(self, stopwords: Set[str] = None):
        self.stopwords = stopwords or DEFAULT_STOPWORDS

    def clean_text_for_sentiment(self, text: str) -> str:
        """
        Cleans text while preserving emotional signals (punctuation & capitalization)
        critical for VADER sentiment scoring.
        """
        if not text or not isinstance(text, str):
            return ""

        # Decode HTML entities (e.g. &amp; -> &)
        text = html.unescape(text)

        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Remove URLs
        text = URL_PATTERN.sub("", text)

        # Remove redundant whitespace
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        return text

    def clean_text_for_nlp(self, text: str) -> str:
        """
        More aggressive cleaning for topic/keyword modeling:
        lowercased, punctuation removed, stopwords stripped.
        """
        cleaned = self.clean_text_for_sentiment(text).lower()

        # Remove mentions and hashtags symbols
        cleaned = MENTION_PATTERN.sub("", cleaned)
        cleaned = SPECIAL_CHAR_PATTERN.sub(" ", cleaned)

        # Tokenize and remove stopwords
        tokens = cleaned.split()
        filtered = [t for t in tokens if t not in self.stopwords and len(t) > 2 and not t.isdigit()]
        return " ".join(filtered)

    def extract_hashtags(self, text: str) -> List[str]:
        """Extracts all lowercase hashtags without the '#' symbol."""
        if not text:
            return []
        matches = HASHTAG_PATTERN.findall(text)
        # Normalize to lowercase and remove duplicates while preserving order
        seen = set()
        result = []
        for tag in matches:
            t = tag.lower().strip()
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def tokenize(self, text: str) -> List[str]:
        """Splits cleaned text into constituent word tokens."""
        return self.clean_text_for_nlp(text).split()

    def is_spam_or_empty(self, text: str, min_length: int = 5) -> bool:
        """Flags potential spam, promotional links without content, or empty text."""
        cleaned = self.clean_text_for_sentiment(text)
        if len(cleaned.strip()) < min_length:
            return True
        # Check if text is just repetitive characters (e.g., 'aaaaaa')
        if len(set(cleaned.replace(" ", ""))) <= 2 and len(cleaned) > 10:
            return True
        return False


# Singleton instance for general reuse
preprocessor = TextPreprocessor()
