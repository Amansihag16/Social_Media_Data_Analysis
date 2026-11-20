"""
SocialPulse Configuration & Settings Manager.
Handles environment variables, Databricks secrets, and storage paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "config" / ".env")


def get_secret_or_env(secret_key: str, default: str = "") -> str:
    """
    Attempts to retrieve a secret from Databricks dbutils first,
    falling back to os.environ.
    """
    try:
        # Check if running inside Databricks environment
        import IPython
        dbutils = IPython.get_ipython().user_ns.get("dbutils")
        if dbutils:
            return dbutils.secrets.get(scope="socialpulse_scope", key=secret_key)
    except Exception:
        pass
    return os.getenv(secret_key, default)


class Settings:
    PROJECT_ROOT = PROJECT_ROOT
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Storage Paths
    DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    DATA_SAMPLE_DIR = PROJECT_ROOT / os.getenv("DATA_SAMPLE_DIR", "data/sample")

    DELTA_BRONZE_DIR = PROJECT_ROOT / os.getenv("DELTA_BRONZE_DIR", "data/delta/bronze")
    DELTA_SILVER_DIR = PROJECT_ROOT / os.getenv("DELTA_SILVER_DIR", "data/delta/silver")
    DELTA_GOLD_DIR = PROJECT_ROOT / os.getenv("DELTA_GOLD_DIR", "data/delta/gold")

    # Reddit
    REDDIT_CLIENT_ID = get_secret_or_env("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = get_secret_or_env("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SocialPulseAnalytics/1.0.0")
    REDDIT_SUBREDDITS = [s.strip() for s in os.getenv("REDDIT_SUBREDDITS", "technology,datascience,gadgets").split(",")]
    REDDIT_POST_LIMIT = int(os.getenv("REDDIT_POST_LIMIT", "100"))

    # YouTube
    YOUTUBE_API_KEY = get_secret_or_env("YOUTUBE_API_KEY")
    YOUTUBE_SEARCH_QUERIES = [q.strip() for q in os.getenv("YOUTUBE_SEARCH_QUERIES", "AI technology,data engineering").split(",")]
    YOUTUBE_MAX_RESULTS = int(os.getenv("YOUTUBE_MAX_RESULTS", "50"))

    # Bluesky
    BLUESKY_HANDLE = get_secret_or_env("BLUESKY_HANDLE")
    BLUESKY_APP_PASSWORD = get_secret_or_env("BLUESKY_APP_PASSWORD")
    BLUESKY_SEARCH_QUERIES = [q.strip() for q in os.getenv("BLUESKY_SEARCH_QUERIES", "technology,data engineering").split(",")]
    BLUESKY_POST_LIMIT = int(os.getenv("BLUESKY_POST_LIMIT", "100"))

    @classmethod
    def ensure_directories(cls):
        """Creates required directories if they don't already exist."""
        for path in [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.DATA_SAMPLE_DIR,
            cls.DELTA_BRONZE_DIR,
            cls.DELTA_SILVER_DIR,
            cls.DELTA_GOLD_DIR,
            cls.PROJECT_ROOT / "reports" / "pipeline_screenshots",
        ]:
            path.mkdir(parents=True, exist_ok=True)


config = Settings()
config.ensure_directories()
