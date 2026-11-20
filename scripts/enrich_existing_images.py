"""
Enrich Existing Articles with Category & Editorial Visuals
Ensures 100% of articles in SQLite database and Silver Parquet have valid image URLs.
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "socialpulse_india" / "data" / "socialpulse_india.db"
PARQUET_PATH = BASE_DIR / "socialpulse_india" / "data" / "silver" / "articles_silver.parquet"

CATEGORY_IMAGE_MAP = {
    "Indian Politics": "https://images.unsplash.com/photo-1598091383021-15ddea10925d?auto=format&fit=crop&w=800&q=80",
    "Indian Economy": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=800&q=80",
    "Economy": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=800&q=80",
    "Business": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=800&q=80",
    "Technology": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80",
    "Artificial Intelligence": "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=800&q=80",
    "Sports": "https://images.unsplash.com/photo-1531415074868-036b107e775a?auto=format&fit=crop&w=800&q=80",
    "Entertainment": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?auto=format&fit=crop&w=800&q=80",
    "International Relations": "https://images.unsplash.com/photo-1541872703-74c5e44368f9?auto=format&fit=crop&w=800&q=80",
    "Health": "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=800&q=80",
    "Education": "https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=800&q=80",
    "Top Stories": "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?auto=format&fit=crop&w=800&q=80",
    "General": "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?auto=format&fit=crop&w=800&q=80"
}

def enrich():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT article_id, category, image_url FROM articles")
    rows = cursor.fetchall()
    updated = 0
    for art_id, cat, img in rows:
        if not img or len(str(img).strip()) < 5:
            fallback = CATEGORY_IMAGE_MAP.get(cat, CATEGORY_IMAGE_MAP["General"])
            cursor.execute("UPDATE articles SET image_url = ? WHERE article_id = ?", (fallback, art_id))
            updated += 1
    conn.commit()
    print(f"Updated {updated} database records with editorial visual fallbacks.")
    
    # Check total
    cursor.execute("SELECT count(*), count(image_url) FROM articles")
    total, with_img = cursor.fetchone()
    print(f"Total articles: {total}, With image_url: {with_img}")
    conn.close()

    # Update Silver Parquet if present
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        if "image_url" not in df.columns:
            df["image_url"] = None
        df["image_url"] = df.apply(
            lambda r: r["image_url"] if (pd.notnull(r["image_url"]) and str(r["image_url"]).strip() != "")
            else CATEGORY_IMAGE_MAP.get(r.get("category"), CATEGORY_IMAGE_MAP["General"]),
            axis=1
        )
        df.to_parquet(PARQUET_PATH, index=False)
        print("Updated Silver parquet with image URLs.")

if __name__ == "__main__":
    enrich()
