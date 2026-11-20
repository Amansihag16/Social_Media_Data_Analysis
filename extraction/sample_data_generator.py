"""
SocialPulse Fresh & High-Fidelity Multi-Source Dataset Generator
===============================================================
Generates 650+ ultra-realistic, authentic, and fresh social media records
across Reddit, YouTube, and Bluesky with timestamps leading up to today (November 2026).
Features authentic tech discussions, real-world outages, enterprise benchmarks,
viral threads, and customer friction patterns.
"""

import json
import sys
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config

random.seed(2026)

# -----------------------------------------------------------------------------
# ULTRA-AUTHENTIC 2026 TECH SCENARIOS & POST TEMPLATES
# -----------------------------------------------------------------------------
AUTHENTIC_POSTS = [
    # --- POSITIVE / VIRAL TECH WINS ---
    {
        "category": "ai",
        "sentiment": "positive",
        "title": "Migrated our ETL pipeline from Pandas to Polars + DuckDB: 18x speedup on single M4 Max node",
        "body": "We previously ran a 4-hour batch job on a 16-node Spark cluster for daily aggregations. Rewrote the core transformations using Polars lazy evaluation and DuckDB parquet streaming. The entire pipeline now finishes in 13 minutes on a single local workstation. Unbelievable performance and memory efficiency! #dataengineering #polars #duckdb #python",
        "tags": ["dataengineering", "duckdb", "polars"]
    },
    {
        "category": "bigdata",
        "sentiment": "positive",
        "title": "Databricks Liquid Clustering on Delta Lake 3.2 slashed our AWS S3 storage and query costs by 62%",
        "body": "Liquid clustering completely eliminated our small file problem and partition skew on our 80TB customer telemetry lakehouse. Queries that used to scan millions of files now execute via instant data skipping in under 800ms. Massive cost reduction for our team. #databricks #deltalake #spark #bigdata",
        "tags": ["databricks", "deltalake", "bigdata"]
    },
    {
        "category": "ai",
        "sentiment": "positive",
        "title": "Open source local LLM inference with Ollama + vLLM is finally faster than proprietary cloud APIs",
        "body": "Benchmarked Llama 3.3 70B quantized with speculative decoding on dual RTX 4090s. Achieving 110 tokens/second with sub-40ms time-to-first-token. Complete data privacy and zero API rate limits for our internal enterprise search. #ai #opensource #llm #machinelearning",
        "tags": ["ai", "llm", "opensource"]
    },
    {
        "category": "devops",
        "sentiment": "positive",
        "title": "Seamless PySpark 3.5 migration with Adaptive Query Execution (AQE) enabled",
        "body": "AQE dynamic partition coalescing solved our straggler tasks overnight. We had 200+ micro-batch streaming jobs suffering from join skews, and AQE re-balanced the shuffle partitions automatically. Outstanding work by the Apache Spark contributors! #apache_spark #pyspark #dataengineering",
        "tags": ["apache_spark", "pyspark", "dataengineering"]
    },
    {
        "category": "cloud",
        "sentiment": "positive",
        "title": "Kudos to the Snowflake engineering team for zero-downtime cross-cloud failover",
        "body": "When AWS us-east-1 had network degradation this morning, our replicated Snowflake databases in GCP failed over in under 3 minutes with zero transactional data loss. Our mission-critical trading pipeline stayed 100% green. Exceptional engineering! #cloud #devops #database",
        "tags": ["cloud", "devops", "tech"]
    },
    {
        "category": "python",
        "sentiment": "positive",
        "title": "Python 3.12 GIL improvements and JIT compiler benchmarks for real-time analytics",
        "body": "Tested the free-threaded Python build (PEP 703) on our multi-threaded telemetry collectors. Near-linear scaling across 32 CPU cores without multiprocessing serialization overhead. Python is entering a golden era for data engineering! #python #datascience #coding",
        "tags": ["python", "datascience", "coding"]
    },

    # --- NEUTRAL / BENCHMARKS & ARCHITECTURE GUIDES ---
    {
        "category": "bigdata",
        "sentiment": "neutral",
        "title": "Architectural breakdown: Apache Iceberg vs Delta Lake vs Apache Hudi in 2026",
        "body": "A deep dive into open table formats: comparing metadata scalability, Puffin stats, partition evolution, schema enforcement, and multi-engine read support across Trino, Athena, Spark, and Databricks. Full benchmarking whitepaper and reproducible scripts included. #dataengineering #bigdata #cloud",
        "tags": ["dataengineering", "bigdata", "cloud"]
    },
    {
        "category": "ai",
        "sentiment": "neutral",
        "title": "Evaluating RAG retrieval architectures: Dense Vector Search vs Hybrid BM25 + Cross-Encoder Reranking",
        "body": "Conducted recall and NDCG@10 evaluations on 200,000 technical PDF manuals. Hybrid search with Reciprocal Rank Fusion (RRF) achieved 94.2% accuracy compared to 81.5% for pure cosine dense retrieval. Here is our architectural diagram. #ai #nlp #datascience",
        "tags": ["ai", "nlp", "datascience"]
    },
    {
        "category": "bigdata",
        "sentiment": "neutral",
        "title": "Medallion Architecture design patterns for late-arriving CDC records in Silver Layer",
        "body": "How is your team handling out-of-order Change Data Capture events from Debezium/Kafka? Exploring watermarking windows, Delta table MERGE with condition checks, and SCD Type 2 dimension historization. Discussion thread below. #databricks #medallion #dataengineering",
        "tags": ["databricks", "medallion", "dataengineering"]
    },
    {
        "category": "devops",
        "sentiment": "neutral",
        "title": "Release Notes: Apache Kafka 3.8 Raft metadata quorum (KRaft) production readiness guide",
        "body": "Official summary of KRaft migration steps, deprecation of ZooKeeper dependencies, dynamic broker configuration updates, and multi-region partition rebalancing metrics. Link to upgrade checklist attached. #devops #kafka #cloud",
        "tags": ["devops", "cloud", "tech"]
    },
    {
        "category": "python",
        "sentiment": "neutral",
        "title": "Overview of PyArrow 16.0 memory management and zero-copy IPC sharing",
        "body": "Technical overview explaining how Arrow RecordBatches share off-heap memory between Python processes and C++ kernels without JSON serialization costs. Essential reading for data platform architects. #python #dataengineering #pyspark",
        "tags": ["python", "dataengineering", "pyspark"]
    },

    # --- NEGATIVE / OUTAGES, BUGS & BILLING ALERTS ---
    {
        "category": "cloud",
        "sentiment": "negative",
        "title": "Major AWS us-east-1 network latency spike impacting DynamoDB and API Gateways",
        "body": "Our production API endpoints are throwing 504 Gateway Timeouts and 500 Internal Server Errors since 09:30 UTC. Database queries that usually take 5ms are stalling at 8,000ms. Cloud status dashboard still deceptively shows 'All Systems Operational'. Frustrating outage! #cloud #outage #bug #latency",
        "tags": ["cloud", "outage", "bug"]
    },
    {
        "category": "billing",
        "sentiment": "negative",
        "title": "Surprise $9,400 cloud invoice due to serverless auto-scaling bug: Zero notification from vendor",
        "body": "A silent loop in an unhandled lambda exception caused thousands of zombie compute clusters to spin up overnight. The cloud vendor sent zero proactive billing threshold alerts and the support team refuses to refund the erroneous charges. Disgraceful customer experience! #billing #complaint #cloud #refund",
        "tags": ["billing", "complaint", "cloud"]
    },
    {
        "category": "bigdata",
        "sentiment": "negative",
        "title": "Severe memory leak in PySpark 3.5.1 streaming jobs on Kubernetes causing daily pod OOM crashes",
        "body": "Our Spark driver JVM runs out of heap memory every 8 to 12 hours under constant micro-batch throughput. Heap dump shows uncollected Py4J gateway references and metadata tombstone bloat. Pipelines are failing during critical client SLA windows. Critical bug! #apache_spark #pyspark #bug #crash",
        "tags": ["apache_spark", "pyspark", "bug"]
    },
    {
        "category": "support",
        "sentiment": "negative",
        "title": "Unacceptable support delay: Priority 1 production ticket ignored for 36 hours",
        "body": "Our enterprise contract guarantees a 15-minute response time for business-critical outages. Our cluster has been down since yesterday morning with corrupted transaction logs, and the vendor support portal has provided zero updates. Looking to migrate off this platform immediately. #complaint #support #latency",
        "tags": ["complaint", "support", "latency"]
    },
    {
        "category": "software",
        "sentiment": "negative",
        "title": "Disastrous breaking API change in SDK release v4.2 with zero deprecation warning",
        "body": "The latest minor version bump removed three fundamental client authentication methods without warning. CI/CD pipelines across all our staging environments broke simultaneously. Whoever approved this release needs to re-read semantic versioning guidelines. #bug #coding #tech",
        "tags": ["bug", "coding", "tech"]
    },
    {
        "category": "hardware",
        "sentiment": "negative",
        "title": "Server cluster thermal throttling and unexpected fan failure during heavy model training",
        "body": "Our dedicated GPU server rack suffered acute thermal throttling after firmware update. Clock speeds dropped by 65%, prolonging a 2-day distributed training run into a week. Hardware vendor RMA process is slow and bureaucratic. #hardware #crash #complaint",
        "tags": ["hardware", "crash", "complaint"]
    }
]

AUTHORS = [
    ("sarah_cloud_architect", "Sarah Chen", "Staff Data Architect @ FinTech"),
    ("alex_spark_lead", "Alex Rivera", "Principal Data Engineer"),
    ("marcus_devops_pro", "Marcus Vance", "SRE & Cloud Infrastructure Lead"),
    ("elena_ai_research", "Elena Rostova", "AI & NLP Systems Researcher"),
    ("kai_pythonista", "Kai Tanaka", "Senior Python & Distributed Systems Dev"),
    ("david_query_wizard", "David Miller", "Lakehouse Optimization Specialist"),
    ("priya_data_lead", "Priya Sharma", "VP of Big Data Analytics"),
    ("jordan_cloud_sre", "Jordan Taylor", "Cloud Reliability Engineer"),
    ("maya_vector_ml", "Maya Lin", "Vector DB & RAG Architect"),
    ("sam_database_ninja", "Sam O'Connor", "Database Performance Engineer"),
    ("tech_sentinel_watch", "Tech Sentinel", "Independent Enterprise Software Analyst"),
    ("infra_survivor_99", "DevOps Survivor", "Senior Cloud Engineer")
]

SUBREDDITS = ["dataengineering", "datascience", "apache_spark", "programming", "LocalLLaMA", "cloudcomputing", "aws"]
CHANNELS = ["Data Engineering Academy", "Spark & Cloud Systems", "PyData Global", "NextGen AI & Data", "Enterprise Tech Pulse"]


def generate_benchmark_datasets():
    """Generates 650+ high-fidelity, fresh social media records."""
    config.ensure_directories()
    sample_dir = config.DATA_SAMPLE_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Base time is right now (November 19, 2026)
    base_time = datetime.now(timezone.utc)
    print(f"Generating 650+ fresh records with real-world timestamps up to {base_time.strftime('%Y-%m-%d %H:%M:%S UTC')}...")

    # -------------------------------------------------------------
    # 1. REDDIT BENCHMARK DATASET (270 POSTS)
    # -------------------------------------------------------------
    reddit_records = []
    for i in range(1, 271):
        # Fresh distribution: 30% from last 48 hours, 40% from last 7 days, 30% older
        r_dist = random.random()
        if r_dist < 0.30:
            minutes_ago = random.randint(15, 2800) # within 48h
            created_dt = base_time - timedelta(minutes=minutes_ago)
        elif r_dist < 0.70:
            days_ago = random.randint(2, 7)
            created_dt = base_time - timedelta(days=days_ago, hours=random.randint(0, 23))
        else:
            days_ago = random.randint(8, 38)
            created_dt = base_time - timedelta(days=days_ago, hours=random.randint(0, 23))

        scenario = random.choice(AUTHENTIC_POSTS)
        author_tuple = random.choice(AUTHORS)

        # Realistic score following power law with a few viral posts
        if random.random() < 0.05:
            score = random.randint(2500, 14500) # Viral post
        elif random.random() < 0.25:
            score = random.randint(600, 2400)   # Popular post
        else:
            score = random.randint(25, 580)     # Normal thread

        comments = max(3, int(score * random.uniform(0.10, 0.35)))

        title_suffix = f" (Case Study #{i})" if i % 4 == 0 else ""
        reddit_records.append({
            "post_id": f"rd_{1000 + i}",
            "post_title": f"{scenario['title']}{title_suffix}",
            "post_description": scenario["body"],
            "subreddit": random.choice(SUBREDDITS),
            "author": author_tuple[0],
            "created_utc": created_dt.isoformat(),
            "score": score,
            "comment_count": comments,
            "post_url": f"https://reddit.com/r/{random.choice(SUBREDDITS)}/comments/rd_{1000 + i}",
            "source_platform": "reddit",
            "ingestion_timestamp": base_time.isoformat()
        })

    # -------------------------------------------------------------
    # 2. YOUTUBE BENCHMARK DATASET (180 VIDEOS + COMMENTS)
    # -------------------------------------------------------------
    youtube_records = []
    yt_comment_samples = [
        ("This is by far the cleanest explanation of Delta Lake transaction logs and liquid clustering on the web!", "positive"),
        ("Saved our team hundreds of hours during our cloud migration. Outstanding production-grade tutorial!", "positive"),
        ("Could you share the GitHub repository containing the PySpark and DuckDB benchmark scripts?", "neutral"),
        ("What AWS instance family did you choose for the memory-optimized driver nodes?", "neutral"),
        ("We tested this exact script in production and ran into an out of memory heap crash under 50GB load.", "negative"),
        ("The audio was distorted around the 12-minute mark and the config file in the description has 404 error.", "negative")
    ]

    for i in range(1, 181):
        r_dist = random.random()
        if r_dist < 0.35:
            minutes_ago = random.randint(30, 2800)
            created_dt = base_time - timedelta(minutes=minutes_ago)
        else:
            days_ago = random.randint(2, 40)
            created_dt = base_time - timedelta(days=days_ago, hours=random.randint(0, 23))

        scenario = random.choice(AUTHENTIC_POSTS)
        channel_name = random.choice(CHANNELS)

        if random.random() < 0.08:
            views = random.randint(45000, 320000) # Viral video
        elif random.random() < 0.30:
            views = random.randint(8000, 42000)
        else:
            views = random.randint(650, 7800)

        likes = max(15, int(views * random.uniform(0.04, 0.09)))
        comments_cnt = max(4, int(likes * random.uniform(0.08, 0.22)))

        comments_list = []
        for c_idx in range(random.randint(2, 4)):
            c_text, c_sent = random.choice(yt_comment_samples)
            comments_list.append({
                "comment_id": f"yt_c_{i}_{c_idx+1}",
                "text": c_text,
                "author": f"engineer_{random.randint(100, 999)}",
                "like_count": random.randint(2, 85),
                "published_at": (created_dt + timedelta(hours=random.randint(1, 36))).isoformat()
            })

        title = f"{scenario['title']} | Production Architecture & Code Demo"
        youtube_records.append({
            "video_id": f"yt_{2000 + i}",
            "video_title": title,
            "title": title,
            "description": f"{scenario['body']}\n\nTimestamps:\n0:00 Introduction\n2:30 Architecture Design\n8:15 Live Code Walkthrough\n15:40 Production Benchmarks",
            "channel_title": channel_name,
            "channel_id": f"UC_{channel_name.replace(' ', '')}",
            "published_at": created_dt.isoformat(),
            "view_count": views,
            "like_count": likes,
            "comment_count": comments_cnt,
            "comments": comments_list,
            "source_platform": "youtube",
            "ingestion_timestamp": base_time.isoformat()
        })

    # -------------------------------------------------------------
    # 3. BLUESKY BENCHMARK DATASET (200 MICRO-POSTS)
    # -------------------------------------------------------------
    bluesky_records = []
    for i in range(1, 201):
        r_dist = random.random()
        if r_dist < 0.40:
            minutes_ago = random.randint(5, 2800)
            created_dt = base_time - timedelta(minutes=minutes_ago)
        else:
            days_ago = random.randint(2, 35)
            created_dt = base_time - timedelta(days=days_ago, hours=random.randint(0, 23))

        scenario = random.choice(AUTHENTIC_POSTS)
        author_tuple = random.choice(AUTHORS)

        text = scenario["body"]
        if len(text) > 280:
            text = text[:272] + "..."

        if random.random() < 0.05:
            likes = random.randint(1200, 6800) # Viral post
        elif random.random() < 0.25:
            likes = random.randint(250, 1150)
        else:
            likes = random.randint(12, 240)

        reposts = max(2, int(likes * random.uniform(0.18, 0.42)))
        replies = max(1, int(likes * random.uniform(0.06, 0.20)))

        bluesky_records.append({
            "post_id": f"bsky_{3000 + i}",
            "post_text": text,
            "author_handle": f"{author_tuple[0]}.bsky.social",
            "created_at": created_dt.isoformat(),
            "like_count": likes,
            "repost_count": reposts,
            "reply_count": replies,
            "hashtags": scenario["tags"],
            "source_platform": "bluesky",
            "ingestion_timestamp": base_time.isoformat()
        })

    # Write JSON files
    with open(sample_dir / "reddit_sample.json", "w", encoding="utf-8") as f:
        json.dump(reddit_records, f, indent=2)

    with open(sample_dir / "youtube_sample.json", "w", encoding="utf-8") as f:
        json.dump(youtube_records, f, indent=2)

    with open(sample_dir / "bluesky_sample.json", "w", encoding="utf-8") as f:
        json.dump(bluesky_records, f, indent=2)

    total_records = len(reddit_records) + len(youtube_records) + len(bluesky_records)
    print(f"[SUCCESS] Generated {total_records} fresh records: Reddit ({len(reddit_records)}), YouTube ({len(youtube_records)}), Bluesky ({len(bluesky_records)})")
    return {
        "reddit_count": len(reddit_records),
        "youtube_count": len(youtube_records),
        "bluesky_count": len(bluesky_records),
        "total_records": total_records
    }


if __name__ == "__main__":
    generate_benchmark_datasets()
