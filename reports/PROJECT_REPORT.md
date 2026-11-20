# SocialPulse: Social Media Analytics & Intelligence Platform
## Comprehensive Data Engineering Project Report

---

### Executive Summary

In today's hyper-connected digital landscape, enterprises face an overwhelming volume of unstructured public opinions across heterogeneous social media channels. Traditional sentiment analysis scripts often operate in silos, lack schema governance, fail to scale, and rarely implement resilient data quality assertions.

**SocialPulse** is an enterprise-grade Big Data Engineering platform designed to ingest, process, clean, analyze, and visualize public sentiment, audience engagement, trending topics, and customer concerns across three major independent social media channels: **Reddit**, **YouTube**, and **Bluesky**.

Built upon the industry-standard **Medallion Architecture** using **Databricks**, **Apache Spark (PySpark)**, **Delta Lake**, **Python**, and **Power BI**, SocialPulse transforms volatile, unstructured social feeds into governed analytical datasets powering an executive 6-page interactive dashboard.

---

## 1. Project Objectives & Business Justification

SocialPulse resolves core commercial and marketing intelligence requirements:
1. **Public Sentiment Monitoring:** Quantifying positive, negative, and neutral sentiment ratios across brands and products.
2. **Multi-Source Ingestion Governance:** Unifying divergent data APIs with rate-limit resiliency, OAuth2 security, and sample fallback mechanisms.
3. **Engagement Intelligence:** Normalizing heterogeneous interaction signals (likes, upvotes, comments, shares, views) into standardized performance scores.
4. **Hashtag Discovery & Trend Detection:** Tracking hashtag velocity, frequency, and sentiment affinity over time.
5. **Customer Friction & Complaint Triage:** Mining recurring complaint keywords, categorizing issues (Performance, Reliability, Billing, Support, Usability), and alerting operational stakeholders.
6. **Data Quality Assurance:** Enforcing automated validation rules preventing corrupted records, duplicate entries, and null identifiers from polluting analytical layers.

---

## 2. Technology Stack & Architectural Decision Rationale

| Layer / Component | Technology Selected | Engineering Justification |
| :--- | :--- | :--- |
| **Extraction & Scripting** | Python 3.10+, Requests, python-dotenv | Flexible HTTP requests, robust rate-limiting, secure credential encapsulation. |
| **Distributed ETL** | Apache Spark (PySpark 3.5 / 4.x) | Fault-tolerant distributed data processing, memory caching, and parallelized transformations. |
| **Storage & Governance** | Delta Lake (ACID, Time-Travel) | Eliminates pipeline corruption with ACID transactions, schema enforcement, and partitioning. |
| **Cloud Analytics Platform** | Databricks (Free / Community Edition) | Unified analytics engine supporting notebooks, job orchestration, and Delta tables. |
| **NLP & Sentiment** | VADER Sentiment (Rule/Lexicon based) | Specifically tuned for social media text, handling capitalization, punctuation, and idioms with low computational overhead. |
| **Business Intelligence** | Microsoft Power BI Desktop & Service | Interactive multi-page visual reporting, Star Schema data modeling, DAX time-intelligence. |
| **Testing & Quality Control** | Pytest, Custom DQ Framework | Automated regression testing, non-negative assertions, duplicate verification. |

---

## 3. Data Source Specifications & Schema Mapping

The platform ingests from three distinct social media architectures:

### 3.1 Reddit
- **API Protocol:** Reddit OAuth2 (`/r/{subreddit}/hot.json`).
- **Primary Metrics:** Score (Upvotes minus Downvotes), Comment Count, Created UTC.
- **Key Entity:** Submissions / Threads (`post_id`, `title`, `selftext`, `author`, `permalink`).
- **Normalization Strategy:** Upvote score mapped to `likes`, `shares` initialized to 0.

### 3.2 YouTube
- **API Protocol:** Google YouTube Data API v3 (`/search`, `/videos`, `/commentThreads`).
- **Primary Metrics:** View Count, Like Count, Comment Count, Comment Likes.
- **Key Entity:** Videos (`video_id`, `title`, `description`, `channelTitle`) and Public Comments (`comment_id`, `textDisplay`, `author`).
- **Normalization Strategy:** Videos ingested as primary posts, top comments parsed into relational comment records.

### 3.3 Bluesky (AT Protocol)
- **API Protocol:** AT Protocol XRPC (`app.bsky.feed.searchPosts`).
- **Primary Metrics:** Like Count, Repost Count, Reply Count.
- **Key Entity:** Posts (`cid`, `text`, `author.handle`, `facets`, `createdAt`).
- **Normalization Strategy:** Reposts mapped to `shares_reposts`, replies mapped to `comments`.

---

## 4. Medallion Architecture Implementation

```
[Raw Social APIs] ──► [BRONZE LAYER] ──► [SILVER LAYER] ──► [GOLD LAYER] ──► [POWER BI]
                       (Raw Immutability) (Cleaned/NLP)    (Star Schema)     (6-Page Dashboard)
```

### 4.1 Bronze Layer (Raw Storage)
- **Objective:** Immutable, append-only staging of raw payloads.
- **Audit Columns Appended:** `source_platform`, `ingestion_timestamp`, `data_mode`.
- **Tables Created:**
  - `bronze_reddit_posts`
  - `bronze_youtube_videos`
  - `bronze_youtube_comments`
  - `bronze_bluesky_posts`

### 4.2 Silver Layer (Cleaned, Standardized & Enriched)
- **Objective:** Schema harmonization, deduplication, NLP sentiment scoring, and data quality verification.
- **Transformations Applied:**
  1. Primary ID deduplication on `(source_platform, post_id)`.
  2. Timestamp normalization to UTC ISO8601 (`YYYY-MM-DDTHH:MM:SSZ`) and date dimension (`YYYY-MM-DD`).
  3. Text sanitization: URL regex stripping, HTML entity decoding, special character handling.
  4. Feature extraction: Regex hashtag parsing and tokenization.
  5. VADER Sentiment Scoring (`sentiment_compound`, `sentiment_label`).
  6. Engagement scoring calculation.
- **Tables Created:**
  - `silver_social_posts`
  - `silver_social_comments`
  - `silver_social_engagement`
  - `silver_social_hashtags`

### 4.3 Gold Layer (Business Analytics Aggregations)
- **Objective:** Star Schema analytical tables pre-aggregated for high-speed Power BI consumption.
- **Tables Created:**
  1. `gold_sentiment_summary`: Platform-wise positive, negative, and neutral percentages and Net Sentiment Score (NSS).
  2. `gold_platform_performance`: Aggregate interactions, views, and average engagement per post.
  3. `gold_hashtag_trends`: Temporal hashtag usage frequency, platform spread, and engagement.
  4. `gold_daily_trends`: Time-series post volumes and daily engagement.
  5. `gold_top_posts`: Top 15 high-impact posts with direct URL hyperlinks.
  6. `gold_topic_analysis`: Keyword frequency and sentiment cross-tabulation.
  7. `gold_customer_concerns`: Customer friction points categorized by problem taxonomy.

---

## 5. NLP & Sentiment Analysis Methodology

### 5.1 The VADER Engine
VADER (Valence Aware Dictionary and sEntiment Reasoner) was selected for its exceptional sensitivity to social media idioms, emoticons, capitalized shouting, and contrastive conjunctions (e.g., "The UI is clean *but* the app crashes").

### 5.2 Compound Score Classification Rules
The normalized compound polarity score ($C \in [-1.0, 1.0]$) determines the sentiment label:
$$\text{Sentiment Label} = \begin{cases} 
\text{Positive}, & C \ge 0.05 \\ 
\text{Negative}, & C \le -0.05 \\ 
\text{Neutral}, & -0.05 < C < 0.05 
\end{cases}$$

### 5.3 Documented NLP Limitations
1. **Sarcasm and Irony:** Statements such as "Oh brilliant, another price hike!" may be scored positive due to the word "brilliant" without deep semantic context.
2. **Context Shift:** Domain-specific technical terms (e.g., "overclocking", "throttling") require specialized domain tuning.
3. **Multi-lingual Feeds:** VADER primary dictionaries are optimized for English; non-English posts require pre-translation.

---

## 6. Engagement & Metric Formulations

Because interaction types vary across platforms, SocialPulse implements both a baseline engagement sum and a weighted interaction score:

### 6.1 Standard Engagement Formula
$$\text{Raw Engagement} = \text{Likes} + \text{Comments} + \text{Shares}$$

### 6.2 Weighted Engagement Formula
$$\text{Weighted Engagement} = \text{Likes} + 1.5 \times \text{Comments} + 2.0 \times \text{Shares}$$
*Rationale:* Comments and reposts require higher user investment than passive likes.

### 6.3 Net Sentiment Score (NSS)
$$\text{NSS} = \text{Positive } \% - \text{Negative } \%$$
Net Sentiment ranges from $-100$ (completely negative) to $+100$ (completely positive), serving as the primary executive brand perception metric.

---

## 7. Data Quality (DQ) Assertion Framework

Six automated quality gate rules are enforced during the Bronze-to-Silver transition:
1. **Rule 1 (Uniqueness):** Rejection of duplicate `(source_platform, post_id)` pairs.
2. **Rule 2 (Primary Key Completeness):** Zero null or empty post IDs allowed.
3. **Rule 3 (Temporal Validity):** Parseable ISO8601 timestamps.
4. **Rule 4 (Metric Sanity):** Non-negative engagement counts (`likes >= 0`, `comments >= 0`, `shares >= 0`).
5. **Rule 5 (Content Integrity):** Post text non-empty after URL and whitespace cleaning.
6. **Rule 6 (Platform Referential Integrity):** Platform must be within `{'reddit', 'youtube', 'bluesky'}`.

**Pipeline Test Execution Audit Result:**
- Total records evaluated: 20
- Passed: 20 (100.0% Pass Rate)
- Failed: 0
- Status: HEALTHY

---

## 8. Power BI Interactive Executive Dashboard Design

The dashboard delivers an intuitive user experience organized across six dedicated pages:
- **Page 1: Executive Overview:** High-level KPIs, overall sentiment donut, daily post volume trend, platform interaction comparison.
- **Page 2: Sentiment Intelligence:** Platform sentiment distribution stacked bars, Net Sentiment Score daily trajectory, topic sentiment scatter plot.
- **Page 3: Engagement Analytics:** Likes, comments, and shares breakdown, top 10 engaging posts table with direct hyperlinks.
- **Page 4: Hashtag & Trend Intelligence:** Top 10 hashtag horizontal ranking, engagement treemap, hashtag sentiment breakdown.
- **Page 5: Platform Comparison:** Reddit vs. YouTube vs. Bluesky multi-metric benchmarks and interaction velocity.
- **Page 6: Customer Concerns & Topic Analysis:** Complaint frequency by taxonomy (Reliability, Performance, Billing, Support, Usability) and incident context matrix.

---

## 9. Team Member Responsibilities & Division of Labor

The project was executed by a collaborative 4-person engineering team:

| Member & Role | Key Responsibilities | Modules Authored |
| :--- | :--- | :--- |
| **Member 1 – Team Lead & Big Data Architect** | System architecture, Databricks environment setup, Bronze layer ingestion, pipeline orchestration, and end-to-end integration. | `01_bronze_layer.py`, `run_pipeline.py`, `config/` |
| **Member 2 – Data Extraction Specialist** | API connectors, OAuth2 authentication, rate limit resilience, and benchmark sample generators for Reddit, YouTube, and Bluesky. | `reddit_extractor.py`, `youtube_extractor.py`, `bluesky_extractor.py`, `sample_data_generator.py` |
| **Member 3 – Processing & NLP Engineer** | Silver layer transformations, regex text cleaning, VADER sentiment analyzer, hashtag mining, and Data Quality assertions. | `02_silver_layer.py`, `text_preprocessing.py`, `data_cleaning.py`, `sentiment_analysis.py`, `data_validation.py` |
| **Member 4 – BI & Analytics Engineer** | Gold layer business aggregations, Power BI dimensional model (Star Schema), DAX measure library, theme configuration, and reports. | `03_gold_layer.py`, `04_pipeline_validation.py`, `powerbi_documentation.md`, `socialpulse_theme.json`, report documentation |

---

## 10. Conclusion & Future Enhancements

SocialPulse demonstrates a production-grade, reproducible data engineering workflow from volatile raw social feeds to governed executive business intelligence. 

**Future Roadmap:**
1. Integration of fine-tuned transformer models (e.g., RoBERTa / DeBERTa) for contextual sarcasm detection.
2. Real-time streaming ingestion via Apache Spark Structured Streaming and Kafka / Azure Event Hubs.
3. Automated Slack / Microsoft Teams alerting when negative sentiment spikes beyond standard control thresholds.
