# SocialPulse: Social Media Analytics & Intelligence Platform
## University Capstone & Industry Project Defense Presentation

**Slide Deck Structure (12 Slides)**

---

### Slide 1: Title & Project Overview
- **Header:** SocialPulse – Social Media Analytics & Intelligence Platform
- **Subtitle:** An End-to-End Big Data Engineering Solution for Sentiment, Trends & Public Opinion Monitoring
- **Team Members:**
  - Member 1: Team Lead & Data Engineer
  - Member 2: Data Extraction Specialist
  - Member 3: Data Processing & NLP Engineer
  - Member 4: Business Intelligence & Analytics Developer
- **Tech Stack:** Python • Apache Spark • Databricks • Delta Lake • Power BI

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Millions of unstructured opinions are published daily across fragmented social platforms (Reddit, YouTube, Bluesky).
- **The Gap:** Organizations struggle with slow, brittle scripts, lack of data quality validation, and disjointed reporting.
- **The Solution:** A unified, automated Lakehouse ETL pipeline delivering clean, validated, and aggregated intelligence to decision-makers.

---

### Slide 3: System Architecture & Data Flow
- **Visual:** Ingestion Sources ➔ Bronze Delta ➔ Silver Delta (NLP/Cleaning) ➔ Gold Delta (Analytics) ➔ Power BI Dashboard
- **Key Architecture Decisions:**
  - Multi-source ingestion with rate-limiting and benchmark sample fallbacks.
  - ACID transactions and time-travel debugging via Delta Lake.
  - Separation of raw storage, cleansed entities, and business-ready analytical datasets.

---

### Slide 4: Data Extraction Engine
- **Reddit API:** OAuth2 authentication, fetching hot posts, upvotes, comments across targeted subreddits.
- **YouTube Data API v3:** Ingesting video metadata, view counts, and public comment threads with quota tracking.
- **Bluesky AT Protocol:** Public XRPC search endpoints and authenticated feeds capturing posts, reposts, and tags.
- **Resilience:** Automatic fallback to benchmark datasets during API rate limits or offline grading.

---

### Slide 5: Medallion Architecture Deep Dive
- **Bronze Layer (Raw):**
  - Immutable raw payload preservation.
  - Added metadata: `source_platform`, `ingestion_timestamp`.
- **Silver Layer (Standardized & Enriched):**
  - Deduplication on `(platform, post_id)`.
  - UTC ISO8601 normalization.
  - Cleaned text, regex hashtag extraction.
  - VADER sentiment compound scoring and classification.
- **Gold Layer (Aggregated Analytics):**
  - 7 Star Schema analytical tables ready for Power BI.

---

### Slide 6: NLP & Sentiment Analysis Engine
- **Algorithm:** VADER (Valence Aware Dictionary and sEntiment Reasoner).
- **Why VADER:** Optimized for social media shorthand, emojis, capitalization shouting, and contrastive words ("but", "however").
- **Scoring Thresholds:**
  - Compound $\ge 0.05 \implies$ **Positive**
  - Compound $\le -0.05 \implies$ **Negative**
  - Else $\implies$ **Neutral**
- **Documented Limitations:** Sarcasm, domain idioms, context shift.

---

### Slide 7: Engagement & Trend Analytics
- **Cross-Platform Normalization:**
  - Standard Interaction: $\text{Likes} + \text{Comments} + \text{Shares}$
  - Weighted Score: $\text{Likes} + 1.5 \times \text{Comments} + 2.0 \times \text{Shares}$
- **Hashtag Intelligence:** Volume tracking, platform spread, sentiment affinity.
- **Net Sentiment Score (NSS):** $\text{Positive } \% - \text{Negative } \%$ brand perception index.

---

### Slide 8: Data Quality & Governance Framework
- **Automated Validation Rules:**
  1. Primary Key Uniqueness (Zero duplicates)
  2. Null & Empty ID rejection
  3. Valid timestamp parseability
  4. Non-negative engagement counts
  5. Content length and spam filtering
- **Execution Result:** 100% Pass Rate on benchmark validation run.

---

### Slide 9: Power BI Interactive Executive Dashboard
- **Page 1:** Executive Overview (High-level KPIs, sentiment donut, daily trends)
- **Page 2:** Sentiment Intelligence (Platform breakdown, NSS trajectory)
- **Page 3:** Engagement Analytics (Top 10 engaging posts portfolio, virality)
- **Page 4:** Hashtag & Trend Intelligence (Top 10 hashtag rankings, treemap)
- **Page 5:** Platform Comparison (Reddit vs. YouTube vs. Bluesky benchmarks)
- **Page 6:** Customer Concerns & Topic Analysis (Complaint taxonomy, incident matrix)

---

### Slide 10: Technical Demonstration & Verification
- **Execution Script:** Single-command run `python run_pipeline.py`.
- **Benchmark Run:** Ingested 20 multi-source posts, 10 comments, 58 hashtag instances.
- **Gold Tables:** 7 analytical datasets exported to Delta and CSV.
- **Test Suite:** 11 passing pytest unit and integration tests in 0.41 seconds.

---

### Slide 11: Team Division of Responsibilities
- **Member 1 (Team Lead & Big Data):** Architecture, Databricks config, Bronze layer, orchestration.
- **Member 2 (Data Extraction):** Reddit, YouTube, and Bluesky extractors, quota management.
- **Member 3 (Processing & NLP):** Silver layer, data cleaning, VADER scoring, quality assertions.
- **Member 4 (BI & Analytics):** Gold layer aggregations, Power BI Star Schema, DAX measures, project report.

---

### Slide 12: Summary, Future Work & Q&A
- **Summary:** Delivered a fully working, reproducible Big Data engineering solution adhering to university and industry standards.
- **Future Enhancements:** Streaming ingestion with Spark Structured Streaming, Kafka, transformer LLMs.
- **Open for Questions!**
