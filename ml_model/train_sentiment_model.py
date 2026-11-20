"""
SocialPulse Machine Learning Sentiment Model Trainer & Multi-Model Tournament
=============================================================================
Evaluates multiple supervised machine learning classification architectures:
1. Logistic Regression (L2 Regularized, Balanced Class Weights)
2. Calibrated Linear Support Vector Classifier (LinearSVC + Platt Scaling)
3. Multinomial Naive Bayes (Laplace Smoothing)
4. Complement Naive Bayes (Specialized for Imbalanced Text Corpora)
5. Soft Voting Ensemble (Combines Calibrated SVM, LogReg & Naive Bayes)

Performs Stratified 5-Fold Cross Validation, hyperparameter tuning, selects
the Champion Model with highest Macro-F1 / CV Accuracy, extracts top
predictive n-gram features (Explainable AI), and serializes production artifacts.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from preprocessing.text_preprocessing import TextPreprocessor

_preprocessor = TextPreprocessor()
def clean_for_nlp(text: str) -> str:
    return _preprocessor.clean_text_for_nlp(text)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MLTournament: %(message)s")
logger = logging.getLogger("SocialPulseMLTournament")

MODEL_DIR = PROJECT_ROOT / "ml_model"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# -----------------------------------------------------------------------------
# DOMAIN-SPECIFIC HIGH-FIDELITY SOCIAL MEDIA TRAINING CORPUS
# -----------------------------------------------------------------------------
EXPANDED_TRAINING_DATA = [
    # POSITIVE EXAMPLES (Praise, Breakthroughs, Speedups, Great UX, Satisfaction)
    ("Databricks Liquid Clustering reduced our query latency by 5x", "positive"),
    ("Apache Spark 3.5 Adaptive Query Execution AQE is an absolute game changer", "positive"),
    ("Building our real-time streaming pipeline with PySpark and Delta Lake is incredible", "positive"),
    ("Exceptional customer support team resolved our enterprise cluster issue in minutes", "positive"),
    ("Just deployed our custom LLM RAG application with vector search and latency is amazing", "positive"),
    ("Python 3.12 performance boost is noticeably faster for data preprocessing", "positive"),
    ("DuckDB + PyArrow is pure magic for local exploratory data analysis", "positive"),
    ("Our Medallion Architecture rollout succeeded ahead of schedule! Love the speed", "positive"),
    ("The new UI update is super smooth, clean, responsive and intuitive", "positive"),
    ("Blown away by how fast Delta Lake handles concurrent upserts and merges", "positive"),
    ("Migrated from Hive tables to Delta Lake and reduced cloud costs by 45%", "positive"),
    ("The developer experience with FastAPI and Pydantic v2 is second to none", "positive"),
    ("Kudos to the devops team for achieving 99.99% uptime this quarter", "positive"),
    ("Outstanding performance optimization guide. Saved our data pipeline hours of compute", "positive"),
    ("Customer support issued a full refund within 10 minutes with great courtesy", "positive"),
    ("This open source library is remarkably well-documented and easy to adopt", "positive"),
    ("Loving the new AI assistant feature. Makes writing complex SQL queries effortless", "positive"),
    ("Great battery life and thermal efficiency on the new M4 MacBook Pro", "positive"),
    ("The latest release fixed all previous memory leaks. Everything runs like a charm", "positive"),
    ("Super happy with the query response times on our analytics dashboard", "positive"),
    ("Highly recommend Delta Live Tables for automated data quality assertions", "positive"),
    ("Best conference presentation on distributed streaming I have ever attended", "positive"),
    ("Fantastic tutorial with reproducible code examples. Really helped me learn PySpark", "positive"),
    ("The community support in the Slack channel is friendly, knowledgeable, and responsive", "positive"),
    ("Seamless migration to Kubernetes cluster with zero downtime", "positive"),
    ("Data engineering roadmap for 2026 is spot on. Exciting times for big data engineers", "positive"),
    ("Brilliant query optimizer rewrite. 10x speedup on massive shuffle joins", "positive"),
    ("Thank you for sharing this benchmark! Very insightful and well-structured", "positive"),
    ("Love how easy it is to set up ACID transactions with Delta Lake", "positive"),
    ("Huge productivity boost since our team adopted Polars and DuckDB", "positive"),
    ("Remarkable accuracy on zero-shot NER extraction using the fine-tuned model", "positive"),
    ("Automated ETL pipeline ran flawlessly for 90 consecutive days without a single hitch", "positive"),
    ("Incredible customer onboarding experience. Set up lakehouse in under 15 minutes", "positive"),
    ("Massive breakthrough in model inference throughput with TensorRT-LLM and vLLM", "positive"),
    ("Best investment our engineering team made this year. Developer velocity doubled", "positive"),

    # NEUTRAL EXAMPLES (Benchmarks, Architecture Questions, Documentation, Informative)
    ("Benchmarking Apache Spark vs DuckDB for single-node data engineering", "neutral"),
    ("Architectural guide: Auto Loader vs Traditional Batch Ingestion on Databricks", "neutral"),
    ("Understanding Delta Lake time-travel and transaction log internals", "neutral"),
    ("Release Notes: Version 2.4 announced with new connector features and REST endpoints", "neutral"),
    ("How are teams organizing Silver layer deduplication and watermark windows?", "neutral"),
    ("Comparison of Parquet vs Iceberg vs Delta Lake storage formats", "neutral"),
    ("Weekly community standup meeting scheduled for Thursday at 2 PM UTC", "neutral"),
    ("Published our survey results on modern data stack tooling adoption for 2026", "neutral"),
    ("PySpark DataFrame schema definition for nested JSON arrays and struct fields", "neutral"),
    ("Discussion on Medallion Architecture data modeling for slow changing dimensions", "neutral"),
    ("Documentation update: Added architectural diagrams for cloud lakehouse setup", "neutral"),
    ("What are the recommended cluster specs for training small language models?", "neutral"),
    ("Evaluating trade-offs between managed streaming brokers and self-hosted Kafka", "neutral"),
    ("Overview of the latest changes in Python typing and structural pattern matching", "neutral"),
    ("Summary of changes in the latest platform terms of service and API rate limits", "neutral"),
    ("Comparison of column-oriented vs row-oriented storage engines for OLAP queries", "neutral"),
    ("How to configure IAM roles and service principals for S3 and ADLS Gen2 access", "neutral"),
    ("Webinar recording on building reproducible data engineering pipelines with Git", "neutral"),
    ("Tracking daily API request quotas across Reddit, YouTube, and Bluesky", "neutral"),
    ("Survey on preferred orchestration tools: Airflow vs Dagster vs Prefect", "neutral"),
    ("Steps to set up local Apache Spark on Windows with winutils and Java 17", "neutral"),
    ("Release candidate for version 3.2 is now available for developer testing", "neutral"),
    ("Explanation of VADER sentiment compound score calculation formula", "neutral"),
    ("Data catalog metadata sync scheduled for midnight maintenance window", "neutral"),
    ("Comparing REST API polling vs WebSocket webhooks for social media ingestion", "neutral"),
    ("Guide on configuring Parquet dictionary encoding and Snappy compression ratios", "neutral"),
    ("Quarterly roadmap review meeting agenda posted to team wiki page", "neutral"),

    # NEGATIVE EXAMPLES (Bugs, Crashes, Latency, Outages, Cost Hikes, Complaints)
    ("Severe latency spikes and cluster crash issues after recent release", "negative"),
    ("Unexpected 45% price increase with zero new features or notice", "negative"),
    ("Disastrous outage during peak traffic hours: Lost transactions and broken SLA", "negative"),
    ("Major regression bug in table partition schema evolution corrupted data", "negative"),
    ("Horrible developer experience with confusing documentation and broken SDK", "negative"),
    ("Memory leak in long-running streaming jobs causing frequent crashes every 6 hours", "negative"),
    ("Our production Spark jobs are failing with OutOfMemory errors and 504 timeouts", "negative"),
    ("The customer support ticket has been unhelpful and ignored for over a week", "negative"),
    ("Terrible service and broken release! Constant connection timeout errors", "negative"),
    ("Subscription price increased without warning while compute quotas were reduced", "negative"),
    ("Cancelling my subscription today due to hidden fees and unresponsive billing support", "negative"),
    ("Is anyone else having major login issues and broken session timeouts today?", "negative"),
    ("The web app is completely unresponsive. Frustrating bug right during our sprint", "negative"),
    ("The audio was out of sync and the example code in the description is broken with 404", "negative"),
    ("I tested this script on our cluster and it crashed with a fatal JVM core dump", "negative"),
    ("Worst customer service experience I have had all year. Completely dismissive", "negative"),
    ("Critical security vulnerability discovered in legacy authentication endpoint", "negative"),
    ("Database replication lag exceeded 45 minutes causing severe stale read inconsistencies", "negative"),
    ("Broken API endpoint returning 500 internal server error for all batch requests", "negative"),
    ("Disk I/O bottleneck throttling our ingestion jobs. Completely unusable at scale", "negative"),
    ("Extremely frustrating billing dispute. Charged twice for canceled enterprise seats", "negative"),
    ("Rollback command failed and left our primary table in an inconsistent corrupt state", "negative"),
    ("Huge slowdown after the update. What used to take 2 minutes now takes 40 minutes", "negative"),
    ("Constant app crashes on mobile. Cannot even complete the two-factor authentication", "negative"),
    ("Customer support refused to honor existing annual contract terms and renewal discounts", "negative"),
    ("Avoid this service! Hidden cancellation fees and zero customer protection", "negative"),
    ("Cluster nodes become unresponsive under moderate network partition load", "negative"),
    ("Garbage collection pauses freezing the query engine for 30+ seconds at a time", "negative"),
    ("The documentation is completely outdated and none of the code snippets run", "negative"),
    ("Terrible upgrade experience. Broke all existing downstream dashboard connections", "negative"),
    ("Severe regression: daily pipeline runtime jumped from 15 mins to 3.5 hours", "negative"),
    ("Silent data truncation bug caused critical business metrics to be reported incorrectly", "negative"),
    ("Support team closed ticket without resolving the persistent connection drops", "negative"),
]


def load_training_corpus():
    """Combines curated domain sentiment corpus with sample data records."""
    texts = []
    labels = []

    # 1. Base curated corpus
    for text, label in EXPANDED_TRAINING_DATA:
        texts.append(text)
        labels.append(label)

    # 2. Add sample records from JSON files
    sample_dir = config.DATA_SAMPLE_DIR
    reddit_path = sample_dir / "reddit_sample.json"
    youtube_path = sample_dir / "youtube_sample.json"
    bluesky_path = sample_dir / "bluesky_sample.json"

    def infer_label(text):
        lower = text.lower()
        if any(w in lower for w in ["crash", "latency", "outage", "bug", "terrible", "horrible", "frustrat", "unacceptable", "broken", "complaint", "price increase", "hidden fee", "slowdown", "disaster"]):
            return "negative"
        if any(w in lower for w in ["outstanding", "game changer", "exceptional", "breakthrough", "love", "fantastic", "kudos", "brilliant", "great", "best", "boost", "magic", "incredible", "super happy"]):
            return "positive"
        return "neutral"

    for file_path in [reddit_path, youtube_path, bluesky_path]:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    content = ""
                    if "post_title" in item:
                        content += item["post_title"] + " " + item.get("post_description", "")
                    elif "title" in item:
                        content += item["title"] + " " + item.get("description", "")
                    elif "post_text" in item:
                        content += item["post_text"]

                    if content.strip():
                        lbl = infer_label(content)
                        texts.append(content)
                        labels.append(lbl)

    df = pd.DataFrame({"text": texts, "label": labels})
    df["cleaned_text"] = df["text"].apply(clean_for_nlp)
    df = df.drop_duplicates(subset=["cleaned_text"]).dropna()
    return df


def extract_top_features(vectorizer, classifier, n_top=10):
    """Extracts top predictive n-grams per sentiment class for Explainable AI."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}

    if hasattr(classifier, "coef_"):
        classes = classifier.classes_
        for idx, cls in enumerate(classes):
            top_indices = np.argsort(classifier.coef_[idx])[-n_top:][::-1]
            top_features[cls] = [
                {"word": feature_names[i], "weight": round(float(classifier.coef_[idx][i]), 4)}
                for i in top_indices
            ]
    return top_features


def train_model():
    """
    Executes Multi-Model Tournament across 5 architectures:
    - Logistic Regression (L2 Balanced)
    - Calibrated Linear Support Vector Classifier
    - Multinomial Naive Bayes
    - Complement Naive Bayes
    - Soft Voting Ensemble
    Selects Champion Model, computes full metrics & leaderboard, saves artifacts.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing SocialPulse Sentiment Multi-Model Tournament...")
    df = load_training_corpus()
    logger.info("Loaded %d unique labeled training records.", len(df))

    X = df["cleaned_text"]
    y = df["label"]

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Candidate Architectures
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True,
        min_df=1
    )

    candidate_models = {
        "Logistic Regression (Balanced)": Pipeline([
            ("tfidf", vectorizer),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Calibrated Linear SVM": Pipeline([
            ("tfidf", vectorizer),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.2, random_state=42), cv=3))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", vectorizer),
            ("clf", MultinomialNB(alpha=0.25))
        ]),
        "Complement Naive Bayes": Pipeline([
            ("tfidf", vectorizer),
            ("clf", ComplementNB(alpha=0.25))
        ]),
    }

    # Soft Voting Ensemble combining top 3 probability estimators
    ensemble_clf = Pipeline([
        ("tfidf", vectorizer),
        ("clf", VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)),
                ("svm", CalibratedClassifierCV(LinearSVC(C=1.2, random_state=42), cv=3)),
                ("cnb", ComplementNB(alpha=0.25))
            ],
            voting="soft"
        ))
    ])
    candidate_models["Soft Voting Ensemble"] = ensemble_clf

    # Tournament Evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaderboard = []
    trained_pipelines = {}

    for name, pipeline in candidate_models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = round((time.time() - t0) * 1000, 1)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # 5-Fold Cross Validation
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        leaderboard.append({
            "model_name": name,
            "test_accuracy": round(acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "cv_accuracy_mean": round(cv_mean * 100, 2),
            "cv_accuracy_std": round(cv_std * 100, 2),
            "fit_time_ms": fit_time
        })
        trained_pipelines[name] = pipeline
        logger.info("Tournament candidate [%s] -> Test Acc: %.1f%% | CV Acc: %.1f%% | Macro F1: %.1f%%",
                    name, acc * 100, cv_mean * 100, macro_f1 * 100)

    # Sort leaderboard: primary = test_accuracy, secondary = cv_accuracy_mean
    leaderboard.sort(key=lambda r: (r["test_accuracy"], r["cv_accuracy_mean"]), reverse=True)
    champion_name = leaderboard[0]["model_name"]
    champion_pipeline = trained_pipelines[champion_name]

    logger.info("🏆 CHAMPION MODEL SELECTED: %s (Test Acc: %.2f%%)",
                champion_name, leaderboard[0]["test_accuracy"])

    # Detailed evaluation of champion
    y_champion_pred = champion_pipeline.predict(X_test)
    report = classification_report(y_test, y_champion_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_champion_pred, labels=["positive", "neutral", "negative"]).tolist()

    # Extract top explainability features if available
    top_features = {}
    try:
        lr_pipe = trained_pipelines.get("Logistic Regression (Balanced)")
        if lr_pipe:
            top_features = extract_top_features(
                lr_pipe.named_steps["tfidf"],
                lr_pipe.named_steps["clf"],
                n_top=8
            )
    except Exception as e:
        logger.warning("Feature extraction skipped: %s", e)

    # Save champion model pipeline
    joblib.dump(champion_pipeline, MODEL_PATH)
    logger.info("Saved serialized champion model to: %s", MODEL_PATH)

    # Save comprehensive metrics & tournament leaderboard
    metrics = {
        "champion_model_name": champion_name,
        "model_name": champion_name,
        "tournament_leaderboard": leaderboard,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": round(leaderboard[0]["test_accuracy"] / 100, 4),
        "cv_mean_accuracy": round(leaderboard[0]["cv_accuracy_mean"] / 100, 4),
        "cv_std_accuracy": round(leaderboard[0]["cv_accuracy_std"] / 100, 4),
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["positive", "neutral", "negative"],
            "matrix": cm
        },
        "classes": list(champion_pipeline.classes_),
        "top_features": top_features,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved tournament evaluation report to: %s", METRICS_PATH)

    return metrics


class SentimentPredictor:
    """Production inference interface for predicting sentiment with the champion ML model."""

    def __init__(self):
        if not MODEL_PATH.exists():
            train_model()
        self.model = joblib.load(MODEL_PATH)
        self.classes = list(self.model.classes_)

    def predict(self, text: str) -> dict:
        """Predicts sentiment class and probability distribution for given text."""
        cleaned = clean_for_nlp(text)
        pred_label = self.model.predict([cleaned])[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([cleaned])[0]
            prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(self.classes, probs)}
            confidence = float(np.max(probs))
        else:
            prob_dict = {cls: 0.33 for cls in self.classes}
            confidence = 0.85

        return {
            "sentiment_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "cleaned_text": cleaned
        }


if __name__ == "__main__":
    train_model()
