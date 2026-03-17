"""Extraction stage for articles.csv."""

from pyspark.sql import DataFrame, SparkSession

from src.articles.schema import load_articles


def run_extraction(spark: SparkSession, raw_dir: str = "data/raw") -> DataFrame:
    """Load raw articles data and print a quick sanity check."""
    path = f"{raw_dir}/articles.csv"
    print(f"\n=== EXTRACTION: {path} ===")

    df = load_articles(spark, path)

    df.printSchema()
    print(f"rows: {df.count():,}  cols: {df.columns}")
    df.show(5, truncate=False)

    return df
