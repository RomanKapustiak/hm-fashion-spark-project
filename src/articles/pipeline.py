"""Articles pipeline orchestration: extract -> preprocess -> save."""

from pyspark.sql import SparkSession

from src.articles.extraction import run_extraction
from src.articles.preprocessing import run_preprocessing


def run(
    spark: SparkSession,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> None:
    """Execute full articles pipeline and save parquet output."""
    raw_df = run_extraction(spark, raw_dir=raw_dir)
    clean_df = run_preprocessing(raw_df)

    out_path = f"{processed_dir}/articles"
    print(f"saving to {out_path} ...")
    clean_df.write.mode("overwrite").parquet(out_path)
    print("done.")
