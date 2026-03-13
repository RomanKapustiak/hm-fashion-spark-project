# extraction module for transactions_train.csv
# stage 3, steps 2-4

from pyspark.sql import SparkSession, DataFrame
from src.transactions.schema import load_transactions


def run_extraction(spark: SparkSession, raw_dir: str = "data/raw") -> DataFrame:
    """load transactions and verify it actually worked"""
    path = f"{raw_dir}/transactions_train.csv"
    print(f"\n=== EXTRACTION: {path} ===")

    df = load_transactions(spark, path)

    # verify - run an action so spark actually reads the file
    df.printSchema()
    print(f"rows: {df.count():,}  cols: {df.columns}")
    df.show(5, truncate=False)

    return df
