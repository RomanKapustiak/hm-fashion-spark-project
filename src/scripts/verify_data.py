"""
H&M Fashion Data Verification
==============================
Verifies that all datasets are accessible and load correctly with PySpark.
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name="HM-Data-Verification"):
    """Create and return a configured SparkSession."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def verify_datasets(spark, data_dir="data/raw"):
    """Load and verify all H&M datasets. Returns dict of {name: success}."""
    datasets = {
        "articles":     f"{data_dir}/articles.csv",
        "customers":    f"{data_dir}/customers.csv",
        "transactions": f"{data_dir}/transactions_train.csv",
    }

    results = {}

    for name, path in datasets.items():
        print(f"\n{'─' * 60}")
        print(f"  Loading: {name}  ({path})")
        print(f"{'─' * 60}")

        try:
            df = spark.read.csv(path, header=True, inferSchema=True)

            row_count = df.count()
            col_count = len(df.columns)

            print(f"  ✓  Rows:    {row_count:,}")
            print(f"  ✓  Columns: {col_count}")
            print(f"  ✓  Columns list: {df.columns}")
            print(f"\n  Schema:")
            df.printSchema()
            print(f"  Sample (5 rows):")
            df.show(5, truncate=False)

            results[name] = True

        except Exception as e:
            print(f"  ✗  FAILED to load {name}: {e}")
            results[name] = False

    return results


def print_summary(results):
    """Print a verification summary."""
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)

    all_ok = True
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  🎉  All datasets loaded successfully!")
    else:
        print("\n  ⚠️   Some datasets failed to load. Check paths above.")

    print("=" * 60)
    return all_ok


def run(data_dir="data/raw"):
    """Full verification pipeline."""
    print("=" * 60)
    print("  H&M Fashion Data — Verification Script")
    print("=" * 60)

    spark = create_spark_session()
    results = verify_datasets(spark, data_dir)
    success = print_summary(results)
    spark.stop()
    return success


if __name__ == "__main__":
    run()
