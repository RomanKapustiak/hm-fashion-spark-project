"""Builds the shared feature tables for the modelling stage.

Run (after main.py has produced data/processed):
    docker compose run --rm spark-app python src/analysis/build_features.py [customers|articles]
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.analysis import features_articles, features_customers  # noqa: E402
from src.analysis.common import ARTICLES, CUSTOMERS, LABEL_COL, SPLIT_COL, DatasetSpec  # noqa: E402
from src.spark_utils import create_spark_session  # noqa: E402

BUILDERS = {
    "customers": (CUSTOMERS, features_customers.build),
    "articles": (ARTICLES, features_articles.build),
}


def _summary(df: DataFrame, spec: DatasetSpec) -> None:
    print(f"\n=== {spec.name}: {len(df.columns)} columns ===")
    print(df.columns)
    df.groupBy(SPLIT_COL).agg(
        F.count(F.lit(1)).alias("rows"),
        F.avg(LABEL_COL).alias("label_mean"),
    ).orderBy(SPLIT_COL).show()


def main(names) -> None:
    spark = create_spark_session("HM-Build-Features")
    try:
        for name in names:
            spec, build = BUILDERS[name]
            print(f"\nbuilding {spec.name} ...")
            build(spark).repartition(16).write.mode("overwrite").parquet(spec.path)
            _summary(spark.read.parquet(spec.path), spec)
    finally:
        spark.stop()


if __name__ == "__main__":
    main(sys.argv[1:] or list(BUILDERS))
