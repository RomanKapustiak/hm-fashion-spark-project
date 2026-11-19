"""Customer-level feature table for the classification task.

Question: will a customer buy anything during the next 4 weeks?

- cutoff date 2020-08-26: features use only transactions BEFORE the cutoff,
  label = 1 if the customer bought anything from the cutoff to 2020-09-22
  (the last 4 weeks of the dataset);
- one row per customer that has at least one purchase before the cutoff.

Output: data/ml/customers_features (parquet) with the shared `split` column.
"""

from __future__ import annotations

import re
from typing import List

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from src.analysis.common import CUSTOMERS, LABEL_COL, assign_split

PROCESSED_DIR = "data/processed"

CUTOFF = "2020-08-26"
LABEL_END = "2020-09-22"
WINDOWS_WEEKS = (1, 4, 12, 26)
# the same 4 weeks one year earlier - seasonal buying habits
LAST_YEAR_START = "2019-08-26"
LAST_YEAR_END = "2019-09-22"
TOP_PRODUCT_GROUPS = 12


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _share(condition: Column) -> Column:
    return F.avg(condition.cast("double"))


def _transaction_features(history: DataFrame, articles: DataFrame) -> DataFrame:
    """Per-customer aggregates over purchase rows (one row = one item bought)."""
    cutoff = F.to_date(F.lit(CUTOFF))
    days_before = F.datediff(cutoff, F.col("t_dat"))

    index_groups = sorted(r[0] for r in articles.select("index_group_name").distinct().collect() if r[0])
    product_groups = [
        r[0] for r in articles.groupBy("product_group_name").count().orderBy(F.desc("count")).limit(TOP_PRODUCT_GROUPS).collect()
        if r[0]
    ]

    enriched = history.join(F.broadcast(articles), on="article_id", how="left")

    aggs: List[Column] = [
        F.count(F.lit(1)).alias("n_items"),
        F.sum("price").alias("total_spend"),
        F.avg("price").alias("avg_price"),
        F.max("price").alias("max_price"),
        F.coalesce(F.stddev("price"), F.lit(0.0)).alias("std_price"),
        F.min(days_before).alias("days_since_last"),
        F.max(days_before).alias("days_since_first"),
        _share(F.col("sales_channel_id") == 2).alias("online_share"),
        _share(F.dayofweek("t_dat").isin(1, 7)).alias("weekend_share"),
        F.sum(F.col("t_dat").between(LAST_YEAR_START, LAST_YEAR_END).cast("int")).alias("n_items_same_season_last_year"),
        F.sum(((days_before > 28) & (days_before <= 56)).cast("int")).alias("n_items_prev_4w"),
        F.size(F.collect_set("article_id")).alias("n_unique_articles"),
        F.size(F.collect_set("product_type_name")).alias("n_unique_product_types"),
        F.size(F.collect_set("department_name")).alias("n_unique_departments"),
    ]
    for weeks in WINDOWS_WEEKS:
        in_window = days_before <= 7 * weeks
        aggs.append(F.sum(in_window.cast("int")).alias(f"n_items_{weeks}w"))
        aggs.append(F.sum(F.when(in_window, F.col("price")).otherwise(0.0)).alias(f"spend_{weeks}w"))
    for group in index_groups:
        aggs.append(_share(F.col("index_group_name") == group).alias(f"share_index_{_slug(group)}"))
    for group in product_groups:
        aggs.append(_share(F.col("product_group_name") == group).alias(f"share_product_{_slug(group)}"))

    features = enriched.groupBy("customer_id").agg(*aggs)
    return (
        features
        .withColumn("trend_4w", (F.col("n_items_4w") + 1) / (F.col("n_items_prev_4w") + 1))
        .withColumn("items_per_unique_article", F.col("n_items") / F.col("n_unique_articles"))
    )


def _visit_features(history: DataFrame) -> DataFrame:
    """Per-customer aggregates over shopping days (several items on one day = one visit)."""
    cutoff = F.to_date(F.lit(CUTOFF))
    visits = history.select("customer_id", "t_dat").distinct().withColumn("days_before", F.datediff(cutoff, "t_dat"))
    aggs = [F.count(F.lit(1)).alias("n_visits")]
    for weeks in WINDOWS_WEEKS:
        aggs.append(F.sum((F.col("days_before") <= 7 * weeks).cast("int")).alias(f"n_visits_{weeks}w"))
    aggs.append((F.max("days_before") - F.min("days_before")).alias("_active_span"))
    return (
        visits.groupBy("customer_id").agg(*aggs)
        # average gap between visits; single-visit customers get the gap "since first visit"
        .withColumn(
            "avg_days_between_visits",
            F.when(F.col("n_visits") > 1, F.col("_active_span") / (F.col("n_visits") - 1)),
        )
        .withColumn("single_visit", (F.col("n_visits") == 1).cast("int"))
        .drop("_active_span")
    )


def _customer_profile(customers: DataFrame) -> DataFrame:
    postal_sizes = customers.groupBy("postal_code").agg(F.count(F.lit(1)).alias("postal_code_customers"))
    return (
        customers
        .join(postal_sizes, on="postal_code", how="left")
        .select(
            "customer_id", "age", "FN", "Active",
            "club_member_status", "fashion_news_frequency", "postal_code_customers",
        )
    )


def build(spark: SparkSession, processed_dir: str = PROCESSED_DIR) -> DataFrame:
    transactions = spark.read.parquet(f"{processed_dir}/transactions").select(
        "t_dat", "customer_id", "article_id", "price", "sales_channel_id"
    )
    articles = spark.read.parquet(f"{processed_dir}/articles").select(
        "article_id", "index_group_name", "product_group_name", "product_type_name", "department_name"
    )
    customers = spark.read.parquet(f"{processed_dir}/customers")

    history = transactions.filter(F.col("t_dat") < F.lit(CUTOFF))
    labels = (
        transactions
        .filter(F.col("t_dat").between(CUTOFF, LABEL_END))
        .select("customer_id").distinct()
        .withColumn(LABEL_COL, F.lit(1.0))
    )

    df = (
        _transaction_features(history, articles)
        .join(_visit_features(history), on="customer_id", how="inner")
        .join(_customer_profile(customers), on="customer_id", how="left")
        .join(labels, on="customer_id", how="left")
        .fillna({LABEL_COL: 0.0})
    )
    for name, dtype in df.dtypes:
        if dtype in ("int", "bigint", "float"):
            df = df.withColumn(name, F.col(name).cast("double"))
    for name in CUSTOMERS.categorical:
        df = df.withColumn(name, F.coalesce(F.col(name), F.lit("UNKNOWN")))
    return assign_split(df, CUSTOMERS.group_col)
