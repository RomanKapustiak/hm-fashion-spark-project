"""Article-level feature table for the regression task.

Question: what is the price of an article given its characteristics?

- one row per article that was sold at least once;
- label = average selling price of the article;
- features: all descriptive attributes of the article, its text description and
  the profile of its demand (who / where / when buys it);
- price-derived values are NOT used as features (that would leak the target);
- split is done by product_code, so colour variants of one product never end up
  in both train and test.

Output: data/ml/articles_features (parquet) with the shared `split` column.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.analysis.common import ARTICLES, LABEL_COL, assign_split

PROCESSED_DIR = "data/processed"


def _demand_profile(transactions: DataFrame, customers: DataFrame) -> DataFrame:
    data_start = transactions.select(F.min("t_dat")).first()[0]
    month = F.month("t_dat")
    sales = transactions.join(customers, on="customer_id", how="left")
    profile = sales.groupBy("article_id").agg(
        F.avg("price").alias(LABEL_COL),
        F.count(F.lit(1)).alias("n_sales"),
        F.countDistinct("customer_id").alias("n_buyers"),
        F.avg((F.col("sales_channel_id") == 2).cast("double")).alias("online_share"),
        F.avg("age").alias("buyer_age_mean"),
        F.coalesce(F.stddev("age"), F.lit(0.0)).alias("buyer_age_std"),
        F.avg((F.col("age") < 25).cast("double")).alias("buyer_under_25_share"),
        F.avg((F.col("club_member_status") == "ACTIVE").cast("double")).alias("buyer_club_active_share"),
        F.avg((F.col("fashion_news_frequency") == "REGULARLY").cast("double")).alias("buyer_news_regular_share"),
        F.datediff(F.min("t_dat"), F.lit(data_start)).alias("first_sale_day"),
        F.datediff(F.max("t_dat"), F.lit(data_start)).alias("last_sale_day"),
        F.avg(month.isin(6, 7, 8).cast("double")).alias("summer_share"),
        F.avg(month.isin(12, 1, 2).cast("double")).alias("winter_share"),
    )
    return (
        profile
        .withColumn("sale_span_days", F.col("last_sale_day") - F.col("first_sale_day") + 1)
        .withColumn("sales_per_day", F.col("n_sales") / F.col("sale_span_days"))
        .withColumn("sales_per_buyer", F.col("n_sales") / F.col("n_buyers"))
    )


def build(spark: SparkSession, processed_dir: str = PROCESSED_DIR) -> DataFrame:
    transactions = spark.read.parquet(f"{processed_dir}/transactions").select(
        "t_dat", "customer_id", "article_id", "price", "sales_channel_id"
    )
    customers = spark.read.parquet(f"{processed_dir}/customers").select(
        "customer_id", "age", "club_member_status", "fashion_news_frequency"
    )
    articles = spark.read.parquet(f"{processed_dir}/articles")

    product_variants = Window.partitionBy("product_code")
    attributes = (
        articles
        .select(
            "article_id", "product_code", *ARTICLES.categorical,
            F.concat_ws(" ", F.coalesce("prod_name", F.lit("")), F.coalesce("detail_desc", F.lit(""))).alias("text"),
        )
        .withColumn("n_product_variants", F.count(F.lit(1)).over(product_variants))
    )

    df = attributes.join(_demand_profile(transactions, customers), on="article_id", how="inner")
    for name, dtype in df.dtypes:
        if name != "product_code" and dtype in ("int", "bigint", "float"):
            df = df.withColumn(name, F.col(name).cast("double"))
    for name in ARTICLES.categorical:
        df = df.withColumn(name, F.coalesce(F.col(name), F.lit("unknown")))
    return assign_split(df, ARTICLES.group_col)
