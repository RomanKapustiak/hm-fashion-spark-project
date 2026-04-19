"""Dmytro transformation pipeline (time-series dynamics and retention)."""

from __future__ import annotations

import io
import os
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pyspark.sql.window import Window

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import create_spark_session


SECONDS_IN_DAY = 86400


def _prepare_query_3_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate retention gaps into readable buckets and compute share."""
    if df.empty:
        return pd.DataFrame(
            columns=["bucket_label", "customers_count", "customers_share_pct"]
        )

    bucket_order = [
        "0 days (same day)",
        "1-7 days",
        "8-30 days",
        "31-90 days",
        "91-180 days",
        "181-365 days",
        "366+ days",
    ]
    prepared = df.copy()
    prepared["bucket_label"] = "366+ days"
    prepared.loc[prepared["days_between"] == 0, "bucket_label"] = "0 days (same day)"
    prepared.loc[
        prepared["days_between"].between(1, 7, inclusive="both"), "bucket_label"
    ] = "1-7 days"
    prepared.loc[
        prepared["days_between"].between(8, 30, inclusive="both"), "bucket_label"
    ] = "8-30 days"
    prepared.loc[
        prepared["days_between"].between(31, 90, inclusive="both"), "bucket_label"
    ] = "31-90 days"
    prepared.loc[
        prepared["days_between"].between(91, 180, inclusive="both"), "bucket_label"
    ] = "91-180 days"
    prepared.loc[
        prepared["days_between"].between(181, 365, inclusive="both"), "bucket_label"
    ] = "181-365 days"

    grouped = (
        prepared.groupby("bucket_label", as_index=False)["customers_count"]
        .sum()
        .set_index("bucket_label")
        .reindex(bucket_order, fill_value=0)
        .reset_index()
    )
    total_customers = grouped["customers_count"].sum()
    grouped["customers_share_pct"] = (
        grouped["customers_count"] / total_customers * 100 if total_customers else 0.0
    )
    return grouped


def query_1_weekly_transaction_dynamics(transactions_df: DataFrame) -> DataFrame:
    """Weekly total transaction count across the full dataset period."""
    weekly = (
        transactions_df.withColumn("year", F.year("t_dat"))
        .withColumn("week", F.weekofyear("t_dat"))
        .groupBy("year", "week")
        .agg(F.count("*").alias("transactions_count"))
    )
    wow_window = Window.orderBy("year", "week")
    yoy_window = Window.partitionBy("week").orderBy("year")
    return (
        weekly.withColumn("prev_week_transactions", F.lag("transactions_count").over(wow_window))
        .withColumn("prev_year_transactions", F.lag("transactions_count").over(yoy_window))
        .withColumn(
            "wow_change_pct",
            F.when(
                F.col("prev_week_transactions") > 0,
                (F.col("transactions_count") - F.col("prev_week_transactions"))
                / F.col("prev_week_transactions")
                * 100.0,
            ),
        )
        .withColumn(
            "yoy_change_pct",
            F.when(
                F.col("prev_year_transactions") > 0,
                (F.col("transactions_count") - F.col("prev_year_transactions"))
                / F.col("prev_year_transactions")
                * 100.0,
            ),
        )
        .orderBy("year", "week")
    )


def query_2_winter_vs_summer_spike(
    transactions_df: DataFrame,
    articles_df: DataFrame,
) -> DataFrame:
    """Sales spike by product group in winter (Dec-Feb) versus summer (Jun-Aug)."""
    seasonal_sales = (
        transactions_df.withColumn("month", F.month("t_dat"))
        .filter(F.col("month").isin([12, 1, 2, 6, 7, 8]))
        .withColumn(
            "season",
            F.when(F.col("month").isin([12, 1, 2]), F.lit("Winter")).otherwise(
                F.lit("Summer")
            ),
        )
        .join(articles_df.select("article_id", "product_group_name"), on="article_id")
        .groupBy("product_group_name", "season")
        .agg(F.count("*").alias("sales_count"))
    )

    return (
        seasonal_sales.groupBy("product_group_name")
        .pivot("season", ["Winter", "Summer"])
        .sum("sales_count")
        .na.fill(0)
        .withColumnRenamed("Winter", "winter_sales")
        .withColumnRenamed("Summer", "summer_sales")
        .withColumn("sales_spike", F.col("winter_sales") - F.col("summer_sales"))
        .withColumn(
            "seasonal_delta_pct",
            F.when(
                F.greatest(F.col("winter_sales"), F.col("summer_sales")) > 0,
                F.col("sales_spike")
                / F.greatest(F.col("winter_sales"), F.col("summer_sales"))
                * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "seasonality_index",
            F.when(
                (F.col("winter_sales") + F.col("summer_sales")) > 0,
                F.col("winter_sales")
                / (F.col("winter_sales") + F.col("summer_sales"))
                * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .orderBy(F.desc("sales_spike"), "product_group_name")
    )


def query_3_retention_gap_distribution(transactions_df: DataFrame) -> DataFrame:
    """Distribution of days between first and second purchase per customer."""
    purchase_order = Window.partitionBy("customer_id").orderBy("t_dat")
    first_two = (
        transactions_df.select("customer_id", "t_dat")
        .withColumn("purchase_rank", F.row_number().over(purchase_order))
        .filter(F.col("purchase_rank") <= 2)
    )

    first_second = (
        first_two.groupBy("customer_id")
        .agg(
            F.min("t_dat").alias("first_purchase_date"),
            F.max("t_dat").alias("second_purchase_date"),
            F.count("*").alias("purchase_rows"),
        )
        .filter(F.col("purchase_rows") == 2)
        .withColumn(
            "days_between",
            F.datediff("second_purchase_date", "first_purchase_date"),
        )
    )

    return (
        first_second.groupBy("days_between")
        .agg(F.count("*").alias("customers_count"))
        .orderBy("days_between")
    )


def query_4_top_peak_days_month_share(transactions_df: DataFrame) -> DataFrame:
    """Top 10 highest daily sales peaks and their share of monthly transaction volume."""
    daily_counts = (
        transactions_df.filter(F.col("t_dat").isNotNull())
        .withColumn("month_start", F.date_trunc("month", F.col("t_dat")))
        .groupBy("t_dat", "month_start")
        .agg(F.count("*").alias("daily_transactions"))
    )
    month_stats = daily_counts.groupBy("month_start").agg(
        F.sum("daily_transactions").alias("monthly_revenue"),
        F.expr("percentile_approx(daily_transactions, 0.5)").alias("monthly_median_daily"),
        F.avg("daily_transactions").alias("monthly_mean_daily"),
        F.stddev_pop("daily_transactions").alias("monthly_std_daily"),
    )
    ranked = (
        daily_counts.join(month_stats, on="month_start", how="left")
        .withColumn(
            "monthly_share",
            F.col("daily_transactions") / F.col("monthly_revenue"),
        )
        .withColumn(
            "peak_intensity_ratio",
            F.when(
                F.col("monthly_median_daily") > 0,
                F.col("daily_transactions") / F.col("monthly_median_daily"),
            ),
        )
        .withColumn(
            "z_score",
            F.when(
                F.col("monthly_std_daily") > 0,
                (F.col("daily_transactions") - F.col("monthly_mean_daily"))
                / F.col("monthly_std_daily"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "peak_rank", F.row_number().over(Window.orderBy(F.desc("daily_transactions")))
        )
    )
    return (
        ranked.filter(F.col("peak_rank") <= 10)
        .withColumnRenamed("daily_transactions", "daily_revenue")
        .drop("monthly_mean_daily", "monthly_std_daily")
        .orderBy(F.desc("daily_revenue"), "t_dat")
    )


def query_5_rolling_activity_30_40(
    transactions_df: DataFrame,
    customers_df: DataFrame,
) -> DataFrame:
    """7-day rolling average of purchase activity for customers aged 30-40."""
    daily_counts = (
        transactions_df.join(
            customers_df.filter((F.col("age") >= 30) & (F.col("age") <= 40)).select(
                "customer_id"
            ),
            on="customer_id",
        )
        .groupBy("t_dat")
        .agg(F.count("*").alias("purchases_count"))
        .orderBy("t_dat")
        .withColumn("event_ts", F.to_timestamp("t_dat"))
        .withColumn("event_seconds", F.col("event_ts").cast("long"))
    )
    rolling_window = Window.orderBy("event_seconds").rangeBetween(-6 * SECONDS_IN_DAY, 0)
    return (
        daily_counts.withColumn(
            "rolling_avg_7d", F.avg("purchases_count").over(rolling_window)
        )
        .withColumn("rolling_std_7d", F.stddev_pop("purchases_count").over(rolling_window))
        .withColumn("upper_band_2sigma", F.col("rolling_avg_7d") + 2 * F.col("rolling_std_7d"))
        .withColumn(
            "lower_band_2sigma",
            F.greatest(F.lit(0.0), F.col("rolling_avg_7d") - 2 * F.col("rolling_std_7d")),
        )
        .withColumn("is_outlier", F.col("purchases_count") > F.col("upper_band_2sigma"))
        .select(
            "t_dat",
            "purchases_count",
            "rolling_avg_7d",
            "rolling_std_7d",
            "upper_band_2sigma",
            "lower_band_2sigma",
            "is_outlier",
        )
        .orderBy("t_dat")
    )


def query_6_frequency_loyal_vs_guest(
    transactions_df: DataFrame,
    customers_df: DataFrame,
) -> DataFrame:
    """Purchase frequency by day-of-week for ACTIVE vs PRE-CREATE customers."""
    base = (
        transactions_df.join(
            customers_df.filter(
                F.col("club_member_status").isin(["ACTIVE", "PRE-CREATE"])
            ).select("customer_id", "club_member_status"),
            on="customer_id",
        )
        .withColumn("day_of_week", F.dayofweek("t_dat"))
        .groupBy("club_member_status", "day_of_week")
        .agg(F.count("*").alias("purchases_count"))
    )
    segment_window = Window.partitionBy("club_member_status")
    day_window = Window.partitionBy("day_of_week")
    return (
        base.withColumn("segment_total_purchases", F.sum("purchases_count").over(segment_window))
        .withColumn(
            "segment_day_share_pct",
            F.when(
                F.col("segment_total_purchases") > 0,
                F.col("purchases_count") / F.col("segment_total_purchases") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("day_avg_across_segments", F.avg("purchases_count").over(day_window))
        .withColumn(
            "weekday_index100",
            F.when(
                F.col("day_avg_across_segments") > 0,
                F.col("purchases_count") / F.col("day_avg_across_segments") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .drop("day_avg_across_segments")
        .orderBy("club_member_status", "day_of_week")
    )


def _append_explain_log(df: DataFrame, header: str, log_file: str) -> None:
    from contextlib import redirect_stdout

    buf = io.StringIO()

    with redirect_stdout(buf):
        df.explain(extended=True)
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 90}\n{header}\n{'=' * 90}\n")
        fh.write(buf.getvalue())


def _save_csv(df: DataFrame, output_dir: str) -> None:
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_dir)


def _plot_query_1(df: pd.DataFrame, output_path: str) -> None:
    df = df.copy()
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)
    plt.figure(figsize=(14, 6), dpi=300)
    sns.lineplot(data=df, x="year_week", y="transactions_count", marker="o", linewidth=2)
    plt.title("Weekly Transaction Dynamics Across the Full Period")
    plt.xlabel("Year-Week")
    plt.ylabel("Transaction Count")
    step = max(1, len(df) // 16)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df.iloc[pos]["year_week"] for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_query_2(df: pd.DataFrame, output_path: str) -> None:
    ordered = df.sort_values("sales_spike", ascending=False).head(12)
    y_pos = range(len(ordered))
    plt.figure(figsize=(12, 7), dpi=300)
    plt.hlines(
        y=y_pos,
        xmin=ordered["summer_sales"],
        xmax=ordered["winter_sales"],
        color="gray",
        alpha=0.7,
    )
    plt.scatter(ordered["summer_sales"], y_pos, color="#4c72b0", label="Summer")
    plt.scatter(ordered["winter_sales"], y_pos, color="#dd8452", label="Winter")
    plt.yticks(y_pos, ordered["product_group_name"])
    plt.title("Winter vs Summer Sales Spike by Product Group")
    plt.xlabel("Sales Count")
    plt.ylabel("Product Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_query_3(df: pd.DataFrame, output_path: str) -> None:
    prepared = _prepare_query_3_plot_data(df)

    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=prepared, x="bucket_label", y="customers_count", color="#4c72b0")
    plt.title("Retention Gap Between 1st and 2nd Purchase (Bucketed)")
    plt.xlabel("Days Between Purchases (Buckets)")
    plt.ylabel("Customer Count")
    plt.xticks(rotation=25, ha="right")
    for idx, row in prepared.iterrows():
        plt.text(
            idx,
            row["customers_count"],
            f"{row['customers_share_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_query_4(df: pd.DataFrame, output_path: str) -> None:
    ordered = df.sort_values("t_dat")
    plt.figure(figsize=(12, 6), dpi=300)
    sns.lineplot(data=ordered, x="t_dat", y="daily_revenue", marker="o", linewidth=2)
    for _, row in ordered.iterrows():
        plt.annotate(
            f"{row['monthly_share']:.1%}",
            (row["t_dat"], row["daily_revenue"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    plt.title("Top 10 Daily Sales Peaks and Their Monthly Share")
    plt.xlabel("Date")
    plt.ylabel("Daily Transaction Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_query_5(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(12, 6), dpi=300)
    sns.lineplot(data=df, x="t_dat", y="purchases_count", label="Daily Purchases", alpha=0.5)
    sns.lineplot(data=df, x="t_dat", y="rolling_avg_7d", label="7-Day Rolling Average", linewidth=2)
    if {"lower_band_2sigma", "upper_band_2sigma"} <= set(df.columns):
        plt.fill_between(
            df["t_dat"],
            df["lower_band_2sigma"],
            df["upper_band_2sigma"],
            color="#4c72b0",
            alpha=0.15,
            label="2σ Band",
        )
    if "is_outlier" in df.columns:
        outlier_mask = df["is_outlier"].astype(bool)
        outliers = df[outlier_mask]
        if not outliers.empty:
            plt.scatter(
                outliers["t_dat"],
                outliers["purchases_count"],
                color="#d62728",
                s=25,
                label="Outlier",
                zorder=5,
            )
    plt.title("7-Day Rolling Purchase Activity for Customers Aged 30-40")
    plt.xlabel("Date")
    plt.ylabel("Purchases")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_query_6(df: pd.DataFrame, output_path: str) -> None:
    weekday_names = {
        1: "Sun",
        2: "Mon",
        3: "Tue",
        4: "Wed",
        5: "Thu",
        6: "Fri",
        7: "Sat",
    }
    pivot = (
        df.pivot(index="day_of_week", columns="club_member_status", values="segment_day_share_pct")
        .fillna(0)
        .sort_index()
    )
    pivot.index = [weekday_names.get(int(idx), str(idx)) for idx in pivot.index]
    plt.figure(figsize=(10, 6), dpi=300)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Within-Segment Purchase Share by Weekday: ACTIVE vs PRE-CREATE")
    plt.xlabel("Club Member Status")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _run_query_pipeline(
    query_name: str,
    query_df: DataFrame,
    csv_output_dir: str,
    explain_log_file: str,
    plot_output_file: str,
    plot_func: Callable[[pd.DataFrame, str], None],
) -> None:
    _append_explain_log(query_df, query_name, explain_log_file)
    _save_csv(query_df, csv_output_dir)
    plot_df = query_df
    for field in query_df.schema.fields:
        if isinstance(field.dataType, BooleanType):
            plot_df = plot_df.withColumn(field.name, F.col(field.name).cast("int"))
    plot_func(plot_df.toPandas(), plot_output_file)


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_root: str = "output/dmytro",
) -> None:
    """Execute all 6 Dmytro transformation queries and write outputs."""
    sns.set_theme(style="whitegrid", palette="muted")

    csv_root = os.path.join(output_root, "csv")
    plots_root = os.path.join(output_root, "plots")
    logs_root = os.path.join(output_root, "logs")
    os.makedirs(csv_root, exist_ok=True)
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    explain_log_file = os.path.join(logs_root, "explain_logs.txt")
    with open(explain_log_file, "w", encoding="utf-8") as fh:
        fh.write("Dmytro Transformation Query Explain Logs\n")

    transactions = spark.read.parquet(os.path.join(processed_dir, "transactions"))
    articles = spark.read.parquet(os.path.join(processed_dir, "articles"))
    customers = spark.read.parquet(os.path.join(processed_dir, "customers"))
    transactions = transactions.repartition("customer_id").cache()
    transactions.count()

    query_outputs: list[tuple[str, DataFrame, Callable[[pd.DataFrame, str], None]]] = [
        (
            "Query 1 - Weekly transaction dynamics",
            query_1_weekly_transaction_dynamics(transactions),
            _plot_query_1,
        ),
        (
            "Query 2 - Winter vs summer spike by product group",
            query_2_winter_vs_summer_spike(transactions, articles),
            _plot_query_2,
        ),
        (
            "Query 3 - Retention gap between first and second purchase",
            query_3_retention_gap_distribution(transactions),
            _plot_query_3,
        ),
        (
            "Query 4 - Top peak days and monthly share",
            query_4_top_peak_days_month_share(transactions),
            _plot_query_4,
        ),
        (
            "Query 5 - 7-day rolling activity for age 30-40",
            query_5_rolling_activity_30_40(transactions, customers),
            _plot_query_5,
        ),
        (
            "Query 6 - Frequency ACTIVE vs PRE-CREATE",
            query_6_frequency_loyal_vs_guest(transactions, customers),
            _plot_query_6,
        ),
    ]

    for idx, (title, query_df, plot_func) in enumerate(query_outputs, start=1):
        _run_query_pipeline(
            query_name=title,
            query_df=query_df,
            csv_output_dir=os.path.join(csv_root, f"query_{idx}"),
            explain_log_file=explain_log_file,
            plot_output_file=os.path.join(plots_root, f"query_{idx}.png"),
            plot_func=plot_func,
        )

    transactions.unpersist()


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Dmytro-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
