"""Roman financial performance transformation pipeline.

Run inside Docker with:
python src/transformations/roman_pipeline.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import create_spark_session


MEMBER_NAME = "roman"
MAX_PANDAS_ROWS = 5000
MIN_PARTITIONS = 8
CHANNEL_LABELS = {
    1: "Store",
    2: "Online",
}
WEEKDAY_LABELS = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}


sns.set_theme(style="whitegrid", palette="muted")


def _build_channel_label_expression() -> F.Column:
    column = F.when(F.col("sales_channel_id") == F.lit(1), F.lit(CHANNEL_LABELS[1]))
    column = column.when(F.col("sales_channel_id") == F.lit(2), F.lit(CHANNEL_LABELS[2]))
    return column.otherwise(F.concat(F.lit("Channel "), F.col("sales_channel_id").cast("string")))


def _build_weekday_label_expression() -> F.Column:
    column = None
    for weekday_number, weekday_name in WEEKDAY_LABELS.items():
        if column is None:
            column = F.when(F.col("weekday_num") == F.lit(weekday_number), F.lit(weekday_name))
        else:
            column = column.when(F.col("weekday_num") == F.lit(weekday_number), F.lit(weekday_name))

    return column.otherwise(F.lit("Unknown"))


def _ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    csv_dir = os.path.join(output_dir, "csv")
    plots_dir = os.path.join(output_dir, "plots")
    logs_dir = os.path.join(output_dir, "logs")

    for directory in (output_dir, csv_dir, plots_dir, logs_dir):
        os.makedirs(directory, exist_ok=True)

    return {
        "base": output_dir,
        "csv": csv_dir,
        "plots": plots_dir,
        "logs": logs_dir,
    }


def _recommended_partitions(spark: SparkSession) -> int:
    return max(spark.sparkContext.defaultParallelism, MIN_PARTITIONS)


def _load_parquet_dataset(spark: SparkSession, processed_dir: str, dataset_name: str) -> DataFrame:
    path = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return spark.read.parquet(path)


def _write_explain_log(df: DataFrame, log_path: str, question_label: str) -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(extended=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 100}\n")
        log_file.write(f"{question_label}\n")
        log_file.write(f"{'=' * 100}\n")
        log_file.write(buffer.getvalue().rstrip())
        log_file.write("\n")


def _save_csv(df: DataFrame, csv_dir: str, output_name: str) -> None:
    output_path = os.path.join(csv_dir, output_name)
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(output_path)
    )


def _to_small_pandas(df: DataFrame, output_name: str, max_rows: int = MAX_PANDAS_ROWS) -> pd.DataFrame:
    row_count = df.count()
    if row_count > max_rows:
        raise ValueError(
            f"Refusing to convert '{output_name}' to pandas because it has {row_count} rows "
            f"(limit: {max_rows})."
        )
    return df.toPandas()


def _finalize_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_no_data_plot(path: str, title: str, message: str = "No data available for this question.") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=16, color="#4c566a")
    ax.set_title(title, fontsize=16, weight="bold")
    _finalize_figure(fig, path)


def _execute_output_bundle(
    df: DataFrame,
    question_label: str,
    output_name: str,
    directories: Dict[str, str],
    plotter,
) -> None:
    _write_explain_log(df, os.path.join(directories["logs"], "explain_logs.txt"), question_label)
    _save_csv(df, directories["csv"], output_name)
    pandas_df = _to_small_pandas(df, output_name)
    plot_path = os.path.join(directories["plots"], f"{output_name}.png")
    plotter(pandas_df, plot_path)


def _plot_q1_revenue_share(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Revenue Share by Sales Channel Across Product Index Groups"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.copy()
    plot_df["channel_name"] = plot_df["channel_name"].fillna("Unknown")
    plot_df["index_group_name"] = plot_df["index_group_name"].fillna("Unknown").str.title()

    pivot_df = (
        plot_df.pivot_table(
            index="index_group_name",
            columns="channel_name",
            values="total_revenue",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=["Store", "Online"], fill_value=0.0)
    )

    shares = pivot_df.div(pivot_df.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
    shares = shares.sort_values(by=["Online", "Store"], ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#6C8EBF", "#E07A5F"]
    shares.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.8,
        color=colors[: len(shares.columns)],
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Index Group Name")
    ax.set_ylabel("Revenue Share")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(title="Sales Channel")
    ax.tick_params(axis="x", rotation=30)
    _finalize_figure(fig, plot_path)


def _plot_q2_weekday_radar(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Average Revenue by Day of Week"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    ordered = pdf.sort_values("weekday_num").reset_index(drop=True)
    labels = ordered["weekday_name"].tolist()
    values = ordered["avg_revenue"].astype(float).tolist()

    angles = [index / float(len(labels)) * 2 * math.pi for index in range(len(labels))]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.plot(angles, values, color="#5B8E7D", linewidth=2.5)
    ax.fill(angles, values, color="#5B8E7D", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    ax.set_rlabel_position(0)
    _finalize_figure(fig, plot_path)


def _plot_q3_donut(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Revenue Share of Top 10 Most Popular Products vs Rest in 2019"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.copy()
    plot_df["assortment_segment"] = plot_df["assortment_segment"].fillna("Unknown")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#4C78A8", "#F58518"]
    wedges, texts, autotexts = ax.pie(
        plot_df["segment_revenue"],
        labels=plot_df["assortment_segment"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(plot_df)],
        pctdistance=0.8,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    centre_circle = Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(11)
    ax.set_title(title, fontsize=16, weight="bold")
    _finalize_figure(fig, plot_path)


def _plot_q4_growth_rate(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Month-over-Month Revenue Growth Rate"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.dropna(subset=["growth_rate_pct"]).copy()
    if plot_df.empty:
        _save_no_data_plot(plot_path, title, "Growth rate cannot be calculated because no prior month exists.")
        return

    plot_df["month_label"] = pd.to_datetime(plot_df["month_start"]).dt.strftime("%Y-%m")
    colors = plot_df["growth_rate_pct"].apply(lambda value: "#2A9D8F" if value >= 0 else "#D1495B")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(plot_df["month_label"], plot_df["growth_rate_pct"], color=colors)
    ax.axhline(0, color="#1f2933", linewidth=1)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Growth Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    _finalize_figure(fig, plot_path)


def _plot_q5_kpi_card(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Average Check for Customers Receiving Fashion News Regularly"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    row = pdf.iloc[0]
    avg_check_value = float(row["avg_check_value"]) if pd.notna(row["avg_check_value"]) else 0.0
    transaction_count = int(row["transaction_count"]) if pd.notna(row["transaction_count"]) else 0
    customer_count = int(row["customer_count"]) if pd.notna(row["customer_count"]) else 0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#F7F7F7")
    ax.text(0.5, 0.82, title, ha="center", va="center", fontsize=18, weight="bold", color="#264653")
    ax.text(0.5, 0.48, f"{avg_check_value:.4f}", ha="center", va="center", fontsize=34, weight="bold", color="#2A9D8F")
    ax.text(0.5, 0.29, "Average Transaction Price", ha="center", va="center", fontsize=14, color="#5C677D")
    ax.text(
        0.5,
        0.12,
        f"Based on {transaction_count:,} transactions across {customer_count:,} customers",
        ha="center",
        va="center",
        fontsize=12,
        color="#5C677D",
    )
    _finalize_figure(fig, plot_path)


def _plot_q6_running_total(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Running Total Revenue in 2019 by Sales Channel"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.copy()
    plot_df["t_dat"] = pd.to_datetime(plot_df["t_dat"])

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        data=plot_df,
        x="t_dat",
        y="running_total_revenue",
        hue="channel_name",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Running Total Revenue")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Sales Channel")
    _finalize_figure(fig, plot_path)


def _question_1(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .alias("transactions")
        .join(F.broadcast(articles_df).alias("articles"), on="article_id", how="inner")
        .withColumn("index_group_name", F.coalesce(F.col("index_group_name"), F.lit("unknown")))
        .groupBy("index_group_name", "sales_channel_id")
        .agg(F.sum("price").alias("total_revenue"))
        .withColumn("channel_name", _build_channel_label_expression())
        .select("index_group_name", "sales_channel_id", "channel_name", "total_revenue")
        .orderBy("index_group_name", "sales_channel_id")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 1: Revenue by sales channel and index group",
        output_name="q1_revenue_by_index_group_channel",
        directories=directories,
        plotter=_plot_q1_revenue_share,
    )


def _question_2(transactions_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    ranking_window = Window.orderBy(F.desc("avg_revenue"), F.asc("weekday_num"))

    final_df = (
        transactions_df
        .repartition(partition_count, "t_dat")
        .filter(F.col("price") > F.lit(0.01))
        .withColumn("weekday_num", F.dayofweek("t_dat"))
        .withColumn("weekday_name", _build_weekday_label_expression())
        .groupBy("weekday_num", "weekday_name")
        .agg(F.avg("price").alias("avg_revenue"))
        .withColumn("revenue_rank", F.dense_rank().over(ranking_window))
        .orderBy("revenue_rank", "weekday_num")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 2: Average revenue ranking by day of week",
        output_name="q2_weekday_avg_revenue_rank",
        directories=directories,
        plotter=_plot_q2_weekday_radar,
    )


def _question_3(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    product_metrics_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .alias("transactions")
        .filter(F.year("t_dat") == F.lit(2019))
        .join(F.broadcast(articles_df).alias("articles"), on="article_id", how="inner")
        .withColumn("prod_name", F.coalesce(F.col("prod_name"), F.lit("unknown product")))
        .groupBy("prod_name")
        .agg(
            F.count(F.lit(1)).alias("transaction_count"),
            F.sum("price").alias("product_revenue"),
        )
    )

    top_products_df = (
        product_metrics_df
        .orderBy(F.desc("transaction_count"), F.desc("product_revenue"), F.asc("prod_name"))
        .limit(10)
        .select("prod_name")
        .withColumn("is_top_10", F.lit(1))
    )

    final_df = (
        product_metrics_df
        .join(F.broadcast(top_products_df), on="prod_name", how="left")
        .withColumn(
            "assortment_segment",
            F.when(F.col("is_top_10") == F.lit(1), F.lit("Top 10 products"))
            .otherwise(F.lit("Rest of assortment")),
        )
        .groupBy("assortment_segment")
        .agg(
            F.sum("product_revenue").alias("segment_revenue"),
            F.sum("transaction_count").alias("segment_transactions"),
        )
        .orderBy(F.desc("segment_revenue"))
    )

    total_revenue = final_df.agg(F.sum("segment_revenue").alias("total_revenue")).collect()[0]["total_revenue"] or 0.0
    final_df = final_df.withColumn(
        "revenue_share_pct",
        F.when(F.lit(total_revenue) != 0, (F.col("segment_revenue") / F.lit(total_revenue)) * F.lit(100.0))
        .otherwise(F.lit(0.0)),
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 3: Revenue share of top 10 products versus the rest in 2019",
        output_name="q3_top10_revenue_share_2019",
        directories=directories,
        plotter=_plot_q3_donut,
    )


def _question_4(transactions_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    month_window = Window.orderBy("month_start")

    final_df = (
        transactions_df
        .repartition(partition_count, "t_dat")
        .withColumn("month_start", F.trunc("t_dat", "month"))
        .groupBy("month_start")
        .agg(F.sum("price").alias("monthly_revenue"))
        .withColumn("previous_month_revenue", F.lag("monthly_revenue").over(month_window))
        .withColumn(
            "growth_rate_pct",
            F.when(
                (F.col("previous_month_revenue").isNotNull()) & (F.col("previous_month_revenue") != 0),
                ((F.col("monthly_revenue") - F.col("previous_month_revenue")) / F.col("previous_month_revenue")) * 100.0,
            ),
        )
        .orderBy("month_start")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 4: Month-over-month revenue growth rate",
        output_name="q4_monthly_growth_rate",
        directories=directories,
        plotter=_plot_q4_growth_rate,
    )


def _question_5(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    final_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .alias("transactions")
        .join(F.broadcast(customers_df).alias("customers"), on="customer_id", how="inner")
        .filter(F.lower(F.trim(F.col("fashion_news_frequency"))) == F.lit("regularly"))
        .agg(
            F.avg("price").alias("avg_check_value"),
            F.count(F.lit(1)).alias("transaction_count"),
            F.countDistinct("customer_id").alias("customer_count"),
        )
        .withColumn("segment_name", F.lit("Regular Fashion News Subscribers"))
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 5: Average check for customers receiving fashion news regularly",
        output_name="q5_regular_fashion_news_avg_check",
        directories=directories,
        plotter=_plot_q5_kpi_card,
    )


def _question_6(transactions_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    running_window = (
        Window.partitionBy("sales_channel_id")
        .orderBy("t_dat")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "t_dat")
        .filter(F.year("t_dat") == F.lit(2019))
        .groupBy("t_dat", "sales_channel_id")
        .agg(F.sum("price").alias("daily_revenue"))
        .withColumn("channel_name", _build_channel_label_expression())
        .withColumn("running_total_revenue", F.sum("daily_revenue").over(running_window))
        .select("t_dat", "sales_channel_id", "channel_name", "daily_revenue", "running_total_revenue")
        .orderBy("t_dat", "sales_channel_id")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 6: Running total revenue in 2019 by sales channel",
        output_name="q6_running_total_2019_by_channel",
        directories=directories,
        plotter=_plot_q6_running_total,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    directories = _ensure_output_dirs(output_dir)
    explain_log_path = os.path.join(directories["logs"], "explain_logs.txt")

    with open(explain_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Roman pipeline explain plans\n")

    transactions_df = _load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = _load_parquet_dataset(spark, processed_dir, "articles")
    customers_df = _load_parquet_dataset(spark, processed_dir, "customers")
    partition_count = _recommended_partitions(spark)

    _question_1(transactions_df, articles_df, directories, partition_count)
    _question_2(transactions_df, directories, partition_count)
    _question_3(transactions_df, articles_df, directories, partition_count)
    _question_4(transactions_df, directories, partition_count)
    _question_5(transactions_df, customers_df, directories, partition_count)
    _question_6(transactions_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Roman-Pipeline")

    try:
        run(spark_session)
    finally:
        spark_session.stop()
