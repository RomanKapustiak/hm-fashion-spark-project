"""Yura visual trends and design features transformation pipeline.

Run inside Docker with:
python src/transformations/yura_pipeline.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import create_spark_session


MEMBER_NAME = "yura"
MAX_PANDAS_ROWS = 5000
MIN_PARTITIONS = 8
CHANNEL_LABELS = {
    1: "Store",
    2: "Online",
}


sns.set_theme(style="whitegrid", palette="muted")


def _build_channel_label_expression() -> F.Column:
    column = F.when(F.col("sales_channel_id") == F.lit(1), F.lit(CHANNEL_LABELS[1]))
    column = column.when(F.col("sales_channel_id") == F.lit(2), F.lit(CHANNEL_LABELS[2]))
    return column.otherwise(F.concat(F.lit("Channel "), F.col("sales_channel_id").cast("string")))


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
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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


def _plot_q1_colour_value_distribution(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Assortment Distribution by Perceived Colour Value"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.copy()
    plot_df["_category_norm"] = plot_df["perceived_colour_value_name"].astype(str).str.strip().str.lower()

    # Keep informative categories only; do not plot placeholder classes with zero observations.
    plot_df = plot_df[
        ~(
            plot_df["_category_norm"].isin(["unknown", "undefined"])
            & (plot_df["article_count"] <= 0)
        )
    ]
    plot_df = plot_df[plot_df["article_count"] > 0]

    if plot_df.empty:
        _save_no_data_plot(
            plot_path,
            title,
            "No non-zero categories available after filtering placeholders.",
        )
        return

    plot_df = plot_df.drop(columns=["_category_norm"]).sort_values("article_count", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=plot_df,
        x="perceived_colour_value_name",
        y="article_count",
        palette="crest",
        ax=ax,
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Perceived Colour Value")
    ax.set_ylabel("Unique Articles")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    _finalize_figure(fig, plot_path)


def _plot_q2_top_stripe_colours(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Top 10 Purchased Colours for Stripe Pattern Items"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.sort_values("purchase_count", ascending=False).reset_index(drop=True)
    plot_df["rank"] = plot_df.index + 1

    max_count = float(plot_df["purchase_count"].max())
    bubble_sizes = ((plot_df["purchase_count"] / max_count) ** 0.5) * 2600 + 600
    inside_label_threshold = max_count * 0.22

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(
        plot_df["rank"],
        plot_df["purchase_count"],
        s=bubble_sizes,
        c=range(len(plot_df)),
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.8,
    )

    for _, row in plot_df.iterrows():
        x_value = row["rank"]
        y_value = row["purchase_count"]
        label = str(row["colour_group_name"]).title()

        if y_value >= inside_label_threshold:
            ax.text(
                x_value,
                y_value,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                weight="bold",
            )
        else:
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1f2933",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.9, "ec": "none"},
                arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.8},
            )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Colour Rank")
    ax.set_ylabel("Purchase Count")
    ax.set_xticks(plot_df["rank"])
    ax.set_xticklabels([f"#{int(rank)}" for rank in plot_df["rank"]])
    ax.set_ylim(0, max_count * 1.20)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    _finalize_figure(fig, plot_path)


def _plot_q3_black_rolling_avg(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "30-Day Rolling Average of Black Item Sales"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.sort_values("sale_date").copy()
    plot_df["sale_date"] = pd.to_datetime(plot_df["sale_date"])

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=plot_df, x="sale_date", y="daily_sales", color="#B0BEC5", linewidth=1.5, ax=ax, label="Daily Sales")
    sns.lineplot(
        data=plot_df,
        x="sale_date",
        y="rolling_30d_avg",
        color="#1D3557",
        linewidth=2.8,
        ax=ax,
        label="Rolling 30-Day Average",
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Purchase Count")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Metric")

    _finalize_figure(fig, plot_path)


def _plot_q4_top_pattern_by_channel(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Most Popular Graphical Appearance by Sales Channel"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.sort_values("sales_channel_id")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="channel_name",
        y="purchase_count",
        hue="graphical_appearance_name",
        dodge=False,
        palette="Set2",
        ax=ax,
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Sales Channel")
    ax.set_ylabel("Purchase Count")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Top Pattern")

    _finalize_figure(fig, plot_path)


def _plot_q5_floral_avg_check(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Average Check for Floral Items by Sales Channel"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.sort_values("sales_channel_id")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="channel_name", y="avg_price", palette="mako", ax=ax)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Sales Channel")
    ax.set_ylabel("Average Price")

    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        ax.annotate(
            f"{row['avg_price']:.4f}\n(n={int(row['transaction_count']):,})",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 6),
            textcoords="offset points",
        )

    _finalize_figure(fig, plot_path)


def _plot_q6_red_mom_waterfall(pdf: pd.DataFrame, plot_path: str) -> None:
    title = "Month-over-Month Revenue Changes for Red Master Colour"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    plot_df = pdf.sort_values("month_start").copy()
    plot_df["month_label"] = pd.to_datetime(plot_df["month_start"]).dt.strftime("%Y-%m")

    plot_df["bar_base"] = plot_df["previous_month_revenue"].fillna(0.0)
    plot_df["bar_height"] = plot_df["revenue_delta"].fillna(plot_df["monthly_revenue"])
    plot_df["bar_color"] = plot_df["bar_height"].apply(lambda value: "#2A9D8F" if value >= 0 else "#D1495B")

    if not plot_df.empty:
        first_index = plot_df.index[0]
        plot_df.loc[first_index, "bar_color"] = "#4C78A8"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        plot_df["month_label"],
        plot_df["bar_height"],
        bottom=plot_df["bar_base"],
        color=plot_df["bar_color"],
        width=0.7,
    )

    ax.plot(
        plot_df["month_label"],
        plot_df["monthly_revenue"],
        color="#1F2933",
        linewidth=2,
        marker="o",
        label="Monthly Revenue",
    )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend()

    _finalize_figure(fig, plot_path)


def _question_1(articles_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    final_df = (
        articles_df
        .repartition(partition_count, "perceived_colour_value_name")
        .withColumn(
            "perceived_colour_value_name",
            F.coalesce(F.col("perceived_colour_value_name"), F.lit("Unknown")),
        )
        .groupBy("perceived_colour_value_name")
        .agg(F.countDistinct("article_id").alias("article_count"))
        .orderBy(F.desc("article_count"), F.asc("perceived_colour_value_name"))
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 1: Assortment distribution by perceived colour value",
        output_name="q1_assortment_by_colour_value",
        directories=directories,
        plotter=_plot_q1_colour_value_distribution,
    )


def _question_2(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    stripe_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("graphical_appearance_name"))) == F.lit("stripe"))
        .select("article_id", "colour_group_name")
        .withColumn("colour_group_name", F.coalesce(F.col("colour_group_name"), F.lit("Unknown")))
        .dropDuplicates(["article_id", "colour_group_name"])
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .join(F.broadcast(stripe_articles), on="article_id", how="inner")
        .groupBy("colour_group_name")
        .agg(F.count(F.lit(1)).alias("purchase_count"))
        .orderBy(F.desc("purchase_count"), F.asc("colour_group_name"))
        .limit(10)
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 2: Top 10 purchased colours for stripe pattern items",
        output_name="q2_top10_stripe_colours",
        directories=directories,
        plotter=_plot_q2_top_stripe_colours,
    )


def _question_3(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    black_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("colour_group_name"))) == F.lit("black"))
        .select("article_id")
        .dropDuplicates(["article_id"])
    )

    rolling_window = Window.orderBy("sale_date").rowsBetween(-29, 0)

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .join(F.broadcast(black_articles), on="article_id", how="inner")
        .withColumn("sale_date", F.to_date("t_dat"))
        .groupBy("sale_date")
        .agg(F.count(F.lit(1)).alias("daily_sales"))
        .withColumn("rolling_30d_avg", F.avg("daily_sales").over(rolling_window))
        .orderBy("sale_date")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 3: Rolling 30-day average sales for black items",
        output_name="q3_black_sales_rolling_30d",
        directories=directories,
        plotter=_plot_q3_black_rolling_avg,
    )


def _question_4(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    ranked_window = Window.partitionBy("sales_channel_id").orderBy(
        F.desc("purchase_count"),
        F.asc("graphical_appearance_name"),
    )

    patterns_df = articles_df.select("article_id", "graphical_appearance_name").withColumn(
        "graphical_appearance_name",
        F.coalesce(F.col("graphical_appearance_name"), F.lit("Unknown")),
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .join(F.broadcast(patterns_df), on="article_id", how="inner")
        .groupBy("sales_channel_id", "graphical_appearance_name")
        .agg(F.count(F.lit(1)).alias("purchase_count"))
        .withColumn("pattern_rank", F.row_number().over(ranked_window))
        .filter(F.col("pattern_rank") == 1)
        .withColumn("channel_name", _build_channel_label_expression())
        .select("sales_channel_id", "channel_name", "graphical_appearance_name", "purchase_count", "pattern_rank")
        .orderBy("sales_channel_id")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 4: Most popular graphical appearance by sales channel",
        output_name="q4_top_pattern_by_channel",
        directories=directories,
        plotter=_plot_q4_top_pattern_by_channel,
    )


def _question_5(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    graphical_norm = F.lower(F.coalesce(F.col("graphical_appearance_name"), F.lit("")))
    prod_name_norm = F.lower(F.coalesce(F.col("prod_name"), F.lit("")))
    detail_desc_norm = F.lower(F.coalesce(F.col("detail_desc"), F.lit("")))

    floral_mask = (
        graphical_norm.contains("floral")
        | graphical_norm.contains("flower")
        | prod_name_norm.contains("floral")
        | prod_name_norm.contains("flower")
        | detail_desc_norm.contains("floral")
        | detail_desc_norm.contains("flower")
    )

    floral_articles = (
        articles_df
        .filter(floral_mask)
        .select("article_id")
        .dropDuplicates(["article_id"])
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .join(F.broadcast(floral_articles), on="article_id", how="inner")
        .groupBy("sales_channel_id")
        .agg(
            F.avg("price").alias("avg_price"),
            F.count(F.lit(1)).alias("transaction_count"),
        )
        .withColumn("channel_name", _build_channel_label_expression())
        .select("sales_channel_id", "channel_name", "avg_price", "transaction_count")
        .orderBy("sales_channel_id")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 5: Average check for floral items by sales channel",
        output_name="q5_floral_avg_check_by_channel",
        directories=directories,
        plotter=_plot_q5_floral_avg_check,
    )


def _question_6(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    red_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("perceived_colour_master_name"))) == F.lit("red"))
        .select("article_id")
        .dropDuplicates(["article_id"])
    )

    month_window = Window.orderBy("month_start")

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .join(F.broadcast(red_articles), on="article_id", how="inner")
        .withColumn("month_start", F.trunc("t_dat", "month"))
        .groupBy("month_start")
        .agg(F.sum("price").alias("monthly_revenue"))
        .withColumn("previous_month_revenue", F.lag("monthly_revenue").over(month_window))
        .withColumn(
            "revenue_delta",
            F.col("monthly_revenue") - F.col("previous_month_revenue"),
        )
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
        question_label="Question 6: Month-over-month growth for red master-colour item revenue",
        output_name="q6_red_mom_growth",
        directories=directories,
        plotter=_plot_q6_red_mom_waterfall,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    directories = _ensure_output_dirs(output_dir)
    explain_log_path = os.path.join(directories["logs"], "explain_logs.txt")

    with open(explain_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Yura pipeline explain plans\n")

    transactions_df = _load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = _load_parquet_dataset(spark, processed_dir, "articles")
    partition_count = _recommended_partitions(spark)

    _question_1(articles_df, directories, partition_count)
    _question_2(transactions_df, articles_df, directories, partition_count)
    _question_3(transactions_df, articles_df, directories, partition_count)
    _question_4(transactions_df, articles_df, directories, partition_count)
    _question_5(transactions_df, articles_df, directories, partition_count)
    _question_6(transactions_df, articles_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Yura-Pipeline")

    try:
        run(spark_session)
    finally:
        spark_session.stop()
