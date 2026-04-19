"""Artem customer demographics and behavior transformation pipeline.

Run inside Docker with:
python src/transformations/artem_pipeline.py
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

MEMBER_NAME = "artem"
MAX_PANDAS_ROWS = 5000
MIN_PARTITIONS = 8
CHANNEL_LABELS = {
    1: "Store",
    2: "Online",
}

sns.set_theme(style="whitegrid", palette="muted")


def _build_channel_label_expression() -> F.Column:
    """Build a PySpark column expression for formatting sales channel IDs as human-readable labels."""
    column = F.when(F.col("sales_channel_id") == F.lit(1), F.lit(CHANNEL_LABELS[1]))
    column = column.when(F.col("sales_channel_id") == F.lit(2), F.lit(CHANNEL_LABELS[2]))
    return column.otherwise(F.concat(F.lit("Channel "), F.col("sales_channel_id").cast("string")))


def _ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    """Ensure the required output directories exist (csv, plots, logs) and return their specific paths."""
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
    """Calculate the recommended minimum partitions based on your cluster properties and preset requirements."""
    return max(spark.sparkContext.defaultParallelism, MIN_PARTITIONS)


def _load_parquet_dataset(spark: SparkSession, processed_dir: str, dataset_name: str) -> DataFrame:
    """Load a specific pre-processed Parquet dataset from the central processed outputs directory."""
    path = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return spark.read.parquet(path)


def _write_explain_log(df: DataFrame, log_path: str, question_label: str) -> None:
    """Execute the PySpark extended explain command on the dataframe and append it to the log file."""
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
    """Coalesce the PySpark dataframe into a single file and securely overwrite as a local CSV format."""
    output_path = os.path.join(csv_dir, output_name)
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(output_path)
    )


def _to_small_pandas(df: DataFrame, output_name: str, max_rows: int = MAX_PANDAS_ROWS) -> pd.DataFrame:
    """Convert an aggregated PySpark dataframe to a Pandas layout if it strictly obeys sizing limits."""
    row_count = df.count()
    if row_count > max_rows:
        raise ValueError(
            f"Refusing to convert '{output_name}' to pandas because it has {row_count} rows "
            f"(limit: {max_rows})."
        )
    return df.toPandas()


def _finalize_figure(fig: plt.Figure, path: str) -> None:
    """Wrap up visualization aesthetics, strictly enforce high DPI margins, and save the PNG file locally."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_no_data_plot(path: str, title: str, message: str = "No data available for this question.") -> None:
    """Produce a generic clean placeholder image for scenarios where a pipeline yields heavily filtered empty sets."""
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
    """Wrap standard end-of-pipeline methods handling logging, saving, pandas conversion, and plot triggers collectively."""
    _write_explain_log(df, os.path.join(directories["logs"], "explain_logs.txt"), question_label)
    _save_csv(df, directories["csv"], output_name)
    pandas_df = _to_small_pandas(df, output_name)
    plot_path = os.path.join(directories["plots"], f"{output_name}.png")
    plotter(pandas_df, plot_path)


def _plot_q1_age_distribution(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q1: Bar chart — number of customers per 5-year age group."""
    title = "Age Distribution of Active Customers"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=pdf, x="age_group", y="customer_count", color="#5B8E7D", ax=ax)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    _finalize_figure(fig, plot_path)


def _plot_q2_purchase_frequency(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q2: Bar chart — average purchases per customer for 'Regularly' vs 'NONE' news subscribers."""
    title = "Average Purchase Count by Fashion News Subscription"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pdf, x="fashion_news_frequency", y="avg_purchases_per_customer", palette="muted", ax=ax)
    
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Fashion News Status")
    ax.set_ylabel("Average Purchases per Customer")
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', 
                    fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    _finalize_figure(fig, plot_path)


def _plot_q3_top_postal_codes(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q3: Horizontal bar chart — top 15 postal codes ranked by active member count."""
    title = "Top 15 Regions (Postal Codes) by Active Members"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=pdf, x="active_member_count", y="postal_code", palette="viridis_r", ax=ax)
    
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Active Member Count (log scale)")
    ax.set_ylabel("Postal Code")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    _finalize_figure(fig, plot_path)


def _plot_q4_spend_quartiles(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q4: Scatter + line chart — average customer age across 4 spending quartiles (Q1 = highest spenders)."""
    title = "Average Customer Age by Spending Quartiles"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    pdf["spend_quartile_label"] = pdf["spend_quartile"].apply(lambda q: f"Q{q} (Rank {q})")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pdf, x="spend_quartile_label", y="average_age", s=300, color="#E07A5F", ax=ax, zorder=5)
    sns.lineplot(data=pdf, x="spend_quartile_label", y="average_age", color="#E07A5F", ax=ax, linewidth=2, linestyle="--", zorder=4)
    
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Spending Quartile (Q1 = Highest Spenders)")
    ax.set_ylabel("Average Age")
    
    _finalize_figure(fig, plot_path)


def _plot_q5_top_youth_departments(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q5: Stacked bar chart — purchase counts for top 5 departments among under-25 customers, split by sales channel."""
    title = "Top 5 Departments among Youth (<25 Years Old) by Channel"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    pivot_df = (
        pdf.pivot_table(
            index="department_name",
            columns="channel_name",
            values="purchase_count",
            aggfunc="sum",
            fill_value=0.0,
        )
    )
    
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values("Total", ascending=False).drop("Total", axis=1)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#4C78A8", "#F58518"]
    pivot_df.plot(kind="bar", stacked=True, ax=ax, color=colors[:len(pivot_df.columns)], width=0.7)
    
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Department Name")
    ax.set_ylabel("Total Purchase Count")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Sales Channel")
    
    _finalize_figure(fig, plot_path)


def _plot_q6_cumulative_new_customers(pdf: pd.DataFrame, plot_path: str) -> None:
    """Q6: Area chart — cumulative total of first-time customers acquired per month over the full dataset period."""
    title = "Cumulative Customer Acquisition Over Time"
    if pdf.empty:
        _save_no_data_plot(plot_path, title)
        return

    pdf["month_label"] = pd.to_datetime(pdf["month_start"]).dt.strftime("%Y-%m")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(pdf["month_label"], pdf["cumulative_new_customers"], color="#6C8EBF", alpha=0.5)
    ax.plot(pdf["month_label"], pdf["cumulative_new_customers"], color="#4C78A8", linewidth=2.5)

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Total Customers")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis="x", rotation=45)
    
    _finalize_figure(fig, plot_path)


def _question_1(customers_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    """Q1: What is the age distribution of active customers in 5-year buckets? 
    Operations: filter, group by. Plot: bar chart."""
    final_df = (
        customers_df
        .repartition(partition_count, "customer_id")
        .filter(F.col("age").isNotNull() & (F.lower(F.col("club_member_status")) == F.lit("active")))
        .withColumn("age_bucket", (F.floor(F.col("age") / 5) * 5).cast("int"))
        .withColumn(
            "age_group", 
            F.concat(F.col("age_bucket").cast("string"), F.lit("-"), (F.col("age_bucket") + 4).cast("string"))
        )
        .groupBy("age_group")
        .agg(F.count(F.lit(1)).alias("customer_count"))
        .orderBy("age_group")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 1: Active customer age distribution (5-year groups)",
        output_name="q1_age_distribution",
        directories=directories,
        plotter=_plot_q1_age_distribution,
    )


def _question_2(
    transactions_df: DataFrame, 
    customers_df: DataFrame, 
    directories: Dict[str, str], 
    partition_count: int
) -> None:
    """Q2: How does avg purchase count differ between 'Regularly' and 'NONE' news subscribers? 
    Operations: filter, join, group by. Plot: bar chart."""
    filtered_customers = (
        customers_df
        .withColumn("fashion_news_frequency", F.upper(F.trim(F.col("fashion_news_frequency"))))
        .filter(F.col("fashion_news_frequency").isin("REGULARLY", "NONE"))
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .join(F.broadcast(filtered_customers), on="customer_id", how="inner")
        .groupBy("customer_id", "fashion_news_frequency")
        .agg(F.count(F.lit(1)).alias("purchase_count"))
        .groupBy("fashion_news_frequency")
        .agg(F.avg("purchase_count").alias("avg_purchases_per_customer"))
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 2: Average purchase frequency by fashion news status",
        output_name="q2_purchase_frequency_by_news",
        directories=directories,
        plotter=_plot_q2_purchase_frequency,
    )


def _question_3(customers_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    """Q3: Which 15 postal codes have the most active club members? 
    Operations: filter, group by, window (row_number). Plot: horizontal bar chart."""
    rank_window = Window.orderBy(F.desc("active_member_count"))

    final_df = (
        customers_df
        .repartition(partition_count, "postal_code")
        .filter(F.lower(F.col("club_member_status")) == F.lit("active"))
        .groupBy("postal_code")
        .agg(F.count(F.lit(1)).alias("active_member_count"))
        .withColumn("postal_rank", F.row_number().over(rank_window))
        .filter(F.col("postal_rank") <= 15)
        .orderBy("postal_rank")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 3: Top 15 postal codes by active members",
        output_name="q3_top15_postal_codes",
        directories=directories,
        plotter=_plot_q3_top_postal_codes,
    )


def _question_4(
    transactions_df: DataFrame, 
    customers_df: DataFrame, 
    directories: Dict[str, str], 
    partition_count: int
) -> None:
    """Q4: What is the average age of customers in each of the 4 spending quartiles? 
    Operations: join, group by, window (ntile). Plot: scatter + line chart."""
    spend_window = Window.orderBy(F.desc("total_spent"))

    customer_spends = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .groupBy("customer_id")
        .agg(F.sum("price").alias("total_spent"))
    )
    
    final_df = (
        customer_spends
        .join(F.broadcast(customers_df.filter(F.col("age").isNotNull())), on="customer_id", how="inner")
        .withColumn("spend_quartile", F.ntile(4).over(spend_window))
        .groupBy("spend_quartile")
        .agg(
            F.avg("age").alias("average_age"), 
            F.count(F.lit(1)).alias("customer_count")
        )
        .orderBy("spend_quartile")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 4: Quartiles by overall spend and average age",
        output_name="q4_spending_quartiles_age",
        directories=directories,
        plotter=_plot_q4_spend_quartiles,
    )


def _question_5(
    transactions_df: DataFrame, 
    customers_df: DataFrame, 
    articles_df: DataFrame, 
    directories: Dict[str, str], 
    partition_count: int
) -> None:
    """Q5: Which 5 departments are most popular among customers under 25? 
    Operations: filter, join x3, group by. Plot: stacked bar chart by channel."""
    youth_customers = (
        customers_df
        .filter(F.col("age") < 25)
        .select("customer_id")
    )

    youth_transactions = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .join(F.broadcast(youth_customers), on="customer_id", how="inner")
        .join(F.broadcast(articles_df), on="article_id", how="inner")
        .withColumn("department_name", F.coalesce(F.col("department_name"), F.lit("Unknown")))
        .withColumn("channel_name", _build_channel_label_expression())
    )

    top_departments = (
        youth_transactions
        .groupBy("department_name")
        .agg(F.count(F.lit(1)).alias("total_purchases"))
        .orderBy(F.desc("total_purchases"))
        .limit(5)
        .select("department_name")
    )

    final_df = (
        youth_transactions
        .join(F.broadcast(top_departments), on="department_name", how="inner")
        .groupBy("department_name", "channel_name")
        .agg(F.count(F.lit(1)).alias("purchase_count"))
        .orderBy("department_name", "channel_name")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 5: Top 5 departments for customers under 25",
        output_name="q5_top_departments_youth",
        directories=directories,
        plotter=_plot_q5_top_youth_departments,
    )


def _question_6(transactions_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    """Q6: How many new customers were acquired each month (cumulative)? 
    Operations: window (min, unbounded sum). Plot: area chart."""
    first_purchase_window = Window.partitionBy("customer_id")
    
    first_purchases = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .withColumn("first_transaction_date", F.min("t_dat").over(first_purchase_window))
        .filter(F.col("t_dat") == F.col("first_transaction_date"))
        .select("customer_id", "first_transaction_date")
        .dropDuplicates(["customer_id"])
    )

    cumulative_window = (
        Window.orderBy("month_start")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    final_df = (
        first_purchases
        .withColumn("month_start", F.trunc("first_transaction_date", "month"))
        .groupBy("month_start")
        .agg(F.count(F.lit(1)).alias("new_customers"))
        .withColumn("cumulative_new_customers", F.sum("new_customers").over(cumulative_window))
        .orderBy("month_start")
    )

    _execute_output_bundle(
        df=final_df,
        question_label="Question 6: Cumulative new customer acquisition over time",
        output_name="q6_cumulative_new_customers",
        directories=directories,
        plotter=_plot_q6_cumulative_new_customers,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    """Execute the sequential extraction and parsing of entire transformation workloads assigned individually."""
    directories = _ensure_output_dirs(output_dir)
    explain_log_path = os.path.join(directories["logs"], "explain_logs.txt")

    with open(explain_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Artem pipeline explain plans\n")

    transactions_df = _load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = _load_parquet_dataset(spark, processed_dir, "articles")
    customers_df = _load_parquet_dataset(spark, processed_dir, "customers")
    partition_count = _recommended_partitions(spark)

    _question_1(customers_df, directories, partition_count)
    _question_2(transactions_df, customers_df, directories, partition_count)
    _question_3(customers_df, directories, partition_count)
    _question_4(transactions_df, customers_df, directories, partition_count)
    _question_5(transactions_df, customers_df, articles_df, directories, partition_count)
    _question_6(transactions_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Artem-Pipeline")

    try:
        run(spark_session)
    finally:
        spark_session.stop()

