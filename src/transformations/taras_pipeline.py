from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import create_spark_session

MEMBER_NAME = "taras"
MAX_PANDAS_ROWS = 10_000
MIN_PARTITIONS = 8

sns.set_theme(style="whitegrid", palette="muted")

def _ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    """Create csv / plots / logs sub-directories and return their paths."""
    csv_dir = os.path.join(output_dir, "csv")
    plots_dir = os.path.join(output_dir, "plots")
    logs_dir = os.path.join(output_dir, "logs")
    for d in (output_dir, csv_dir, plots_dir, logs_dir):
        os.makedirs(d, exist_ok=True)
    return {"base": output_dir, "csv": csv_dir, "plots": plots_dir, "logs": logs_dir}


def _recommended_partitions(spark: SparkSession) -> int:
    return max(spark.sparkContext.defaultParallelism, MIN_PARTITIONS)


def _load_parquet(spark: SparkSession, processed_dir: str, name: str) -> DataFrame:
    path = os.path.join(processed_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return spark.read.parquet(path)


def _write_explain_log(df: DataFrame, log_path: str, label: str) -> None:
    """Append the extended explain plan for *df* to *log_path*."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(extended=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 100}\n")
        fh.write(f"{label}\n")
        fh.write(f"{'=' * 100}\n")
        fh.write(buf.getvalue().rstrip())
        fh.write("\n")


def _save_csv(df: DataFrame, csv_dir: str, name: str) -> None:
    out = os.path.join(csv_dir, name)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out)


def _to_small_pandas(df: DataFrame, name: str, max_rows: int = MAX_PANDAS_ROWS) -> pd.DataFrame:
    n = df.count()
    if n > max_rows:
        raise ValueError(
            f"Refusing to convert '{name}' to pandas: {n} rows (limit {max_rows})."
        )
    return df.toPandas()


def _finalize_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_no_data_plot(path: str, title: str, msg: str = "No data available.") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=16, color="#4c566a")
    ax.set_title(title, fontsize=16, weight="bold")
    _finalize_figure(fig, path)


def _execute_output_bundle(
    df: DataFrame,
    question_label: str,
    output_name: str,
    dirs: Dict[str, str],
    plotter,
) -> None:
    """Log explain plan, save CSV, convert to Pandas and plot."""
    _write_explain_log(df, os.path.join(dirs["logs"], "explain_logs.txt"), question_label)
    _save_csv(df, dirs["csv"], output_name)
    pdf = _to_small_pandas(df, output_name)
    plot_path = os.path.join(dirs["plots"], f"{output_name}.png")
    plotter(pdf, plot_path)


def _plot_q1_donut(pdf: pd.DataFrame, path: str) -> None:
    """Donut chart — unique article counts by global index group."""
    title = "Unique Articles per Global Index Group"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.sort_values("unique_articles", ascending=False).copy()
    df["index_group_name"] = df["index_group_name"].fillna("unknown").str.title()

    colors = sns.color_palette("Set2", n_colors=len(df))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        df["unique_articles"],
        labels=df["index_group_name"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.78,
        wedgeprops={"linewidth": 2, "edgecolor": "white", "width": 0.45},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("#2e3440")
    for t in texts:
        t.set_fontsize(11)
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    _finalize_figure(fig, path)


def _question_1(articles_df: DataFrame, dirs: Dict[str, str]) -> None:
    """Q1: How many unique articles exist in each index_group_name?"""
    final = (
        articles_df
        .groupBy("index_group_name")
        .agg(F.countDistinct("article_id").alias("unique_articles"))
        .orderBy(F.desc("unique_articles"))
    )
    _execute_output_bundle(
        df=final,
        question_label="Question 1: Unique articles per index_group_name (GroupBy, Count)",
        output_name="q1_unique_articles_per_index_group",
        dirs=dirs,
        plotter=_plot_q1_donut,
    )


def _plot_q2_hbar(pdf: pd.DataFrame, path: str) -> None:
    """Horizontal bar chart — departments ranked by sales count (Ladieswear)."""
    title = "Top Departments by Sales Count (Ladieswear)"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.sort_values("sales_count", ascending=True).tail(15).copy()
    df["department_name"] = df["department_name"].fillna("unknown").str.title()

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("viridis", n_colors=len(df))
    ax.barh(df["department_name"], df["sales_count"], color=palette)
    ax.set_xlabel("Number of Sales", fontsize=12)
    ax.set_ylabel("Department", fontsize=12)
    ax.set_title(title, fontsize=16, weight="bold")

    for i, (val, name) in enumerate(zip(df["sales_count"], df["department_name"])):
        ax.text(val + df["sales_count"].max() * 0.01, i, f"{val:,.0f}",
                va="center", fontsize=9, color="#2e3440")
    _finalize_figure(fig, path)


def _question_2(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    """Q2: Which department generates the most sales for Ladieswear?"""
    ladieswear = articles_df.filter(F.col("index_name") == F.lit("ladieswear"))

    final = (
        transactions_df
        .repartition(partitions, "article_id")
        .alias("t")
        .join(F.broadcast(ladieswear).alias("a"), on="article_id", how="inner")
        .groupBy("department_name")
        .agg(F.count(F.lit(1)).alias("sales_count"))
        .orderBy(F.desc("sales_count"))
    )
    _execute_output_bundle(
        df=final,
        question_label="Question 2: Top department by sales for Ladieswear (Filter, Join, GroupBy, Count)",
        output_name="q2_ladieswear_department_sales",
        dirs=dirs,
        plotter=_plot_q2_hbar,
    )


def _plot_q3_top3_table(pdf: pd.DataFrame, path: str) -> None:
    """Styled matplotlib table showing top-3 products per garment group."""
    title = "Top-3 Most Expensive Trousers per Garment Group"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.copy()
    df["garment_group_name"] = df["garment_group_name"].fillna("unknown").str.title()
    df["prod_name"] = df["prod_name"].fillna("unknown").str.title()
    df["max_price"] = df["max_price"].apply(lambda v: f"{v:.4f}")
    df["price_rank"] = df["price_rank"].astype(int)

    table_data = df[["garment_group_name", "prod_name", "max_price", "price_rank"]].values.tolist()
    col_labels = ["Garment Group", "Product Name", "Max Historical Price", "Rank"]

    n_rows = len(table_data)
    fig_height = max(4, 0.5 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4C78A8")
        cell.set_text_props(color="white", weight="bold")

    for i in range(1, n_rows + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    _finalize_figure(fig, path)


def _question_3(articles_df: DataFrame, transactions_df: DataFrame, dirs: Dict[str, str], partitions: int) -> None:
    """Q3: Top 3 most expensive products (by max historical price) per garment group, Trousers only.

    Operations: Filter (trousers), Window (dense_rank over max price).
    """
    article_max_price = (
        transactions_df
        .groupBy("article_id")
        .agg(F.max("price").alias("max_price"))
    )

    trousers = articles_df.filter(F.col("product_type_name") == F.lit("trousers"))

    trousers_with_price = (
        trousers.alias("a")
        .join(article_max_price.alias("p"), on="article_id", how="inner")
        .select("a.garment_group_name", "a.prod_name", "a.article_id", "p.max_price")
    )

    price_window = Window.partitionBy("garment_group_name").orderBy(F.desc("max_price"))

    final = (
        trousers_with_price
        .withColumn("price_rank", F.dense_rank().over(price_window))
        .filter(F.col("price_rank") <= 3)
        .orderBy("garment_group_name", "price_rank")
    )

    _execute_output_bundle(
        df=final,
        question_label="Question 3: Top-3 most expensive trousers per garment group (Filter, Window/dense_rank)",
        output_name="q3_top3_expensive_trousers",
        dirs=dirs,
        plotter=_plot_q3_top3_table,
    )


def _plot_q4_pareto(pdf: pd.DataFrame, path: str) -> None:
    """Pareto chart — section sales count + cumulative %."""
    title = "Pareto: Top-5 Sections by Sales in Baby/Children"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.sort_values("sales_count", ascending=False).head(5).copy()
    df["section_name"] = df["section_name"].fillna("unknown").str.title()
    df = df.reset_index(drop=True)

    total = df["sales_count"].sum()
    df["cumulative_pct"] = df["sales_count"].cumsum() / total * 100

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_colors = sns.color_palette("Blues_d", n_colors=len(df))
    bars = ax1.bar(df["section_name"], df["sales_count"], color=bar_colors,
                   edgecolor="white", linewidth=1.2, zorder=3)
    ax1.set_xlabel("Section", fontsize=12)
    ax1.set_ylabel("Number of Sales", fontsize=12, color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.tick_params(axis="x", rotation=25)

    for bar, val in zip(bars, df["sales_count"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=9, color="#2e3440")

    ax2 = ax1.twinx()
    ax2.plot(df["section_name"], df["cumulative_pct"], color="#E07A5F",
             marker="o", linewidth=2.5, markersize=8, zorder=4)
    ax2.set_ylabel("Cumulative Share (%)", fontsize=12, color="#E07A5F")
    ax2.tick_params(axis="y", labelcolor="#E07A5F")
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    ax1.set_title(title, fontsize=16, weight="bold")
    ax1.grid(axis="y", alpha=0.3)
    _finalize_figure(fig, path)


def _question_4(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    """Q4: Sales share (%) of top-5 most popular sections in Baby/Children."""
    baby_articles = articles_df.filter(
        F.lower(F.col("index_group_name")) == F.lit("baby/children")
    )

    joined = (
        transactions_df
        .repartition(partitions, "article_id")
        .alias("t")
        .join(F.broadcast(baby_articles).alias("a"), on="article_id", how="inner")
    )

    section_sales = (
        joined
        .groupBy("section_name")
        .agg(F.count(F.lit(1)).alias("sales_count"))
    )

    total_window = Window.partitionBy()  # entire dataset
    final = (
        section_sales
        .withColumn("total_sales", F.sum("sales_count").over(total_window))
        .withColumn("sales_share_pct", (F.col("sales_count") / F.col("total_sales")) * 100.0)
        .orderBy(F.desc("sales_count"))
    )

    _execute_output_bundle(
        df=final,
        question_label="Question 4: Sales share of top-5 sections in Baby/Children (Filter, Join, GroupBy, Window/sum-over)",
        output_name="q4_baby_children_section_share",
        dirs=dirs,
        plotter=_plot_q4_pareto,
    )


try:
    import squarify

    _HAS_SQUARIFY = True
except ImportError:
    _HAS_SQUARIFY = False


def _plot_q5_treemap(pdf: pd.DataFrame, path: str) -> None:
    """Treemap — sweater sales distribution across index groups."""
    title = "Sweater Sales Distribution by Index Group (Treemap)"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.sort_values("sales_count", ascending=False).copy()
    df["index_group_name"] = df["index_group_name"].fillna("unknown").str.title()
    total = df["sales_count"].sum()
    df["pct"] = (df["sales_count"] / total * 100).round(1)
    labels = [f"{n}\n{c:,.0f} ({p}%)" for n, c, p in
              zip(df["index_group_name"], df["sales_count"], df["pct"])]

    if _HAS_SQUARIFY:
        colors = sns.color_palette("Set2", n_colors=len(df))
        fig, ax = plt.subplots(figsize=(14, 8))
        squarify.plot(
            sizes=df["sales_count"].tolist(),
            label=labels,
            color=colors,
            alpha=0.85,
            ax=ax,
            text_kwargs={"fontsize": 11, "weight": "bold", "color": "#2e3440"},
            edgecolor="white",
            linewidth=3,
        )
        ax.axis("off")
        ax.set_title(title, fontsize=16, weight="bold", pad=15)
        _finalize_figure(fig, path)
    else:
        colors = sns.color_palette("Set2", n_colors=len(df))
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(df["index_group_name"], df["sales_count"], color=colors,
                edgecolor="white", linewidth=1.5)
        for i, (val, pct) in enumerate(zip(df["sales_count"], df["pct"])):
            ax.text(val + total * 0.005, i, f"{val:,.0f} ({pct}%)",
                    va="center", fontsize=10, color="#2e3440")
        ax.set_xlabel("Number of Sales", fontsize=12)
        ax.set_ylabel("Index Group", fontsize=12)
        ax.set_title(title + "\n(squarify not installed — showing bar chart)",
                     fontsize=14, weight="bold")
        _finalize_figure(fig, path)


def _question_5(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    """Q5: Sweater sales distribution across index groups."""
    sweater_articles = articles_df.filter(
        F.col("product_type_name") == F.lit("sweater")
    )

    final = (
        transactions_df
        .repartition(partitions, "article_id")
        .alias("t")
        .join(F.broadcast(sweater_articles).alias("a"), on="article_id", how="inner")
        .groupBy("index_group_name")
        .agg(F.count(F.lit(1)).alias("sales_count"))
        .orderBy(F.desc("sales_count"))
    )

    _execute_output_bundle(
        df=final,
        question_label="Question 5: Sweater sales by index_group_name (Filter, Join, GroupBy)",
        output_name="q5_sweater_sales_by_index_group",
        dirs=dirs,
        plotter=_plot_q5_treemap,
    )


def _plot_q6_bump(pdf: pd.DataFrame, path: str) -> None:
    """Bump chart — monthly rank movement for top departments."""
    title = "Department Popularity Rank Over Time (Bump Chart)"
    if pdf.empty:
        _save_no_data_plot(path, title)
        return

    df = pdf.copy()
    df["department_name"] = df["department_name"].fillna("unknown").str.title()
    df["year_month"] = df["year_month"].astype(str)

    avg_rank = df.groupby("department_name")["dept_rank"].mean().nsmallest(10)
    top_depts = avg_rank.index.tolist()
    df = df[df["department_name"].isin(top_depts)]

    if df.empty:
        _save_no_data_plot(path, title)
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    palette = sns.color_palette("tab10", n_colors=len(top_depts))

    for idx, dept in enumerate(top_depts):
        dept_df = df[df["department_name"] == dept].sort_values("year_month")
        ax.plot(
            dept_df["year_month"],
            dept_df["dept_rank"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=dept,
            color=palette[idx],
            alpha=0.9,
        )
        if not dept_df.empty:
            last = dept_df.iloc[-1]
            ax.annotate(
                dept,
                xy=(last["year_month"], last["dept_rank"]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                color=palette[idx],
                weight="bold",
                va="center",
            )

    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel("Year-Month", fontsize=12)
    ax.set_ylabel("Popularity Rank", fontsize=12)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.tick_params(axis="x", rotation=45)

    xticks = sorted(df["year_month"].unique())
    step = max(1, len(xticks) // 12)
    ax.set_xticks([xticks[i] for i in range(0, len(xticks), step)])

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        title="Department",
        title_fontsize=10,
        frameon=True,
    )
    ax.grid(axis="y", alpha=0.3)
    _finalize_figure(fig, path)


def _question_6(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    """Q6: Rank all departments by monthly popularity and compute rank change."""
    joined = (
        transactions_df
        .repartition(partitions, "article_id")
        .alias("t")
        .join(F.broadcast(articles_df).alias("a"), on="article_id", how="inner")
    )

    monthly_dept = (
        joined
        .withColumn("year_month", F.date_format(F.col("t_dat"), "yyyy-MM"))
        .groupBy("year_month", "department_name")
        .agg(F.count(F.lit(1)).alias("monthly_sales"))
    )

    rank_window = Window.partitionBy("year_month").orderBy(F.desc("monthly_sales"))

    lag_window = Window.partitionBy("department_name").orderBy("year_month")

    final = (
        monthly_dept
        .withColumn("dept_rank", F.rank().over(rank_window))
        .withColumn("prev_rank", F.lag("dept_rank", 1).over(lag_window))
        .withColumn(
            "rank_change",
            F.when(F.col("prev_rank").isNotNull(), F.col("prev_rank") - F.col("dept_rank"))
            .otherwise(F.lit(None)),
        )
        .orderBy("year_month", "dept_rank")
    )

    _execute_output_bundle(
        df=final,
        question_label="Question 6: Department rank changes month-over-month (Join, GroupBy, Window/rank+lag)",
        output_name="q6_department_rank_bump",
        dirs=dirs,
        plotter=_plot_q6_bump,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    """Execute all 6 Taras business questions."""
    print(f"\n{'=' * 60}")
    print(f"  Taras Pipeline — Product Hierarchy & Department Analysis")
    print(f"{'=' * 60}\n")

    dirs = _ensure_output_dirs(output_dir)
    log_path = os.path.join(dirs["logs"], "explain_logs.txt")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("Taras pipeline explain plans\n")

    articles_df = _load_parquet(spark, processed_dir, "articles")
    transactions_df = _load_parquet(spark, processed_dir, "transactions")
    partitions = _recommended_partitions(spark)

    print("[Q1] Unique articles per index_group_name …")
    _question_1(articles_df, dirs)
    print("     ✓ done")

    print("[Q2] Top department by sales for Ladieswear …")
    _question_2(transactions_df, articles_df, dirs, partitions)
    print("     ✓ done")

    print("[Q3] Top-3 most expensive trousers per garment group …")
    _question_3(articles_df, transactions_df, dirs, partitions)
    print("     ✓ done")

    print("[Q4] Sales share of top-5 sections in Baby/Children …")
    _question_4(transactions_df, articles_df, dirs, partitions)
    print("     ✓ done")

    print("[Q5] Sweater sales by index group …")
    _question_5(transactions_df, articles_df, dirs, partitions)
    print("     ✓ done")

    print("[Q6] Department rank changes month-over-month …")
    _question_6(transactions_df, articles_df, dirs, partitions)
    print("     ✓ done")

    print(f"\n{'=' * 60}")
    print(f"  All 6 questions completed. Outputs → output/{MEMBER_NAME}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Taras-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
