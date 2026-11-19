"""Collects test metrics of all members' models and compares them.

Reads output/ml/<member>/metrics__*.csv, adds trivial baselines and writes
output/ml/comparison/<task>_comparison.csv plus bar charts.

Run (after the model scripts):
    docker compose run --rm spark-app python src/analysis/compare.py
"""

from __future__ import annotations

import glob
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.analysis.common import (  # noqa: E402
    ARTICLES,
    CLASSIFICATION,
    CUSTOMERS,
    OUTPUT_DIR,
    REGRESSION,
    SPLIT_COL,
    baseline_metrics,
    load_dataset,
    save_plot,
    save_table,
)
from src.spark_utils import create_spark_session  # noqa: E402

COMPARISON = "comparison"
DISPLAY = {
    CLASSIFICATION: (["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"], "f1"),
    REGRESSION: (["rmse", "mae", "r2"], "r2"),
}


def _baselines() -> pd.DataFrame:
    spark = create_spark_session("HM-Compare-Baselines")
    rows = []
    try:
        for spec, name in ((CUSTOMERS, "majority_class"), (ARTICLES, "train_mean")):
            df = load_dataset(spark, spec)
            train, test = df.filter(F.col(SPLIT_COL) == "train"), df.filter(F.col(SPLIT_COL) == "test")
            rows.append({"member": "baseline", "task": spec.task, "model": name,
                         **baseline_metrics(train, test, spec.task)})
    finally:
        spark.stop()
    return pd.DataFrame(rows)


def _member_metrics() -> pd.DataFrame:
    files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*", "metrics__*.csv"))
             if os.sep + COMPARISON + os.sep not in f]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _bar_chart(table: pd.DataFrame, metrics, title: str):
    labels = table["member"] + "\n" + table["model"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 0.45 * len(table) + 2), sharey=True)
    for ax, metric in zip(axes, metrics):
        ax.barh(labels, table[metric], color="#457b9d")
        for y, value in enumerate(table[metric]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(title)
    return fig


def main() -> None:
    results = pd.concat([_member_metrics(), _baselines()], ignore_index=True)
    for task, (metrics, sort_by) in DISPLAY.items():
        table = results[results["task"] == task]
        if table.empty:
            continue
        columns = ["member", "model", *metrics, "threshold", "fit_seconds", "num_features", "params"]
        table = table[[c for c in columns if c in table.columns]].sort_values(sort_by, ascending=False)
        path = save_table(table, COMPARISON, f"{task}_comparison")
        print(f"\n=== {task} (test set) -> {path}")
        print(table.drop(columns=["params"], errors="ignore").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        chart_metrics = [m for m in metrics if m in table.columns and table[m].notna().any()]
        save_plot(_bar_chart(table.fillna(0), chart_metrics[:4], f"{task}: test metrics"), COMPARISON, f"{task}_comparison")


if __name__ == "__main__":
    main()
