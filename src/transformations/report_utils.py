from __future__ import annotations

import contextlib
import io
import os
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


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


def ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    csv_dir = os.path.join(output_dir, "csv")
    logs_dir = os.path.join(output_dir, "logs")
    for directory in (output_dir, csv_dir, logs_dir):
        os.makedirs(directory, exist_ok=True)
    return {
        "base": output_dir,
        "csv": csv_dir,
        "logs": logs_dir,
    }


def initialize_explain_log(log_path: str, title: str) -> None:
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"{title}\n")


def recommended_partitions(spark: SparkSession) -> int:
    return max(spark.sparkContext.defaultParallelism, MIN_PARTITIONS)


def load_parquet_dataset(spark: SparkSession, processed_dir: str, dataset_name: str) -> DataFrame:
    path = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return spark.read.parquet(path)


def write_explain_log(df: DataFrame, log_path: str, question_label: str) -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(extended=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 100}\n")
        log_file.write(f"{question_label}\n")
        log_file.write(f"{'=' * 100}\n")
        log_file.write(buffer.getvalue().rstrip())
        log_file.write("\n")


def save_csv(df: DataFrame, csv_dir: str, output_name: str) -> None:
    output_path = os.path.join(csv_dir, output_name)
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(output_path)
    )


def save_report(df: DataFrame, question_label: str, output_name: str, directories: Dict[str, str]) -> None:
    write_explain_log(df, os.path.join(directories["logs"], "explain_logs.txt"), question_label)
    save_csv(df, directories["csv"], output_name)


def build_channel_label_expression(column_name: str = "sales_channel_id") -> F.Column:
    return (
        F.when(F.col(column_name) == F.lit(1), F.lit(CHANNEL_LABELS[1]))
        .when(F.col(column_name) == F.lit(2), F.lit(CHANNEL_LABELS[2]))
        .otherwise(F.concat(F.lit("Channel "), F.col(column_name).cast("string")))
    )


def build_weekday_label_expression(column_name: str = "Weekday_Num") -> F.Column:
    expression = None
    for weekday_number, weekday_name in WEEKDAY_LABELS.items():
        if expression is None:
            expression = F.when(F.col(column_name) == F.lit(weekday_number), F.lit(weekday_name))
        else:
            expression = expression.when(F.col(column_name) == F.lit(weekday_number), F.lit(weekday_name))
    return expression.otherwise(F.lit("Unknown"))


def build_age_group_expression(column_name: str = "age") -> F.Column:
    age_bucket = (F.floor(F.col(column_name) / 10) * 10).cast("int")
    return (
        F.when(F.col(column_name).isNull(), F.lit("Unknown"))
        .otherwise(
            F.concat(
                age_bucket.cast("string"),
                F.lit("-"),
                (age_bucket + F.lit(9)).cast("string"),
            )
        )
    )


def round_existing(df: DataFrame, digits_by_column: Dict[str, int]) -> DataFrame:
    rounded_df = df
    for column_name, digits in digits_by_column.items():
        if column_name in rounded_df.columns:
            rounded_df = rounded_df.withColumn(column_name, F.round(F.col(column_name), digits))
    return rounded_df


def readable_name(column_name: str, fallback: str) -> F.Column:
    return F.initcap(F.coalesce(F.trim(F.col(column_name)), F.lit(fallback)))
