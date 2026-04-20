from datetime import date
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from src.transformations import dmytro_pipeline as dp


def _sample_transactions(spark):
    rows = [
        (date(2019, 1, 2), "c1", "a1", 0.050, 1),
        (date(2019, 1, 5), "c1", "a2", 0.030, 2),
        (date(2019, 6, 4), "c2", "a1", 0.040, 1),
        (date(2019, 12, 10), "c2", "a3", 0.070, 2),
        (date(2019, 12, 12), "c2", "a3", 0.060, 2),
        (date(2019, 8, 2), "c3", "a2", 0.020, 1),
    ]
    return spark.createDataFrame(
        rows,
        ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    )


def _sample_articles(spark):
    rows = [("a1", "Top"), ("a2", "Bottom"), ("a3", "Dress")]
    return spark.createDataFrame(rows, ["article_id", "product_group_name"])


def _sample_customers(spark):
    rows = [("c1", 35, "ACTIVE"), ("c2", 38, "PRE-CREATE"), ("c3", 24, "ACTIVE")]
    return spark.createDataFrame(rows, ["customer_id", "age", "club_member_status"])


def test_query_1_weekly_transaction_dynamics(spark):
    result = dp.query_1_weekly_transaction_dynamics(_sample_transactions(spark))
    assert {
        "year",
        "week",
        "transactions_count",
        "prev_week_transactions",
        "wow_change_pct",
        "prev_year_transactions",
        "yoy_change_pct",
    } <= set(result.columns)
    assert result.agg(F.sum("transactions_count")).collect()[0][0] == 6


def test_query_2_winter_summer_spike(spark):
    result = dp.query_2_winter_vs_summer_spike(
        _sample_transactions(spark),
        _sample_articles(spark),
    )
    assert {
        "product_group_name",
        "winter_sales",
        "summer_sales",
        "sales_spike",
        "seasonal_delta_pct",
        "seasonality_index",
    } <= set(result.columns)
    top_product = result.orderBy(F.col("sales_spike").desc()).first()["product_group_name"]
    assert top_product == "Dress"


def test_query_3_retention_days_distribution(spark):
    result = dp.query_3_retention_gap_distribution(_sample_transactions(spark))
    assert {"days_between", "customers_count"} <= set(result.columns)
    assert result.filter(F.col("days_between") == 3).count() == 1


def test_query_3_plot_preparation_buckets_tail_and_preserves_total_customers():
    raw = pd.DataFrame(
        {
            "days_between": [0, 1, 2, 3, 10, 45, 120, 400],
            "customers_count": [1000, 90, 80, 70, 60, 50, 40, 30],
        }
    )

    prepared = dp._prepare_query_3_plot_data(raw)

    assert {"bucket_label", "customers_count", "customers_share_pct"} <= set(prepared.columns)
    assert len(prepared) < len(raw)
    assert prepared["bucket_label"].iloc[0] == "0 days (same day)"
    assert int(prepared["customers_count"].sum()) == int(raw["customers_count"].sum())


def test_query_4_top_peak_days_share(spark):
    result = dp.query_4_top_peak_days_month_share(_sample_transactions(spark))
    assert {
        "t_dat",
        "daily_revenue",
        "monthly_revenue",
        "monthly_share",
        "monthly_median_daily",
        "peak_intensity_ratio",
        "z_score",
    } <= set(result.columns)
    assert result.count() <= 10


def test_query_5_rolling_activity_age_30_40(spark):
    result = dp.query_5_rolling_activity_30_40(
        _sample_transactions(spark),
        _sample_customers(spark),
    )
    assert {
        "t_dat",
        "purchases_count",
        "rolling_avg_7d",
        "rolling_std_7d",
        "upper_band_2sigma",
        "lower_band_2sigma",
        "is_outlier",
    } <= set(result.columns)
    assert result.count() > 0


def test_query_6_frequency_active_vs_precreate(spark):
    result = dp.query_6_frequency_loyal_vs_guest(
        _sample_transactions(spark),
        _sample_customers(spark),
    )
    assert {
        "club_member_status",
        "day_of_week",
        "purchases_count",
        "segment_total_purchases",
        "segment_day_share_pct",
        "weekday_index100",
    } <= set(result.columns)
    statuses = {row["club_member_status"] for row in result.select("club_member_status").distinct().collect()}
    assert statuses == {"ACTIVE", "PRE-CREATE"}

    share_sums = (
        result.groupBy("club_member_status")
        .agg(F.sum("segment_day_share_pct").alias("sum_pct"))
        .collect()
    )
    for row in share_sums:
        assert 99.9 <= row["sum_pct"] <= 100.1


def test_run_creates_outputs(spark, tmp_path: Path):
    processed_dir = tmp_path / "processed"
    output_root = tmp_path / "output" / "dmytro"

    _sample_transactions(spark).write.mode("overwrite").parquet(str(processed_dir / "transactions"))
    _sample_articles(spark).write.mode("overwrite").parquet(str(processed_dir / "articles"))
    _sample_customers(spark).write.mode("overwrite").parquet(str(processed_dir / "customers"))

    dp.run(
        spark=spark,
        processed_dir=str(processed_dir),
        output_root=str(output_root),
    )

    explain_log = output_root / "logs" / "explain_logs.txt"
    assert explain_log.exists()
    assert "Query 1" in explain_log.read_text(encoding="utf-8")

    csv_dir = output_root / "csv"
    plot_dir = output_root / "plots"
    assert csv_dir.exists()
    assert plot_dir.exists()
    assert len(list(csv_dir.glob("*"))) > 0
    assert len(list(plot_dir.glob("*.png"))) == 6
