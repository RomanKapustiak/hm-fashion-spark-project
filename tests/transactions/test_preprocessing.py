# tests for transactions preprocessing

import pytest
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType
from pyspark.sql import functions as F

from src.transactions.preprocessing import (
    run_preprocessing,
    _step3_type_casting,
    _step5_nulls_and_duplicates,
)

# SparkSession provided by tests/conftest.py

RAW_SCHEMA = StructType([
    StructField("t_dat",             DateType(),    True),
    StructField("customer_id",       StringType(),  True),
    StructField("article_id",        StringType(),  True),
    StructField("price",             DoubleType(),  True),
    StructField("sales_channel_id",  IntegerType(), True),
])


def _make_df(spark, rows):
    return spark.createDataFrame(rows, schema=RAW_SCHEMA)


class TestTypeCasting:
    def test_derived_date_columns_added(self, spark):
        from datetime import date
        df = _make_df(spark, [(date(2020, 3, 15), "aaa", "0001", 0.05, 1)])
        result = _step3_type_casting(df)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day_of_week" in result.columns

    def test_year_extracted_correctly(self, spark):
        from datetime import date
        df = _make_df(spark, [(date(2021, 7, 4), "aaa", "0001", 0.05, 2)])
        assert _step3_type_casting(df).select("year").first()[0] == 2021

    def test_month_extracted_correctly(self, spark):
        from datetime import date
        df = _make_df(spark, [(date(2019, 11, 20), "aaa", "0001", 0.05, 1)])
        assert _step3_type_casting(df).select("month").first()[0] == 11


class TestNullsAndDuplicates:
    def test_rows_with_null_price_are_dropped(self, spark):
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 1, 2), "bbb", "0002", None, 2),
        ]
        assert _step5_nulls_and_duplicates(_make_df(spark, rows)).count() == 1

    def test_rows_with_null_customer_id_are_dropped(self, spark):
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 1, 2), None, "0002", 0.03, 2),
        ]
        assert _step5_nulls_and_duplicates(_make_df(spark, rows)).count() == 1

    def test_exact_duplicates_are_kept(self, spark):
        # customers can buy same item twice - DO NOT drop any rows even if 100% identical
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),  # exact dup - keep
            (date(2020, 1, 2), "bbb", "0002", 0.03, 1),
            (date(2020, 1, 2), "bbb", "0002", 0.03, 2),  # different channel - keep
        ]
        assert _step5_nulls_and_duplicates(_make_df(spark, rows)).count() == 4


    def test_clean_data_unchanged(self, spark):
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 1, 2), "bbb", "0002", 0.03, 2),
            (date(2020, 1, 3), "ccc", "0003", 0.07, 1),
        ]
        assert _step5_nulls_and_duplicates(_make_df(spark, rows)).count() == 3


class TestFullPreprocessingPipeline:
    def test_pipeline_runs_on_clean_data(self, spark):
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 2, 14), "bbb", "0002", 0.03, 2),
            (date(2020, 6, 1), "ccc", "0003", 0.07, 1),
        ]
        assert run_preprocessing(_make_df(spark, rows)).count() > 0

    def test_pipeline_output_has_derived_columns(self, spark):
        from datetime import date
        df = _make_df(spark, [(date(2020, 5, 10), "aaa", "0001", 0.05, 1)])
        result = run_preprocessing(df)
        for col in ("year", "month", "day_of_week"):
            assert col in result.columns

    def test_pipeline_removes_nulls(self, spark):
        from datetime import date
        rows = [
            (date(2020, 1, 1), "aaa", "0001", 0.05, 1),
            (date(2020, 1, 2), None, "0002", 0.03, 2),
        ]
        assert run_preprocessing(_make_df(spark, rows)).count() == 1
