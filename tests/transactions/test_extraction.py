# tests for transactions extraction (schema + loader)

import pytest
from pyspark.sql.types import StringType, DoubleType, IntegerType, DateType

from src.transactions.schema import TRANSACTIONS_SCHEMA, load_transactions

# SparkSession provided by tests/conftest.py


class TestTransactionsSchema:
    def test_schema_has_correct_field_names(self):
        expected = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}
        assert {f.name for f in TRANSACTIONS_SCHEMA.fields} == expected

    def test_t_dat_is_date_type(self):
        field = next(f for f in TRANSACTIONS_SCHEMA.fields if f.name == "t_dat")
        assert isinstance(field.dataType, DateType)

    def test_customer_id_is_string_type(self):
        field = next(f for f in TRANSACTIONS_SCHEMA.fields if f.name == "customer_id")
        assert isinstance(field.dataType, StringType)

    def test_article_id_is_string_type(self):
        # must be string - article_id has leading zeros
        field = next(f for f in TRANSACTIONS_SCHEMA.fields if f.name == "article_id")
        assert isinstance(field.dataType, StringType)

    def test_price_is_double_type(self):
        field = next(f for f in TRANSACTIONS_SCHEMA.fields if f.name == "price")
        assert isinstance(field.dataType, DoubleType)

    def test_sales_channel_id_is_integer_type(self):
        field = next(f for f in TRANSACTIONS_SCHEMA.fields if f.name == "sales_channel_id")
        assert isinstance(field.dataType, IntegerType)

    def test_schema_has_exactly_five_fields(self):
        assert len(TRANSACTIONS_SCHEMA.fields) == 5


class TestLoadTransactions:
    SAMPLE_CSV = (
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2019-01-15,abc123,0663713001,0.05083,2\n"
        "2019-03-22,def456,0541518023,0.03049,1\n"
        "2020-07-04,ghi789,0706016001,0.08136,2\n"
    )

    def test_load_returns_dataframe_with_correct_row_count(self, spark, tmp_path):
        csv_file = tmp_path / "transactions_train.csv"
        csv_file.write_text(self.SAMPLE_CSV)
        assert load_transactions(spark, str(csv_file)).count() == 3

    def test_load_returns_correct_column_names(self, spark, tmp_path):
        csv_file = tmp_path / "transactions_train.csv"
        csv_file.write_text(self.SAMPLE_CSV)
        df = load_transactions(spark, str(csv_file))
        assert set(df.columns) == {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}

    def test_price_column_is_double(self, spark, tmp_path):
        csv_file = tmp_path / "transactions_train.csv"
        csv_file.write_text(self.SAMPLE_CSV)
        assert dict(load_transactions(spark, str(csv_file)).dtypes)["price"] == "double"

    def test_article_id_preserves_leading_zeros(self, spark, tmp_path):
        csv_file = tmp_path / "transactions_train.csv"
        csv_file.write_text(self.SAMPLE_CSV)
        first = load_transactions(spark, str(csv_file)).select("article_id").first()[0]
        assert first.startswith("0")
