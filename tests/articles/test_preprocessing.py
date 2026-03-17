"""Tests for article-specific preprocessing rules."""

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.articles.preprocessing import run_preprocessing

RAW_SCHEMA = StructType([
    StructField("article_id", StringType(), True),
    StructField("prod_name", StringType(), True),
    StructField("product_type_name", StringType(), True),
    StructField("product_group_name", StringType(), True),
    StructField("department_name", StringType(), True),
    StructField("index_name", StringType(), True),
    StructField("section_name", StringType(), True),
    StructField("garment_group_name", StringType(), True),
    StructField("detail_desc", StringType(), True),
    StructField("product_code", IntegerType(), True),
])

CSV_HEADER = (
    "article_id,prod_name,product_type_name,product_group_name,department_name,"
    "index_name,section_name,garment_group_name,detail_desc,product_code\n"
)


def _make_df(spark, tmp_path, body: str):
    csv_file = tmp_path / "articles_preprocessing.csv"
    csv_file.write_text(CSV_HEADER + body)
    return spark.read.option("header", "true").schema(RAW_SCHEMA).csv(str(csv_file))


class TestRunPreprocessing:
    def test_text_columns_are_lowercased(self, spark, tmp_path):
        body = "0108775015,Soft T-SHIRT,T-SHIRT,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,BASIC COTTON TOP,108775\n"
        result = run_preprocessing(_make_df(spark, tmp_path, body)).first().asDict()

        assert result["prod_name"] == "soft t-shirt"
        assert result["product_type_name"] == "t-shirt"
        assert result["detail_desc"] == "basic cotton top"

    def test_rows_with_null_required_fields_are_dropped(self, spark, tmp_path):
        body = (
            "0108775015,Soft T-shirt,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108775\n"
            "0739590027,,Dress,Garment Full body,Dresses,Divided,Young Girl,Dresses Ladies,Short sleeve,739590\n"
        )
        result = run_preprocessing(_make_df(spark, tmp_path, body))

        assert result.count() == 1
        assert result.select("article_id").first()[0] == "0108775015"

    def test_deduplicates_by_article_id(self, spark, tmp_path):
        body = (
            "0108775015,Soft T-shirt,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108775\n"
            "0108775015,Soft T-shirt,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108775\n"
            "0739590027,Summer Dress,Dress,Garment Full body,Dresses,Divided,Young Girl,Dresses Ladies,Short sleeve,739590\n"
        )
        result = run_preprocessing(_make_df(spark, tmp_path, body))

        assert result.count() == 2
        assert result.select("article_id").distinct().count() == 2

    def test_deduplicate_conflicts_keep_deterministic_row(self, spark, tmp_path):
        body = (
            "0108775015,Zeta Tee,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108775\n"
            "0108775015,Alpha Tee,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108776\n"
            "0739590027,Summer Dress,Dress,Garment Full body,Dresses,Divided,Young Girl,Dresses Ladies,Short sleeve,739590\n"
        )
        result = run_preprocessing(_make_df(spark, tmp_path, body))
        kept_row = result.filter(result.article_id == "0108775015").first().asDict()

        assert result.count() == 2
        assert kept_row["prod_name"] == "alpha tee"
        assert kept_row["product_code"] == 108776

    def test_preserves_valid_rows(self, spark, tmp_path):
        body = (
            "0108775015,Soft T-shirt,T-shirt,Garment Upper body,Jersey Basic,Ladieswear,Everyday,Jersey Basic,Cotton top,108775\n"
            "0739590027,Summer Dress,Dress,Garment Full body,Dresses,Divided,Young Girl,Dresses Ladies,Short sleeve,739590\n"
        )
        result = run_preprocessing(_make_df(spark, tmp_path, body))

        assert result.count() == 2
        assert set(result.columns) == set(RAW_SCHEMA.fieldNames())
