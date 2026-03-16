"""Schema definitions and extraction helpers for articles.csv."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

ARTICLES_SCHEMA: StructType = StructType([
    StructField("article_id", StringType(), True),
    StructField("product_code", IntegerType(), True),
    StructField("prod_name", StringType(), True),
    StructField("product_type_no", IntegerType(), True),
    StructField("product_type_name", StringType(), True),
    StructField("product_group_name", StringType(), True),
    StructField("graphical_appearance_no", IntegerType(), True),
    StructField("graphical_appearance_name", StringType(), True),
    StructField("colour_group_code", IntegerType(), True),
    StructField("colour_group_name", StringType(), True),
    StructField("perceived_colour_value_id", IntegerType(), True),
    StructField("perceived_colour_value_name", StringType(), True),
    StructField("perceived_colour_master_id", IntegerType(), True),
    StructField("perceived_colour_master_name", StringType(), True),
    StructField("department_no", IntegerType(), True),
    StructField("department_name", StringType(), True),
    StructField("index_code", StringType(), True),
    StructField("index_name", StringType(), True),
    StructField("index_group_no", IntegerType(), True),
    StructField("index_group_name", StringType(), True),
    StructField("section_no", IntegerType(), True),
    StructField("section_name", StringType(), True),
    StructField("garment_group_no", IntegerType(), True),
    StructField("garment_group_name", StringType(), True),
    StructField("detail_desc", StringType(), True),
])


def load_articles(spark: SparkSession, path: str) -> DataFrame:
    """Load articles.csv with explicit schema and no inferSchema."""
    return spark.read.schema(ARTICLES_SCHEMA).option("header", "true").csv(path)
