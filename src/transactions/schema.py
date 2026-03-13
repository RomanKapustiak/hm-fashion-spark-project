from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, DateType,
)

# explicit schema so spark doesn't guess types wrong
# article_id stays as string because it has leading zeros (e.g. 0663713001)
TRANSACTIONS_SCHEMA: StructType = StructType([
    StructField("t_dat",             DateType(),    True),  # purchase date
    StructField("customer_id",       StringType(),  True),  # sha256 hash
    StructField("article_id",        StringType(),  True),  # keep as string, leading zeros
    StructField("price",             DoubleType(),  True),  # normalized price (~0.0-0.6)
    StructField("sales_channel_id",  IntegerType(), True),  # 1=store, 2=online
])


def load_transactions(spark: SparkSession, path: str) -> DataFrame:
    """load transactions csv with explicit schema, no inferSchema"""
    return (
        spark.read
        .schema(TRANSACTIONS_SCHEMA)
        .option("header", "true")
        .option("dateFormat", "yyyy-MM-dd")
        .csv(path)
    )
