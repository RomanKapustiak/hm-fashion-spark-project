from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# explicit schema to keep IDs safe from being parsed as numbers
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", StringType(), True),
    StructField("FN", DoubleType(), True),
    StructField("Active", DoubleType(), True),
    StructField("club_member_status", StringType(), True),
    StructField("fashion_news_frequency", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("postal_code", StringType(), True)
])

def load_customers(spark, path):
    """Load customers.csv using the explicit schema."""
    return spark.read.csv(
        path,
        header=True,
        schema=CUSTOMERS_SCHEMA
    )
