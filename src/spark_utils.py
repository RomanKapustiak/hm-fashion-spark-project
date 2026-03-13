# shared spark session factory - everyone imports from here
# usage: from src.spark_utils import create_spark_session

from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "HM-Fashion-Spark") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
