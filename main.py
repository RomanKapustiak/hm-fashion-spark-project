"""
H&M Fashion Recommendations - PySpark pipeline entry point

Run with: docker compose run spark-app python main.py
"""

from src.spark_utils import create_spark_session

from src.transactions.pipeline import run as transactions_run
# from src.articles.pipeline import run as articles_run
from src.customers.pipeline import run as customers_run

if __name__ == "__main__":
    spark = create_spark_session("HM-Fashion-Pipeline")

    try:
        transactions_run(spark)
        # articles_run(spark)
        customers_run(spark)
    finally:
        spark.stop()

