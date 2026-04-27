"""Artem customer demographics and behavior transformation pipeline."""

from __future__ import annotations

import os
import sys
from typing import Dict

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark_utils import create_spark_session
from src.transformations.report_utils import (
    build_age_group_expression,
    build_channel_label_expression,
    ensure_output_dirs,
    initialize_explain_log,
    load_parquet_dataset,
    recommended_partitions,
    round_existing,
    save_report,
)

MEMBER_NAME = "artem"


def _customer_transaction_profile(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    partition_count: int,
) -> DataFrame:
    return (
        transactions_df
        .repartition(partition_count, "customer_id")
        .groupBy("customer_id", "sales_channel_id")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Spend"),
            F.avg("price").alias("Avg_Price"),
        )
        .join(
            customers_df.select(
                "customer_id",
                "age",
                "club_member_status",
                "fashion_news_frequency",
            ),
            on="customer_id",
            how="inner",
        )
        .withColumn("Age_Group", build_age_group_expression("age"))
        .withColumn(
            "Club_Member_Status",
            F.initcap(F.coalesce(F.trim(F.col("club_member_status")), F.lit("Unknown"))),
        )
        .withColumn(
            "Fashion_News_Frequency",
            F.initcap(F.coalesce(F.trim(F.col("fashion_news_frequency")), F.lit("Unknown"))),
        )
        .withColumn("Channel_Name", build_channel_label_expression())
    )


# Питання: Якими є показники клієнтської активності та витрат у 10 найчисленніших сегментах, сформованих за віковою групою та статусом клубного членства?
# Опис колонок:
# - Вікова_група: Віковий діапазон клієнтів.
# - Статус_клубного_членства: Текстовий статус клубного членства клієнта.
# - Кількість_клієнтів: Кількість унікальних клієнтів у сегменті.
# - Кількість_транзакцій: Загальна кількість транзакцій у сегменті.
# - Середня_кількість_транзакцій_на_клієнта: Середня кількість транзакцій на одного клієнта сегмента.
# - Середні_витрати_на_клієнта: Середня сума витрат одного клієнта сегмента.
# - Середня_ціна_товару: Середня ціна однієї покупки в сегменті.
def _question_1(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    profile_df = _customer_transaction_profile(transactions_df, customers_df, partition_count)

    final_df = (
        profile_df
        .filter(F.col("age").isNotNull())
        .filter(F.col("Transaction_Count") > 0)
        .groupBy("Age_Group", "Club_Member_Status")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.sum("Transaction_Count").alias("Transaction_Count"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Customer"),
            F.avg("Total_Spend").alias("Avg_Spend_Per_Customer"),
            F.avg("Avg_Price").alias("Avg_Price"),
        )
        .orderBy(F.desc("Customer_Count"), "Age_Group", "Club_Member_Status")
        .limit(10)
    )

    final_df = round_existing(
        final_df,
        {
            "Avg_Transactions_Per_Customer": 2,
            "Avg_Spend_Per_Customer": 2,
            "Avg_Price": 4,
        },
    ).select(
        F.col("Age_Group").alias("Вікова_група"),
        F.col("Club_Member_Status").alias("Статус_клубного_членства"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Avg_Transactions_Per_Customer").alias("Середня_кількість_транзакцій_на_клієнта"),
        F.col("Avg_Spend_Per_Customer").alias("Середні_витрати_на_клієнта"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
    )

    save_report(
        final_df,
        "Question 1: Customer and transaction profile by age group and club member status",
        "q1_age_distribution",
        directories,
    )


# Питання: Якими є показники частоти покупок і середніх витрат у 12 найчисленніших сегментах клієнтів, сформованих за частотою отримання модних новин і віковими групами?
# Опис колонок:
# - Частота_отримання_модних_новин: Частота отримання модних новин клієнтом.
# - Вікова_група: Віковий діапазон клієнтів.
# - Кількість_клієнтів: Кількість унікальних клієнтів у сегменті.
# - Середня_кількість_транзакцій_на_клієнта: Середня кількість транзакцій на одного клієнта сегмента.
# - Середні_витрати_на_клієнта: Середня сума витрат одного клієнта сегмента.
# - Середня_ціна_товару: Середня ціна покупки в сегменті.
def _question_2(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    profile_df = _customer_transaction_profile(transactions_df, customers_df, partition_count)

    final_df = (
        profile_df
        .filter(F.col("age").isNotNull())
        .filter(F.col("Fashion_News_Frequency").isin("Regularly", "Monthly", "None"))
        .groupBy("Fashion_News_Frequency", "Age_Group")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Customer"),
            F.avg("Total_Spend").alias("Avg_Spend_Per_Customer"),
            F.avg("Avg_Price").alias("Avg_Price"),
        )
        .orderBy(F.desc("Customer_Count"), "Fashion_News_Frequency", "Age_Group")
        .limit(12)
    )

    final_df = round_existing(
        final_df,
        {
            "Avg_Transactions_Per_Customer": 2,
            "Avg_Spend_Per_Customer": 2,
            "Avg_Price": 4,
        },
    ).select(
        F.col("Fashion_News_Frequency").alias("Частота_отримання_модних_новин"),
        F.col("Age_Group").alias("Вікова_група"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Avg_Transactions_Per_Customer").alias("Середня_кількість_транзакцій_на_клієнта"),
        F.col("Avg_Spend_Per_Customer").alias("Середні_витрати_на_клієнта"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
    )

    save_report(
        final_df,
        "Question 2: Purchase frequency and spend by fashion news frequency and age group",
        "q2_purchase_frequency_by_news",
        directories,
    )


# Питання: Які 8 ключових сегментів за каналами продажу та статусами клубного членства мають найвищі показники клієнтської активності?
# Опис колонок:
# - Канал_продажу: Назва каналу продажу.
# - Статус_клубного_членства: Текстовий статус клубного членства клієнта.
# - Кількість_клієнтів: Кількість унікальних клієнтів у сегменті.
# - Середня_кількість_транзакцій_на_клієнта: Середня кількість транзакцій на одного клієнта сегмента.
# - Середні_витрати_на_клієнта: Середня сума витрат одного клієнта сегмента.
# - Частка_від_клієнтів_даного_статусу_(відсоток): Частка клієнтів цього каналу серед усіх клієнтів даного статусу.
def _question_3(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    profile_df = _customer_transaction_profile(transactions_df, customers_df, partition_count)
    segment_window = Window.partitionBy("Club_Member_Status")

    final_df = (
        profile_df
        .filter(F.col("Club_Member_Status") != "Unknown")
        .filter(F.col("Transaction_Count") > 0)
        .groupBy("Channel_Name", "Club_Member_Status")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Customer"),
            F.avg("Total_Spend").alias("Avg_Spend_Per_Customer"),
        )
        .withColumn("Status_Customer_Total", F.sum("Customer_Count").over(segment_window))
        .withColumn(
            "Customer_Share_Pct",
            F.when(
                F.col("Status_Customer_Total") > 0,
                F.col("Customer_Count") / F.col("Status_Customer_Total") * F.lit(100.0),
            ).otherwise(F.lit(0.0)),
        )
        .drop("Status_Customer_Total")
        .orderBy(F.desc("Customer_Count"), "Club_Member_Status", "Channel_Name")
        .limit(8)
    )

    final_df = round_existing(
        final_df,
        {
            "Avg_Transactions_Per_Customer": 2,
            "Avg_Spend_Per_Customer": 2,
            "Customer_Share_Pct": 2,
        },
    ).select(
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Club_Member_Status").alias("Статус_клубного_членства"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Avg_Transactions_Per_Customer").alias("Середня_кількість_транзакцій_на_клієнта"),
        F.col("Avg_Spend_Per_Customer").alias("Середні_витрати_на_клієнта"),
        F.col("Customer_Share_Pct").alias("Частка_від_клієнтів_даного_статусу_(відсоток)"),
    )

    save_report(
        final_df,
        "Question 3: Customer activity by sales channel and club member status",
        "q3_top15_postal_codes",
        directories,
    )


# Питання: Якими є характеристики 12 найчисленніших сегментів клієнтів, сформованих за квартилями витрат і віковими групами?
# Опис колонок:
# - Квартиль_витрат: Квартиль клієнтів за загальними витратами.
# - Вікова_група: Віковий діапазон клієнтів.
# - Кількість_клієнтів: Кількість унікальних клієнтів у сегменті.
# - Середній_вік: Середній вік клієнтів сегмента.
# - Середні_загальні_витрати_на_клієнта: Середня сума витрат одного клієнта сегмента.
# - Середня_кількість_транзакцій_на_клієнта: Середня кількість транзакцій на одного клієнта сегмента.
def _question_4(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    spend_window = Window.orderBy(F.desc("Total_Spend"))

    customer_totals_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Spend"),
        )
        .join(customers_df.select("customer_id", "age"), on="customer_id", how="inner")
        .filter(F.col("age").isNotNull())
        .withColumn("Age_Group", build_age_group_expression("age"))
        .withColumn("Spend_Quartile", F.ntile(4).over(spend_window))
    )

    final_df = (
        customer_totals_df
        .groupBy("Spend_Quartile", "Age_Group")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.avg("age").alias("Avg_Age"),
            F.avg("Total_Spend").alias("Avg_Total_Spend"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Customer"),
        )
        .orderBy(F.desc("Customer_Count"), "Spend_Quartile", "Age_Group")
        .limit(12)
    )

    final_df = round_existing(
        final_df,
        {
            "Avg_Age": 2,
            "Avg_Total_Spend": 2,
            "Avg_Transactions_Per_Customer": 2,
        },
    ).select(
        F.col("Spend_Quartile").alias("Квартиль_витрат"),
        F.col("Age_Group").alias("Вікова_група"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Avg_Age").alias("Середній_вік"),
        F.col("Avg_Total_Spend").alias("Середні_загальні_витрати_на_клієнта"),
        F.col("Avg_Transactions_Per_Customer").alias("Середня_кількість_транзакцій_на_клієнта"),
    )

    save_report(
        final_df,
        "Question 4: Customer profile by spending quartile and age group",
        "q4_spending_quartiles_age",
        directories,
    )


# Питання: Які 10 пріоритетних молодіжних відділів забезпечують найвищі показники виручки та найбільшу частку продажів у розрізі каналів?
# Опис колонок:
# - Назва_відділу: Текстова назва товарного відділу.
# - Канал_продажу: Назва каналу продажу.
# - Кількість_транзакцій: Загальна кількість транзакцій сегмента.
# - Загальна_виручка: Загальна виручка сегмента.
# - Середня_ціна_товару: Середня ціна покупки сегмента.
# - Частка_від_виручки_каналу_(відсоток): Частка виручки відділу серед усієї виручки цього каналу.
def _question_5(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    youth_customers_df = customers_df.filter((F.col("age") >= 16) & (F.col("age") <= 24)).select("customer_id")

    youth_transactions_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .join(F.broadcast(youth_customers_df), on="customer_id", how="inner")
        .join(
            F.broadcast(
                articles_df.select("article_id", "department_name").withColumn(
                    "Department_Name",
                    F.initcap(F.coalesce(F.trim(F.col("department_name")), F.lit("Unknown"))),
                )
            ),
            on="article_id",
            how="inner",
        )
        .withColumn("Channel_Name", build_channel_label_expression())
    )

    revenue_window = Window.partitionBy("Channel_Name")

    final_df = (
        youth_transactions_df
        .groupBy("Department_Name", "Channel_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .filter(F.col("Transaction_Count") >= 100)
        .withColumn("Channel_Revenue_Total", F.sum("Total_Revenue").over(revenue_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(
                F.col("Channel_Revenue_Total") > 0,
                F.col("Total_Revenue") / F.col("Channel_Revenue_Total") * F.lit(100.0),
            ).otherwise(F.lit(0.0)),
        )
        .drop("Channel_Revenue_Total")
        .orderBy(F.desc("Total_Revenue"), "Department_Name", "Channel_Name")
        .limit(10)
    )

    final_df = round_existing(
        final_df,
        {
            "Total_Revenue": 2,
            "Avg_Price": 4,
            "Revenue_Share_Pct": 2,
        },
    ).select(
        F.col("Department_Name").alias("Назва_відділу"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_каналу_(відсоток)"),
    )

    save_report(
        final_df,
        "Question 5: Revenue and sales share of youth departments by channel",
        "q5_top_departments_youth",
        directories,
    )


# Питання: Якими є показники залучення нових клієнтів і накопичення клієнтської бази у вибірці з 12 квартально-статусних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року, у якому клієнт здійснив першу покупку.
# - Статус_клубного_членства: Текстовий статус клубного членства клієнта.
# - Кількість_нових_клієнтів: Кількість нових клієнтів у кварталі.
# - Накопичена_кількість_нових_клієнтів: Накопичена кількість нових клієнтів у межах статусу.
# - Частка_від_нових_клієнтів_кварталу_(відсоток): Частка нових клієнтів цього статусу серед усіх нових клієнтів кварталу.
def _question_6(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    first_purchase_window = Window.partitionBy("customer_id").orderBy("t_dat", "sales_channel_id")
    cumulative_window = (
        Window.partitionBy("Club_Member_Status")
        .orderBy("Quarter_Label")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    quarter_window = Window.partitionBy("Quarter_Label")

    first_purchase_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .filter(F.year("t_dat") == 2019)
        .withColumn("Purchase_Rank", F.row_number().over(first_purchase_window))
        .filter(F.col("Purchase_Rank") == 1)
        .join(customers_df.select("customer_id", "club_member_status"), on="customer_id", how="left")
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .withColumn(
            "Club_Member_Status",
            F.initcap(F.coalesce(F.trim(F.col("club_member_status")), F.lit("Unknown"))),
        )
    )

    final_df = (
        first_purchase_df
        .groupBy("Quarter_Label", "Club_Member_Status")
        .agg(F.countDistinct("customer_id").alias("New_Customers"))
        .withColumn("Cumulative_New_Customers", F.sum("New_Customers").over(cumulative_window))
        .withColumn("Quarter_Total_Customers", F.sum("New_Customers").over(quarter_window))
        .withColumn(
            "Quarterly_Share_Pct",
            F.when(
                F.col("Quarter_Total_Customers") > 0,
                F.col("New_Customers") / F.col("Quarter_Total_Customers") * F.lit(100.0),
            ).otherwise(F.lit(0.0)),
        )
        .drop("Quarter_Total_Customers")
        .orderBy("Quarter_Label", "Club_Member_Status")
        .limit(12)
    )

    final_df = round_existing(final_df, {"Quarterly_Share_Pct": 2}).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Club_Member_Status").alias("Статус_клубного_членства"),
        F.col("New_Customers").alias("Кількість_нових_клієнтів"),
        F.col("Cumulative_New_Customers").alias("Накопичена_кількість_нових_клієнтів"),
        F.col("Quarterly_Share_Pct").alias("Частка_від_нових_клієнтів_кварталу_(відсоток)"),
    )

    save_report(
        final_df,
        "Question 6: Quarterly new customers and cumulative customer base by club member status for 2019",
        "q6_cumulative_new_customers",
        directories,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    directories = ensure_output_dirs(output_dir)
    initialize_explain_log(os.path.join(directories["logs"], "explain_logs.txt"), "Artem pipeline explain plans")

    transactions_df = load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = load_parquet_dataset(spark, processed_dir, "articles")
    customers_df = load_parquet_dataset(spark, processed_dir, "customers")
    partition_count = recommended_partitions(spark)

    _question_1(transactions_df, customers_df, directories, partition_count)
    _question_2(transactions_df, customers_df, directories, partition_count)
    _question_3(transactions_df, customers_df, directories, partition_count)
    _question_4(transactions_df, customers_df, directories, partition_count)
    _question_5(transactions_df, customers_df, articles_df, directories, partition_count)
    _question_6(transactions_df, customers_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Artem-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
