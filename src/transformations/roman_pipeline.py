"""Roman financial performance transformation pipeline."""

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
    build_weekday_label_expression,
    ensure_output_dirs,
    initialize_explain_log,
    load_parquet_dataset,
    recommended_partitions,
    round_existing,
    save_report,
)

MEMBER_NAME = "roman"


# Питання: Які 10 сегментів за глобальною товарною групою та каналом продажу забезпечують найвищі показники виручки, середньої ціни та частки каналу?
# Опис колонок:
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Канал_продажу: Назва каналу продажу.
# - Кількість_транзакцій: Загальна кількість транзакцій у сегменті.
# - Загальна_виручка: Загальна виручка у сегменті.
# - Середня_ціна_товару: Середня ціна покупки у сегменті.
# - Частка_від_виручки_глобальної_групи_(відсоток): Частка виручки каналу в межах даної глобальної товарної групи.
def _question_1(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    share_window = Window.partitionBy("Index_Group_Name")

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "index_group_name"), on="article_id", how="inner")
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .withColumn("Channel_Name", build_channel_label_expression())
        .groupBy("Index_Group_Name", "Channel_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Index_Group_Revenue", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(
                F.col("Index_Group_Revenue") > 0,
                F.col("Total_Revenue") / F.col("Index_Group_Revenue") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .drop("Index_Group_Revenue")
        .orderBy(F.desc("Total_Revenue"), "Index_Group_Name", "Channel_Name")
        .limit(10)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_глобальної_групи_(відсоток)"),
    )
    save_report(
        final_df,
        "Question 1: Revenue, average price, and channel share by index group",
        "q1_revenue_by_index_group_channel",
        directories,
    )


# Питання: Яким є фінансовий профіль у вибірці з 14 сегментів, сформованих за днями тижня та каналами продажу?
# Опис колонок:
# - Канал_продажу: Назва каналу продажу.
# - День_тижня: Назва дня тижня.
# - Кількість_транзакцій: Загальна кількість транзакцій у цей день тижня.
# - Загальна_виручка: Загальна виручка у цей день тижня.
# - Середня_ціна_товару: Середня ціна покупки у цей день тижня.
# - Ранг_дня_за_виручкою_в_каналі: Позиція дня тижня за виручкою в межах каналу.
def _question_2(
    transactions_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    ranking_window = Window.partitionBy("Channel_Name").orderBy(F.desc("Total_Revenue"), F.asc("Weekday_Num"))

    final_df = (
        transactions_df
        .repartition(partition_count, "t_dat")
        .filter(F.col("t_dat").isNotNull())
        .filter(F.col("price") > 0.01)
        .withColumn("Weekday_Num", F.dayofweek("t_dat"))
        .withColumn("Weekday_Name", build_weekday_label_expression("Weekday_Num"))
        .withColumn("Channel_Name", build_channel_label_expression())
        .groupBy("Weekday_Num", "Weekday_Name", "Channel_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Revenue_Rank", F.dense_rank().over(ranking_window))
        .orderBy("Channel_Name", "Revenue_Rank", "Weekday_Name")
        .limit(14)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4}).select(
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Weekday_Name").alias("День_тижня"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Rank").alias("Ранг_дня_за_виручкою_в_каналі"),
    )
    save_report(
        final_df,
        "Question 2: Financial profile of weekdays by sales channel",
        "q2_weekday_avg_revenue_rank",
        directories,
    )


# Питання: Які 15 комбінацій товарних груп і секцій формують найвищі показники виручки у 2019 році?
# Опис колонок:
# - Товарна_група: Назва товарної групи.
# - Секція: Назва секції одягу.
# - Кількість_транзакцій: Загальна кількість транзакцій у сегменті.
# - Загальна_виручка: Загальна виручка сегмента у 2019 році.
# - Середня_ціна_товару: Середня ціна покупки у сегменті.
# - Частка_від_виручки_товарної_групи_(відсоток): Частка виручки секції в межах товарної групи.
def _question_3(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    share_window = Window.partitionBy("Product_Group_Name")

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(
            articles_df.select("article_id", "product_group_name", "section_name"),
            on="article_id",
            how="inner",
        )
        .withColumn("Product_Group_Name", F.initcap(F.coalesce(F.trim(F.col("product_group_name")), F.lit("Unknown"))))
        .withColumn("Section_Name", F.initcap(F.coalesce(F.trim(F.col("section_name")), F.lit("Unknown"))))
        .groupBy("Product_Group_Name", "Section_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .filter(F.col("Transaction_Count") >= 100)
        .withColumn("Product_Group_Revenue", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(
                F.col("Product_Group_Revenue") > 0,
                F.col("Total_Revenue") / F.col("Product_Group_Revenue") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .drop("Product_Group_Revenue")
        .orderBy(F.desc("Total_Revenue"), "Product_Group_Name", "Section_Name")
        .limit(15)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Product_Group_Name").alias("Товарна_група"),
        F.col("Section_Name").alias("Секція"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_товарної_групи_(відсоток)"),
    )
    save_report(
        final_df,
        "Question 3: Revenue report by product group and section for 2019",
        "q3_top10_revenue_share_2019",
        directories,
    )


# Питання: Якими є зміни виручки та темпи її зростання у вибірці з 20 квартально-категорійних сегментів провідних глобальних товарних груп у 2019 році?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Кількість_транзакцій: Загальна кількість транзакцій у кварталі.
# - Квартальна_виручка: Загальна виручка групи у кварталі.
# - Виручка_попереднього_кварталу: Виручка цієї групи у попередньому кварталі.
# - Темп_зростання_до_попереднього_кварталу_(відсоток): Темп зміни виручки відносно попереднього кварталу.
def _question_4(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    total_revenue_window = Window.orderBy(F.desc("Revenue_2019"))
    quarter_window = Window.partitionBy("Index_Group_Name").orderBy("Quarter_Label")

    top_groups_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "index_group_name"), on="article_id", how="inner")
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .groupBy("Index_Group_Name")
        .agg(F.sum("price").alias("Revenue_2019"))
        .withColumn("Revenue_Rank", F.dense_rank().over(total_revenue_window))
        .filter(F.col("Revenue_Rank") <= 5)
        .select("Index_Group_Name")
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "index_group_name"), on="article_id", how="inner")
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .join(F.broadcast(top_groups_df), on="Index_Group_Name", how="inner")
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .groupBy("Quarter_Label", "Index_Group_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Quarter_Revenue"),
        )
        .withColumn("Previous_Quarter_Revenue", F.lag("Quarter_Revenue").over(quarter_window))
        .withColumn(
            "Growth_Rate_Pct",
            F.when(
                F.col("Previous_Quarter_Revenue") > 0,
                (F.col("Quarter_Revenue") - F.col("Previous_Quarter_Revenue")) / F.col("Previous_Quarter_Revenue") * 100.0,
            ),
        )
        .orderBy("Quarter_Label", F.desc("Quarter_Revenue"))
        .limit(20)
    )

    final_df = round_existing(final_df, {"Quarter_Revenue": 2, "Previous_Quarter_Revenue": 2, "Growth_Rate_Pct": 2}).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Quarter_Revenue").alias("Квартальна_виручка"),
        F.col("Previous_Quarter_Revenue").alias("Виручка_попереднього_кварталу"),
        F.col("Growth_Rate_Pct").alias("Темп_зростання_до_попереднього_кварталу_(відсоток)"),
    )
    save_report(
        final_df,
        "Question 4: Quarterly revenue growth by top index groups in 2019",
        "q4_monthly_growth_rate",
        directories,
    )


# Питання: Якими є показники середнього чека, частоти покупок і витрат у 15 найчисленніших сегментах, сформованих за підпискою на модні новини, каналом та віком?
# Опис колонок:
# - Частота_отримання_модних_новин: Частота отримання модних новин клієнтом.
# - Канал_продажу: Назва каналу продажу.
# - Вікова_група: Віковий діапазон клієнтів.
# - Кількість_клієнтів: Кількість унікальних клієнтів у сегменті.
# - Кількість_транзакцій: Загальна кількість транзакцій у сегменті.
# - Середня_ціна_товару: Середня ціна покупки у сегменті.
# - Середня_кількість_транзакцій_на_клієнта: Середня кількість транзакцій на клієнта сегмента.
# - Середні_витрати_на_клієнта: Середні витрати одного клієнта сегмента.
def _question_5(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    customer_channel_df = (
        transactions_df
        .repartition(partition_count, "customer_id")
        .filter(F.col("price") > 0)
        .groupBy("customer_id", "sales_channel_id")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Spend"),
            F.avg("price").alias("Avg_Price"),
        )
        .join(customers_df.select("customer_id", "fashion_news_frequency", "age"), on="customer_id", how="inner")
        .withColumn("Channel_Name", build_channel_label_expression())
        .withColumn("Age_Group", build_age_group_expression("age"))
        .withColumn(
            "Fashion_News_Frequency",
            F.initcap(F.coalesce(F.trim(F.col("fashion_news_frequency")), F.lit("Unknown"))),
        )
    )

    final_df = (
        customer_channel_df
        .filter(F.col("Fashion_News_Frequency").isin("Regularly", "Monthly", "None"))
        .groupBy("Fashion_News_Frequency", "Channel_Name", "Age_Group")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.sum("Transaction_Count").alias("Transaction_Count"),
            F.avg("Avg_Price").alias("Avg_Price"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Customer"),
            F.avg("Total_Spend").alias("Avg_Spend_Per_Customer"),
        )
        .orderBy(F.desc("Customer_Count"), "Fashion_News_Frequency", "Channel_Name", "Age_Group")
        .limit(15)
    )

    final_df = round_existing(
        final_df,
        {
            "Avg_Price": 4,
            "Avg_Transactions_Per_Customer": 2,
            "Avg_Spend_Per_Customer": 2,
        },
    ).select(
        F.col("Fashion_News_Frequency").alias("Частота_отримання_модних_новин"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Age_Group").alias("Вікова_група"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Avg_Transactions_Per_Customer").alias("Середня_кількість_транзакцій_на_клієнта"),
        F.col("Avg_Spend_Per_Customer").alias("Середні_витрати_на_клієнта"),
    )
    save_report(
        final_df,
        "Question 5: Customer spend profile by fashion news frequency, channel, and age group",
        "q5_regular_fashion_news_avg_check",
        directories,
    )


# Питання: Якими є показники квартальної та накопиченої виручки у вибірці з 8 квартально-канальних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Канал_продажу: Назва каналу продажу.
# - Квартальна_виручка: Загальна виручка каналу у кварталі.
# - Накопичена_виручка_каналу: Накопичена виручка каналу від початку року.
# - Частка_від_виручки_кварталу_(відсоток): Частка виручки каналу серед усієї виручки кварталу.
def _question_6(
    transactions_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    cumulative_window = (
        Window.partitionBy("Channel_Name")
        .orderBy("Quarter_Label")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    quarter_share_window = Window.partitionBy("Quarter_Label")

    final_df = (
        transactions_df
        .repartition(partition_count, "t_dat")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .withColumn("Channel_Name", build_channel_label_expression())
        .groupBy("Quarter_Label", "Channel_Name")
        .agg(F.sum("price").alias("Quarter_Revenue"))
        .withColumn("Cumulative_Revenue", F.sum("Quarter_Revenue").over(cumulative_window))
        .withColumn("Quarter_Total_Revenue", F.sum("Quarter_Revenue").over(quarter_share_window))
        .withColumn(
            "Quarter_Share_Pct",
            F.when(
                F.col("Quarter_Total_Revenue") > 0,
                F.col("Quarter_Revenue") / F.col("Quarter_Total_Revenue") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .drop("Quarter_Total_Revenue")
        .orderBy("Quarter_Label", "Channel_Name")
        .limit(8)
    )

    final_df = round_existing(final_df, {"Quarter_Revenue": 2, "Cumulative_Revenue": 2, "Quarter_Share_Pct": 2}).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Quarter_Revenue").alias("Квартальна_виручка"),
        F.col("Cumulative_Revenue").alias("Накопичена_виручка_каналу"),
        F.col("Quarter_Share_Pct").alias("Частка_від_виручки_кварталу_(відсоток)"),
    )
    save_report(
        final_df,
        "Question 6: Quarterly and cumulative revenue by sales channel in 2019",
        "q6_running_total_2019_by_channel",
        directories,
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    directories = ensure_output_dirs(output_dir)
    initialize_explain_log(os.path.join(directories["logs"], "explain_logs.txt"), "Roman pipeline explain plans")

    transactions_df = load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = load_parquet_dataset(spark, processed_dir, "articles")
    customers_df = load_parquet_dataset(spark, processed_dir, "customers")
    partition_count = recommended_partitions(spark)

    _question_1(transactions_df, articles_df, directories, partition_count)
    _question_2(transactions_df, directories, partition_count)
    _question_3(transactions_df, articles_df, directories, partition_count)
    _question_4(transactions_df, articles_df, directories, partition_count)
    _question_5(transactions_df, customers_df, directories, partition_count)
    _question_6(transactions_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Roman-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
