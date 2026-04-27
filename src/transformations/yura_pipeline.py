"""Yura visual trends and design features transformation pipeline."""

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
    build_channel_label_expression,
    ensure_output_dirs,
    initialize_explain_log,
    load_parquet_dataset,
    recommended_partitions,
    round_existing,
    save_report,
)

MEMBER_NAME = "yura"


# Питання: Яким є розподіл асортименту у вибірці з 15 сегментів, сформованих за інтенсивністю кольору та глобальною товарною групою?
# Опис колонок:
# - Інтенсивність_кольору: Назва інтенсивності кольору.
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Кількість_артикулів: Кількість унікальних артикулів у сегменті.
# - Кількість_секцій: Кількість різних секцій у сегменті.
# - Частка_від_артикулів_цієї_інтенсивності_кольору_(відсоток): Частка артикулів сегмента серед усіх артикулів цієї інтенсивності кольору.
def _question_1(articles_df: DataFrame, directories: Dict[str, str], partition_count: int) -> None:
    share_window = Window.partitionBy("Perceived_Colour_Value_Name")

    final_df = (
        articles_df
        .repartition(partition_count, "perceived_colour_value_name")
        .filter(F.col("article_id").isNotNull())
        .withColumn("Perceived_Colour_Value_Name", F.initcap(F.coalesce(F.trim(F.col("perceived_colour_value_name")), F.lit("Unknown"))))
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .groupBy("Perceived_Colour_Value_Name", "Index_Group_Name")
        .agg(
            F.countDistinct("article_id").alias("Article_Count"),
            F.countDistinct("section_name").alias("Distinct_Sections"),
        )
        .withColumn("Colour_Value_Total", F.sum("Article_Count").over(share_window))
        .withColumn(
            "Article_Share_Pct",
            F.when(F.col("Colour_Value_Total") > 0, F.col("Article_Count") / F.col("Colour_Value_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Colour_Value_Total")
        .orderBy(F.desc("Article_Count"), "Perceived_Colour_Value_Name")
        .limit(15)
    )

    final_df = round_existing(final_df, {"Article_Share_Pct": 2}).select(
        F.col("Perceived_Colour_Value_Name").alias("Інтенсивність_кольору"),
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Article_Count").alias("Кількість_артикулів"),
        F.col("Distinct_Sections").alias("Кількість_секцій"),
        F.col("Article_Share_Pct").alias("Частка_від_артикулів_цієї_інтенсивності_кольору_(відсоток)"),
    )
    save_report(final_df, "Question 1: Assortment distribution by colour value and index group", "q1_assortment_by_colour_value", directories)


# Питання: Які 12 сегментів, сформованих за кольоровою групою та каналом, забезпечують найвищі показники продажів і виручки смугастих товарів?
# Опис колонок:
# - Кольорова_група: Назва кольорової групи.
# - Канал_продажу: Назва каналу продажу.
# - Кількість_транзакцій: Загальна кількість транзакцій сегмента.
# - Загальна_виручка: Загальна виручка сегмента.
# - Середня_ціна_товару: Середня ціна покупки сегмента.
# - Частка_від_виручки_каналу_(відсоток): Частка виручки сегмента серед виручки відповідного каналу.
def _question_2(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    share_window = Window.partitionBy("Channel_Name")

    stripe_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("graphical_appearance_name"))) == "stripe")
        .select("article_id", "colour_group_name")
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.col("price") > 0)
        .join(F.broadcast(stripe_articles), on="article_id", how="inner")
        .withColumn("Channel_Name", build_channel_label_expression())
        .withColumn("Colour_Group_Name", F.initcap(F.coalesce(F.trim(F.col("colour_group_name")), F.lit("Unknown"))))
        .groupBy("Colour_Group_Name", "Channel_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .filter(F.col("Transaction_Count") >= 50)
        .withColumn("Channel_Revenue_Total", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Channel_Revenue_Total") > 0, F.col("Total_Revenue") / F.col("Channel_Revenue_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Channel_Revenue_Total")
        .orderBy(F.desc("Total_Revenue"), "Channel_Name")
        .limit(12)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Colour_Group_Name").alias("Кольорова_група"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_каналу_(відсоток)"),
    )
    save_report(final_df, "Question 2: Stripe-item colour performance by channel", "q2_top10_stripe_colours", directories)


# Питання: Якими є показники продажів чорних товарів у вибірці з 8 квартально-канальних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Канал_продажу: Назва каналу продажу.
# - Кількість_транзакцій: Загальна кількість транзакцій чорних товарів у кварталі.
# - Загальна_виручка: Загальна виручка чорних товарів у кварталі.
# - Середня_кількість_транзакцій_за_день: Середня денна кількість транзакцій чорних товарів у кварталі.
# - Середня_кількість_транзакцій_за_останні_3_квартали: Ковзне середнє квартальної кількості транзакцій.
def _question_3(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    rolling_window = Window.partitionBy("Channel_Name").orderBy("Quarter_Label").rowsBetween(-2, 0)

    black_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("colour_group_name"))) == "black")
        .select("article_id")
        .dropDuplicates(["article_id"])
    )

    daily_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.col("price") > 0)
        .filter(F.year("t_dat") == 2019)
        .join(F.broadcast(black_articles), on="article_id", how="inner")
        .withColumn("Channel_Name", build_channel_label_expression())
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .groupBy("Quarter_Label", "Channel_Name", "t_dat")
        .agg(
            F.count("*").alias("Daily_Transaction_Count"),
            F.sum("price").alias("Daily_Revenue"),
        )
    )

    final_df = (
        daily_df
        .groupBy("Quarter_Label", "Channel_Name")
        .agg(
            F.sum("Daily_Transaction_Count").alias("Transaction_Count"),
            F.sum("Daily_Revenue").alias("Total_Revenue"),
            F.avg("Daily_Transaction_Count").alias("Avg_Daily_Transaction_Count"),
        )
        .withColumn("Rolling_3_Quarter_Avg_Transactions", F.avg("Transaction_Count").over(rolling_window))
        .orderBy("Quarter_Label", "Channel_Name")
        .limit(8)
    )

    final_df = round_existing(
        final_df,
        {
            "Total_Revenue": 2,
            "Avg_Daily_Transaction_Count": 2,
            "Rolling_3_Quarter_Avg_Transactions": 2,
        },
    ).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Daily_Transaction_Count").alias("Середня_кількість_транзакцій_за_день"),
        F.col("Rolling_3_Quarter_Avg_Transactions").alias("Середня_кількість_транзакцій_за_останні_3_квартали"),
    )
    save_report(final_df, "Question 3: Quarterly black-item sales trend by channel", "q3_black_sales_rolling_30d", directories)


# Питання: Які 10 сегментів, сформованих за графічним візерунком і каналом продажу, демонструють найвищі результати?
# Опис колонок:
# - Канал_продажу: Назва каналу продажу.
# - Графічний_візерунок: Назва графічного візерунку.
# - Кількість_транзакцій: Загальна кількість транзакцій сегмента.
# - Загальна_виручка: Загальна виручка сегмента.
# - Середня_ціна_товару: Середня ціна покупки сегмента.
# - Ранг_візерунку_за_виручкою_в_каналі: Позиція візерунку за виручкою в межах каналу.
def _question_4(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    rank_window = Window.partitionBy("Channel_Name").orderBy(F.desc("Total_Revenue"), F.desc("Transaction_Count"))

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "graphical_appearance_name"), on="article_id", how="inner")
        .withColumn("Channel_Name", build_channel_label_expression())
        .withColumn("Graphical_Appearance_Name", F.initcap(F.coalesce(F.trim(F.col("graphical_appearance_name")), F.lit("Unknown"))))
        .groupBy("Channel_Name", "Graphical_Appearance_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .filter(F.col("Transaction_Count") >= 100)
        .withColumn("Pattern_Rank", F.dense_rank().over(rank_window))
        .filter(F.col("Pattern_Rank") <= 5)
        .orderBy("Channel_Name", "Pattern_Rank", "Graphical_Appearance_Name")
        .limit(10)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4}).select(
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Graphical_Appearance_Name").alias("Графічний_візерунок"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Pattern_Rank").alias("Ранг_візерунку_за_виручкою_в_каналі"),
    )
    save_report(final_df, "Question 4: Top graphical appearances by sales channel", "q4_top_pattern_by_channel", directories)


# Питання: Які 12 сегментів, сформованих за флоральною товарною групою та каналом, генерують найвищі показники виручки?
# Опис колонок:
# - Канал_продажу: Назва каналу продажу.
# - Товарна_група: Назва товарної групи.
# - Кількість_транзакцій: Загальна кількість транзакцій сегмента.
# - Загальна_виручка: Загальна виручка сегмента.
# - Середня_ціна_товару: Середня ціна покупки сегмента.
# - Частка_від_виручки_каналу_(відсоток): Частка виручки сегмента серед виручки каналу.
def _question_5(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    share_window = Window.partitionBy("Channel_Name")
    floral_mask = (
        F.lower(F.coalesce(F.col("graphical_appearance_name"), F.lit(""))).contains("floral")
        | F.lower(F.coalesce(F.col("prod_name"), F.lit(""))).contains("flower")
        | F.lower(F.coalesce(F.col("detail_desc"), F.lit(""))).contains("floral")
    )

    floral_articles = articles_df.filter(floral_mask).select("article_id", "product_group_name")

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.col("price") > 0)
        .join(F.broadcast(floral_articles), on="article_id", how="inner")
        .withColumn("Channel_Name", build_channel_label_expression())
        .withColumn("Product_Group_Name", F.initcap(F.coalesce(F.trim(F.col("product_group_name")), F.lit("Unknown"))))
        .groupBy("Channel_Name", "Product_Group_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Channel_Revenue_Total", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Channel_Revenue_Total") > 0, F.col("Total_Revenue") / F.col("Channel_Revenue_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Channel_Revenue_Total")
        .orderBy(F.desc("Total_Revenue"), "Channel_Name")
        .limit(12)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Product_Group_Name").alias("Товарна_група"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_каналу_(відсоток)"),
    )
    save_report(final_df, "Question 5: Floral-item revenue by channel and product group", "q5_floral_avg_check_by_channel", directories)


# Питання: Якими є зміни виручки червоних товарів у вибірці з 20 квартально-товарних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Товарна_група: Назва товарної групи.
# - Квартальна_виручка: Загальна виручка товарної групи у кварталі.
# - Виручка_попереднього_кварталу: Виручка цієї товарної групи у попередньому кварталі.
# - Темп_зростання_до_попереднього_кварталу_(відсоток): Темп зміни виручки відносно попереднього кварталу.
# - Частка_від_виручки_кварталу_(відсоток): Частка виручки товарної групи серед усієї виручки червоних товарів у кварталі.
def _question_6(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    directories: Dict[str, str],
    partition_count: int,
) -> None:
    total_revenue_window = Window.orderBy(F.desc("Revenue_2019"))
    growth_window = Window.partitionBy("Product_Group_Name").orderBy("Quarter_Label")
    share_window = Window.partitionBy("Quarter_Label")

    red_articles = (
        articles_df
        .filter(F.lower(F.trim(F.col("perceived_colour_master_name"))) == "red")
        .select("article_id", "product_group_name")
    )

    top_groups_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(F.broadcast(red_articles), on="article_id", how="inner")
        .withColumn("Product_Group_Name", F.initcap(F.coalesce(F.trim(F.col("product_group_name")), F.lit("Unknown"))))
        .groupBy("Product_Group_Name")
        .agg(F.sum("price").alias("Revenue_2019"))
        .withColumn("Revenue_Rank", F.dense_rank().over(total_revenue_window))
        .filter(F.col("Revenue_Rank") <= 5)
        .select("Product_Group_Name")
    )

    final_df = (
        transactions_df
        .repartition(partition_count, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(F.broadcast(red_articles), on="article_id", how="inner")
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .withColumn("Product_Group_Name", F.initcap(F.coalesce(F.trim(F.col("product_group_name")), F.lit("Unknown"))))
        .join(F.broadcast(top_groups_df), on="Product_Group_Name", how="inner")
        .groupBy("Quarter_Label", "Product_Group_Name")
        .agg(F.sum("price").alias("Quarter_Revenue"))
        .withColumn("Previous_Quarter_Revenue", F.lag("Quarter_Revenue").over(growth_window))
        .withColumn(
            "Growth_Rate_Pct",
            F.when(
                F.col("Previous_Quarter_Revenue") > 0,
                (F.col("Quarter_Revenue") - F.col("Previous_Quarter_Revenue")) / F.col("Previous_Quarter_Revenue") * 100.0,
            ),
        )
        .withColumn("Quarter_Total_Revenue", F.sum("Quarter_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Quarter_Total_Revenue") > 0, F.col("Quarter_Revenue") / F.col("Quarter_Total_Revenue") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Quarter_Total_Revenue")
        .orderBy("Quarter_Label", F.desc("Quarter_Revenue"))
        .limit(20)
    )

    final_df = round_existing(
        final_df,
        {
            "Quarter_Revenue": 2,
            "Previous_Quarter_Revenue": 2,
            "Growth_Rate_Pct": 2,
            "Revenue_Share_Pct": 2,
        },
    ).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Product_Group_Name").alias("Товарна_група"),
        F.col("Quarter_Revenue").alias("Квартальна_виручка"),
        F.col("Previous_Quarter_Revenue").alias("Виручка_попереднього_кварталу"),
        F.col("Growth_Rate_Pct").alias("Темп_зростання_до_попереднього_кварталу_(відсоток)"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_кварталу_(відсоток)"),
    )
    save_report(final_df, "Question 6: Quarterly revenue growth of red items by product group", "q6_red_mom_growth", directories)


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    directories = ensure_output_dirs(output_dir)
    initialize_explain_log(os.path.join(directories["logs"], "explain_logs.txt"), "Yura pipeline explain plans")

    transactions_df = load_parquet_dataset(spark, processed_dir, "transactions")
    articles_df = load_parquet_dataset(spark, processed_dir, "articles")
    partition_count = recommended_partitions(spark)

    _question_1(articles_df, directories, partition_count)
    _question_2(transactions_df, articles_df, directories, partition_count)
    _question_3(transactions_df, articles_df, directories, partition_count)
    _question_4(transactions_df, articles_df, directories, partition_count)
    _question_5(transactions_df, articles_df, directories, partition_count)
    _question_6(transactions_df, articles_df, directories, partition_count)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Yura-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
