"""Taras product hierarchy and department transformation pipeline."""

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
    ensure_output_dirs,
    initialize_explain_log,
    load_parquet_dataset,
    recommended_partitions,
    round_existing,
    save_report,
)

MEMBER_NAME = "taras"


# Питання: Якими є структурні характеристики асортименту у вибірці з 10 глобальних товарних груп?
# Опис колонок:
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Кількість_унікальних_артикулів: Кількість унікальних артикулів у групі.
# - Кількість_типів_товарів: Кількість різних типів товарів у групі.
# - Кількість_секцій: Кількість різних секцій у групі.
# - Кількість_відділів: Кількість різних відділів у групі.
# - Частка_від_усіх_артикулів_(відсоток): Частка артикулів групи серед усього асортименту.
def _question_1(articles_df: DataFrame, dirs: Dict[str, str]) -> None:
    total_window = Window.partitionBy()

    final_df = (
        articles_df
        .filter(F.col("article_id").isNotNull())
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .groupBy("Index_Group_Name")
        .agg(
            F.countDistinct("article_id").alias("Distinct_Articles"),
            F.countDistinct("product_type_name").alias("Distinct_Product_Types"),
            F.countDistinct("section_name").alias("Distinct_Sections"),
            F.countDistinct("department_name").alias("Distinct_Departments"),
        )
        .withColumn("Total_Articles", F.sum("Distinct_Articles").over(total_window))
        .withColumn(
            "Article_Share_Pct",
            F.when(F.col("Total_Articles") > 0, F.col("Distinct_Articles") / F.col("Total_Articles") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Total_Articles")
        .orderBy(F.desc("Distinct_Articles"))
        .limit(10)
    )

    final_df = round_existing(final_df, {"Article_Share_Pct": 2}).select(
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Distinct_Articles").alias("Кількість_унікальних_артикулів"),
        F.col("Distinct_Product_Types").alias("Кількість_типів_товарів"),
        F.col("Distinct_Sections").alias("Кількість_секцій"),
        F.col("Distinct_Departments").alias("Кількість_відділів"),
        F.col("Article_Share_Pct").alias("Частка_від_усіх_артикулів_(відсоток)"),
    )
    save_report(final_df, "Question 1: Assortment structure by index group", "q1_unique_articles_per_index_group", dirs)


# Питання: Які 15 відділів у сегменті Ladieswear формують найвищі показники продажів за обсягом, виручкою та часткою?
# Опис колонок:
# - Назва_відділу: Назва товарного відділу.
# - Кількість_транзакцій: Загальна кількість транзакцій у відділі.
# - Загальна_виручка: Загальна виручка відділу.
# - Середня_ціна_товару: Середня ціна покупки у відділі.
# - Частка_від_виручки_Ladieswear_(відсоток): Частка виручки відділу серед усієї виручки Ladieswear.
def _question_2(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    share_window = Window.partitionBy()

    final_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .filter(F.col("price") > 0)
        .join(
            articles_df.filter(F.lower(F.col("index_name")) == "ladieswear").select("article_id", "department_name"),
            on="article_id",
            how="inner",
        )
        .withColumn("Department_Name", F.initcap(F.coalesce(F.trim(F.col("department_name")), F.lit("Unknown"))))
        .groupBy("Department_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Revenue_Total", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Revenue_Total") > 0, F.col("Total_Revenue") / F.col("Revenue_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Revenue_Total")
        .orderBy(F.desc("Total_Revenue"))
        .limit(15)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Department_Name").alias("Назва_відділу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_Ladieswear_(відсоток)"),
    )
    save_report(final_df, "Question 2: Department sales report for Ladieswear", "q2_ladieswear_department_sales", dirs)


# Питання: Які 12 сегментів, сформованих за групою одягу та відділом, мають найвищу частку преміальних моделей trousers?
# Опис колонок:
# - Група_одягу: Назва групи одягу.
# - Назва_відділу: Назва товарного відділу.
# - Кількість_преміальних_артикулів: Кількість артикулів, що потрапили до преміального сегмента.
# - Середня_максимальна_ціна: Середнє максимальне історичне значення ціни для артикулів сегмента.
# - Найвища_максимальна_ціна: Найвище зафіксоване значення максимальної ціни у сегменті.
# - Середня_кількість_транзакцій_на_артикул: Середня кількість транзакцій на один артикул сегмента.
def _question_3(
    articles_df: DataFrame,
    transactions_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    article_metrics_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .groupBy("article_id")
        .agg(F.max("price").alias("Max_Price"), F.count("*").alias("Transaction_Count"))
    )

    trousers_df = (
        articles_df
        .filter(F.lower(F.col("product_type_name")) == "trousers")
        .select("article_id", "garment_group_name", "department_name")
        .join(article_metrics_df, on="article_id", how="inner")
    )

    premium_threshold_df = trousers_df.agg(F.expr("percentile_approx(Max_Price, 0.75)").alias("Premium_Threshold"))

    final_df = (
        trousers_df
        .crossJoin(premium_threshold_df)
        .filter(F.col("Max_Price") >= F.col("Premium_Threshold"))
        .withColumn("Garment_Group_Name", F.initcap(F.coalesce(F.trim(F.col("garment_group_name")), F.lit("Unknown"))))
        .withColumn("Department_Name", F.initcap(F.coalesce(F.trim(F.col("department_name")), F.lit("Unknown"))))
        .groupBy("Garment_Group_Name", "Department_Name")
        .agg(
            F.countDistinct("article_id").alias("Premium_Article_Count"),
            F.avg("Max_Price").alias("Avg_Max_Price"),
            F.max("Max_Price").alias("Top_Max_Price"),
            F.avg("Transaction_Count").alias("Avg_Transactions_Per_Article"),
        )
        .orderBy(F.desc("Avg_Max_Price"), "Garment_Group_Name", "Department_Name")
        .limit(12)
    )

    final_df = round_existing(
        final_df,
        {"Avg_Max_Price": 4, "Top_Max_Price": 4, "Avg_Transactions_Per_Article": 2},
    ).select(
        F.col("Garment_Group_Name").alias("Група_одягу"),
        F.col("Department_Name").alias("Назва_відділу"),
        F.col("Premium_Article_Count").alias("Кількість_преміальних_артикулів"),
        F.col("Avg_Max_Price").alias("Середня_максимальна_ціна"),
        F.col("Top_Max_Price").alias("Найвища_максимальна_ціна"),
        F.col("Avg_Transactions_Per_Article").alias("Середня_кількість_транзакцій_на_артикул"),
    )
    save_report(final_df, "Question 3: Premium trousers summary by garment group and department", "q3_top3_expensive_trousers", dirs)


# Питання: Які 10 секцій у Baby/Children забезпечують найвищі показники виручки та найбільшу частку продажів?
# Опис колонок:
# - Назва_секції: Назва секції одягу.
# - Кількість_транзакцій: Загальна кількість транзакцій у секції.
# - Загальна_виручка: Загальна виручка секції.
# - Середня_ціна_товару: Середня ціна покупки у секції.
# - Частка_від_виручки_Baby_Children_(відсоток): Частка виручки секції серед усієї виручки Baby/Children.
# - Ранг_секції_за_виручкою: Позиція секції за виручкою.
def _question_4(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    revenue_window = Window.partitionBy()
    rank_window = Window.orderBy(F.desc("Total_Revenue"))

    final_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .filter(F.col("price") > 0)
        .join(
            articles_df.filter(F.lower(F.col("index_group_name")) == "baby/children").select("article_id", "section_name"),
            on="article_id",
            how="inner",
        )
        .withColumn("Section_Name", F.initcap(F.coalesce(F.trim(F.col("section_name")), F.lit("Unknown"))))
        .groupBy("Section_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Revenue_Total", F.sum("Total_Revenue").over(revenue_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Revenue_Total") > 0, F.col("Total_Revenue") / F.col("Revenue_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .withColumn("Revenue_Rank", F.dense_rank().over(rank_window))
        .drop("Revenue_Total")
        .orderBy("Revenue_Rank", "Section_Name")
        .limit(10)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Section_Name").alias("Назва_секції"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_Baby_Children_(відсоток)"),
        F.col("Revenue_Rank").alias("Ранг_секції_за_виручкою"),
    )
    save_report(final_df, "Question 4: Revenue share of Baby and Children sections", "q4_baby_children_section_share", dirs)


# Питання: Які 10 глобальних товарних груп демонструють найвищі показники продажів для категорії sweater?
# Опис колонок:
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Кількість_транзакцій: Загальна кількість транзакцій у групі.
# - Загальна_виручка: Загальна виручка групи.
# - Середня_ціна_товару: Середня ціна покупки у групі.
# - Частка_від_виручки_sweater_(відсоток): Частка виручки групи серед усієї виручки sweater.
def _question_5(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    share_window = Window.partitionBy()

    final_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .filter(F.col("price") > 0)
        .join(
            articles_df.filter(F.lower(F.col("product_type_name")) == "sweater").select("article_id", "index_group_name"),
            on="article_id",
            how="inner",
        )
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .groupBy("Index_Group_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Revenue_Total", F.sum("Total_Revenue").over(share_window))
        .withColumn(
            "Revenue_Share_Pct",
            F.when(F.col("Revenue_Total") > 0, F.col("Total_Revenue") / F.col("Revenue_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Revenue_Total")
        .orderBy(F.desc("Total_Revenue"))
        .limit(10)
    )

    final_df = round_existing(final_df, {"Total_Revenue": 2, "Avg_Price": 4, "Revenue_Share_Pct": 2}).select(
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Revenue_Share_Pct").alias("Частка_від_виручки_sweater_(відсоток)"),
    )
    save_report(final_df, "Question 5: Sweater sales and revenue share by index group", "q5_sweater_sales_by_index_group", dirs)


# Питання: Якими є зміни рангів за виручкою у вибірці з 20 квартально-відділових сегментів провідних відділів?
# Опис колонок:
# - Квартал: Квартал спостереження.
# - Назва_відділу: Назва товарного відділу.
# - Кількість_транзакцій: Загальна кількість транзакцій відділу у кварталі.
# - Квартальна_виручка: Загальна виручка відділу у кварталі.
# - Ранг_відділу_за_виручкою: Позиція відділу за виручкою у кварталі.
# - Ранг_відділу_у_попередньому_кварталі: Позиція відділу за виручкою у попередньому кварталі.
# - Зміна_рангу_порівняно_з_попереднім_кварталом: Різниця між попереднім та поточним рангом.
def _question_6(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    dirs: Dict[str, str],
    partitions: int,
) -> None:
    total_revenue_window = Window.orderBy(F.desc("Total_Revenue_All"))
    rank_window = Window.partitionBy("Quarter_Label").orderBy(F.desc("Quarter_Revenue"))
    lag_window = Window.partitionBy("Department_Name").orderBy("Quarter_Label")

    top_departments_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "department_name"), on="article_id", how="inner")
        .withColumn("Department_Name", F.initcap(F.coalesce(F.trim(F.col("department_name")), F.lit("Unknown"))))
        .groupBy("Department_Name")
        .agg(F.sum("price").alias("Total_Revenue_All"))
        .withColumn("Department_Global_Rank", F.dense_rank().over(total_revenue_window))
        .filter(F.col("Department_Global_Rank") <= 5)
        .select("Department_Name")
    )

    final_df = (
        transactions_df
        .repartition(partitions, "article_id")
        .filter(F.year("t_dat") == 2019)
        .filter(F.col("price") > 0)
        .join(articles_df.select("article_id", "department_name"), on="article_id", how="inner")
        .withColumn("Department_Name", F.initcap(F.coalesce(F.trim(F.col("department_name")), F.lit("Unknown"))))
        .join(F.broadcast(top_departments_df), on="Department_Name", how="inner")
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .groupBy("Quarter_Label", "Department_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Quarter_Revenue"),
        )
        .withColumn("Department_Rank", F.dense_rank().over(rank_window))
        .withColumn("Prev_Department_Rank", F.lag("Department_Rank").over(lag_window))
        .withColumn(
            "Rank_Change",
            F.when(F.col("Prev_Department_Rank").isNotNull(), F.col("Prev_Department_Rank") - F.col("Department_Rank")),
        )
        .orderBy("Quarter_Label", "Department_Rank")
        .limit(20)
    )

    final_df = round_existing(final_df, {"Quarter_Revenue": 2}).select(
        F.col("Quarter_Label").alias("Квартал"),
        F.col("Department_Name").alias("Назва_відділу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Quarter_Revenue").alias("Квартальна_виручка"),
        F.col("Department_Rank").alias("Ранг_відділу_за_виручкою"),
        F.col("Prev_Department_Rank").alias("Ранг_відділу_у_попередньому_кварталі"),
        F.col("Rank_Change").alias("Зміна_рангу_порівняно_з_попереднім_кварталом"),
    )
    save_report(final_df, "Question 6: Quarterly department rank changes by revenue", "q6_department_rank_bump", dirs)


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_dir: str = f"output/{MEMBER_NAME}",
) -> None:
    dirs = ensure_output_dirs(output_dir)
    initialize_explain_log(os.path.join(dirs["logs"], "explain_logs.txt"), "Taras pipeline explain plans")

    articles_df = load_parquet_dataset(spark, processed_dir, "articles")
    transactions_df = load_parquet_dataset(spark, processed_dir, "transactions")
    partitions = recommended_partitions(spark)

    _question_1(articles_df, dirs)
    _question_2(transactions_df, articles_df, dirs, partitions)
    _question_3(articles_df, transactions_df, dirs, partitions)
    _question_4(transactions_df, articles_df, dirs, partitions)
    _question_5(transactions_df, articles_df, dirs, partitions)
    _question_6(transactions_df, articles_df, dirs, partitions)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Taras-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
