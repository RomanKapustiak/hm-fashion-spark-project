"""Dmytro transformation pipeline (time-series dynamics and retention reports)."""

from __future__ import annotations

import os
import sys

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
    round_existing,
    save_report,
)


# Питання: Якими є показники динаміки транзакцій і виручки у вибірці з 8 квартально-канальних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Канал_продажу: Назва каналу продажу.
# - Кількість_транзакцій: Загальна кількість транзакцій у кварталі.
# - Загальна_виручка: Загальна виручка у кварталі.
# - Середня_ціна_товару: Середня ціна покупки у кварталі.
# - Кількість_транзакцій_попереднього_кварталу: Кількість транзакцій цього каналу у попередньому кварталі.
# - Виручка_попереднього_кварталу: Виручка цього каналу у попередньому кварталі.
# - Зміна_кількості_транзакцій_до_попереднього_кварталу_(відсоток): Темп зміни кількості транзакцій відносно попереднього кварталу.
# - Зміна_виручки_до_попереднього_кварталу_(відсоток): Темп зміни виручки відносно попереднього кварталу.
def query_1_weekly_transaction_dynamics(transactions_df: DataFrame) -> DataFrame:
    """Quarterly transaction and revenue report by channel for 2019."""
    quarter_window = Window.partitionBy("Channel_Name").orderBy("Quarter_Label")

    result_df = (
        transactions_df
        .filter(F.col("t_dat").isNotNull())
        .filter(F.col("price") > 0)
        .filter(F.year("t_dat") == 2019)
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .withColumn("Channel_Name", build_channel_label_expression())
        .groupBy("Quarter_Label", "Channel_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
            F.avg("price").alias("Avg_Price"),
        )
        .withColumn("Prev_Quarter_Transactions", F.lag("Transaction_Count").over(quarter_window))
        .withColumn("Prev_Quarter_Revenue", F.lag("Total_Revenue").over(quarter_window))
        .withColumn(
            "QoQ_Transaction_Change_Pct",
            F.when(
                F.col("Prev_Quarter_Transactions") > 0,
                (F.col("Transaction_Count") - F.col("Prev_Quarter_Transactions")) / F.col("Prev_Quarter_Transactions") * 100.0,
            ),
        )
        .withColumn(
            "QoQ_Revenue_Change_Pct",
            F.when(
                F.col("Prev_Quarter_Revenue") > 0,
                (F.col("Total_Revenue") - F.col("Prev_Quarter_Revenue")) / F.col("Prev_Quarter_Revenue") * 100.0,
            ),
        )
        .orderBy("Quarter_Label", "Channel_Name")
        .limit(8)
    )
    return round_existing(
        result_df,
        {
            "Total_Revenue": 2,
            "Avg_Price": 4,
            "Prev_Quarter_Revenue": 2,
            "QoQ_Transaction_Change_Pct": 2,
            "QoQ_Revenue_Change_Pct": 2,
        },
    ).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Total_Revenue").alias("Загальна_виручка"),
        F.col("Avg_Price").alias("Середня_ціна_товару"),
        F.col("Prev_Quarter_Transactions").alias("Кількість_транзакцій_попереднього_кварталу"),
        F.col("Prev_Quarter_Revenue").alias("Виручка_попереднього_кварталу"),
        F.col("QoQ_Transaction_Change_Pct").alias("Зміна_кількості_транзакцій_до_попереднього_кварталу_(відсоток)"),
        F.col("QoQ_Revenue_Change_Pct").alias("Зміна_виручки_до_попереднього_кварталу_(відсоток)"),
    )


# Питання: Які 15 комбінацій товарних і глобальних товарних груп характеризуються найбільшою різницею між зимовою та літньою виручкою?
# Опис колонок:
# - Товарна_група: Назва товарної групи.
# - Глобальна_товарна_група: Назва глобальної товарної групи.
# - Кількість_транзакцій_взимку: Кількість транзакцій узимку.
# - Кількість_транзакцій_влітку: Кількість транзакцій улітку.
# - Виручка_взимку: Загальна виручка узимку.
# - Виручка_влітку: Загальна виручка влітку.
# - Різниця_між_виручкою_взимку_та_влітку: Абсолютна різниця між зимовою та літньою виручкою.
# - Частка_зимової_виручки_від_сумарної_сезонної_виручки_(відсоток): Частка зимової виручки від суми зимової та літньої виручки.
def query_2_winter_vs_summer_spike(
    transactions_df: DataFrame,
    articles_df: DataFrame,
) -> DataFrame:
    """Seasonal report by product group and index group for winter versus summer."""
    seasonal_df = (
        transactions_df
        .filter(F.col("t_dat").isNotNull())
        .withColumn("Month_Num", F.month("t_dat"))
        .filter(F.col("Month_Num").isin([12, 1, 2, 6, 7, 8]))
        .join(
            articles_df.select("article_id", "product_group_name", "index_group_name"),
            on="article_id",
            how="inner",
        )
        .withColumn(
            "Season_Name",
            F.when(F.col("Month_Num").isin([12, 1, 2]), F.lit("Winter")).otherwise(F.lit("Summer")),
        )
        .withColumn("Product_Group_Name", F.initcap(F.coalesce(F.trim(F.col("product_group_name")), F.lit("Unknown"))))
        .withColumn("Index_Group_Name", F.initcap(F.coalesce(F.trim(F.col("index_group_name")), F.lit("Unknown"))))
        .groupBy("Product_Group_Name", "Index_Group_Name", "Season_Name")
        .agg(
            F.count("*").alias("Transaction_Count"),
            F.sum("price").alias("Total_Revenue"),
        )
    )

    winter_df = seasonal_df.filter(F.col("Season_Name") == "Winter").select(
        "Product_Group_Name",
        "Index_Group_Name",
        F.col("Transaction_Count").alias("Winter_Transaction_Count"),
        F.col("Total_Revenue").alias("Winter_Revenue"),
    )
    summer_df = seasonal_df.filter(F.col("Season_Name") == "Summer").select(
        "Product_Group_Name",
        "Index_Group_Name",
        F.col("Transaction_Count").alias("Summer_Transaction_Count"),
        F.col("Total_Revenue").alias("Summer_Revenue"),
    )

    result_df = (
        winter_df
        .join(summer_df, on=["Product_Group_Name", "Index_Group_Name"], how="full")
        .na.fill(0, subset=["Winter_Transaction_Count", "Winter_Revenue", "Summer_Transaction_Count", "Summer_Revenue"])
        .withColumn("Revenue_Delta", F.col("Winter_Revenue") - F.col("Summer_Revenue"))
        .withColumn(
            "Seasonality_Index_Pct",
            F.when(
                (F.col("Winter_Revenue") + F.col("Summer_Revenue")) > 0,
                F.col("Winter_Revenue") / (F.col("Winter_Revenue") + F.col("Summer_Revenue")) * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .orderBy(F.desc(F.abs("Revenue_Delta")), "Product_Group_Name")
        .limit(15)
    )
    return round_existing(
        result_df,
        {
            "Winter_Revenue": 2,
            "Summer_Revenue": 2,
            "Revenue_Delta": 2,
            "Seasonality_Index_Pct": 2,
        },
    ).select(
        F.col("Product_Group_Name").alias("Товарна_група"),
        F.col("Index_Group_Name").alias("Глобальна_товарна_група"),
        F.col("Winter_Transaction_Count").alias("Кількість_транзакцій_взимку"),
        F.col("Summer_Transaction_Count").alias("Кількість_транзакцій_влітку"),
        F.col("Winter_Revenue").alias("Виручка_взимку"),
        F.col("Summer_Revenue").alias("Виручка_влітку"),
        F.col("Revenue_Delta").alias("Різниця_між_виручкою_взимку_та_влітку"),
        F.col("Seasonality_Index_Pct").alias("Частка_зимової_виручки_від_сумарної_сезонної_виручки_(відсоток)"),
    )


# Питання: Якими є показники утримання клієнтів у 12 найчисленніших сегментах, сформованих за віковими групами та інтервалами між першою і другою покупкою?
# Опис колонок:
# - Вікова_група: Віковий діапазон клієнтів.
# - Інтервал_між_першою_та_другою_покупкою: Категорія тривалості між першою та другою покупкою.
# - Кількість_клієнтів: Кількість клієнтів у сегменті.
# - Середня_вартість_першої_покупки: Середня вартість першої покупки клієнтів сегмента.
# - Середня_кількість_днів_між_покупками: Середня кількість днів між першою та другою покупкою.
# - Частка_від_клієнтів_цієї_вікової_групи_(відсоток): Частка клієнтів сегмента серед усіх клієнтів відповідної вікової групи.
def query_3_retention_gap_distribution(
    transactions_df: DataFrame,
    customers_df: DataFrame,
) -> DataFrame:
    """Retention-stage report by age group with customer share and first-order value."""
    purchase_order = Window.partitionBy("customer_id").orderBy("t_dat")
    share_window = Window.partitionBy("Age_Group")

    ordered_df = (
        transactions_df
        .filter(F.col("t_dat").isNotNull())
        .withColumn("Purchase_Rank", F.row_number().over(purchase_order))
        .filter(F.col("Purchase_Rank") <= 2)
    )

    first_second_df = (
        ordered_df
        .groupBy("customer_id")
        .agg(
            F.min("t_dat").alias("First_Purchase_Date"),
            F.max("t_dat").alias("Second_Purchase_Date"),
            F.max(F.when(F.col("Purchase_Rank") == 1, F.col("price"))).alias("First_Order_Value"),
            F.count("*").alias("Purchase_Row_Count"),
        )
        .filter(F.col("Purchase_Row_Count") == 2)
        .withColumn("Days_Between", F.datediff("Second_Purchase_Date", "First_Purchase_Date"))
        .join(customers_df.select("customer_id", "age"), on="customer_id", how="left")
        .withColumn("Age_Group", build_age_group_expression("age"))
        .withColumn(
            "Retention_Bucket",
            F.when(F.col("Days_Between") == 0, F.lit("Same Day"))
            .when(F.col("Days_Between").between(1, 7), F.lit("1-7 Days"))
            .when(F.col("Days_Between").between(8, 30), F.lit("8-30 Days"))
            .when(F.col("Days_Between").between(31, 90), F.lit("31-90 Days"))
            .when(F.col("Days_Between").between(91, 180), F.lit("91-180 Days"))
            .otherwise(F.lit("181+ Days")),
        )
    )

    result_df = (
        first_second_df
        .groupBy("Age_Group", "Retention_Bucket")
        .agg(
            F.countDistinct("customer_id").alias("Customer_Count"),
            F.avg("First_Order_Value").alias("Avg_First_Order_Value"),
            F.avg("Days_Between").alias("Avg_Days_Between"),
        )
        .withColumn("Age_Group_Total", F.sum("Customer_Count").over(share_window))
        .withColumn(
            "Customer_Share_Pct",
            F.when(F.col("Age_Group_Total") > 0, F.col("Customer_Count") / F.col("Age_Group_Total") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Age_Group_Total")
        .orderBy(F.desc("Customer_Count"), "Age_Group", "Retention_Bucket")
        .limit(12)
    )
    return round_existing(
        result_df,
        {
            "Avg_First_Order_Value": 4,
            "Avg_Days_Between": 2,
            "Customer_Share_Pct": 2,
        },
    ).select(
        F.col("Age_Group").alias("Вікова_група"),
        F.col("Retention_Bucket").alias("Інтервал_між_першою_та_другою_покупкою"),
        F.col("Customer_Count").alias("Кількість_клієнтів"),
        F.col("Avg_First_Order_Value").alias("Середня_вартість_першої_покупки"),
        F.col("Avg_Days_Between").alias("Середня_кількість_днів_між_покупками"),
        F.col("Customer_Share_Pct").alias("Частка_від_клієнтів_цієї_вікової_групи_(відсоток)"),
    )


# Питання: Якими є характеристики пікових днів продажів у вибірці з 8 квартально-канальних сегментів 2019 року?
# Опис колонок:
# - Квартал_2019_року: Квартал 2019 року.
# - Канал_продажу: Назва каналу продажу.
# - Піковий_день: Дата з найбільшою кількістю транзакцій у кварталі.
# - Кількість_транзакцій_у_піковий_день: Кількість транзакцій у піковий день.
# - Кількість_транзакцій_у_кварталі: Загальна кількість транзакцій у кварталі.
# - Виручка_у_піковий_день: Загальна виручка пікового дня.
# - Частка_пікового_дня_від_транзакцій_кварталу_(відсоток): Частка транзакцій пікового дня від усіх транзакцій кварталу.
# - Співвідношення_пікового_дня_до_середнього_денного_рівня: Відношення транзакцій пікового дня до середньоденного рівня кварталу.
def query_4_top_peak_days_month_share(transactions_df: DataFrame) -> DataFrame:
    """Quarterly peak-day report by channel including quarter share and intensity."""
    peak_window = Window.partitionBy("Quarter_Label", "Channel_Name").orderBy(F.desc("Daily_Transaction_Count"), F.asc("t_dat"))

    daily_df = (
        transactions_df
        .filter(F.col("t_dat").isNotNull())
        .filter(F.year("t_dat") == 2019)
        .withColumn("Quarter_Label", F.concat(F.lit("2019-Q"), F.quarter("t_dat")))
        .withColumn("Channel_Name", build_channel_label_expression())
        .groupBy("Quarter_Label", "Channel_Name", "t_dat")
        .agg(
            F.count("*").alias("Daily_Transaction_Count"),
            F.sum("price").alias("Daily_Revenue"),
        )
    )

    quarterly_df = (
        daily_df
        .groupBy("Quarter_Label", "Channel_Name")
        .agg(
            F.sum("Daily_Transaction_Count").alias("Quarter_Transaction_Count"),
            F.avg("Daily_Transaction_Count").alias("Avg_Daily_Transaction_Count"),
        )
    )

    result_df = (
        daily_df
        .join(quarterly_df, on=["Quarter_Label", "Channel_Name"], how="inner")
        .withColumn("Peak_Rank", F.row_number().over(peak_window))
        .filter(F.col("Peak_Rank") == 1)
        .withColumnRenamed("t_dat", "Peak_Day")
        .withColumn(
            "Peak_Day_Share_Pct",
            F.when(
                F.col("Quarter_Transaction_Count") > 0,
                F.col("Daily_Transaction_Count") / F.col("Quarter_Transaction_Count") * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "Peak_vs_Avg_Daily_Ratio",
            F.when(F.col("Avg_Daily_Transaction_Count") > 0, F.col("Daily_Transaction_Count") / F.col("Avg_Daily_Transaction_Count")),
        )
        .orderBy("Quarter_Label", "Channel_Name")
        .limit(8)
    )
    return round_existing(
        result_df,
        {
            "Daily_Revenue": 2,
            "Peak_Day_Share_Pct": 2,
            "Peak_vs_Avg_Daily_Ratio": 2,
        },
    ).select(
        F.col("Quarter_Label").alias("Квартал_2019_року"),
        F.col("Channel_Name").alias("Канал_продажу"),
        F.col("Peak_Day").alias("Піковий_день"),
        F.col("Daily_Transaction_Count").alias("Кількість_транзакцій_у_піковий_день"),
        F.col("Quarter_Transaction_Count").alias("Кількість_транзакцій_у_кварталі"),
        F.col("Daily_Revenue").alias("Виручка_у_піковий_день"),
        F.col("Peak_Day_Share_Pct").alias("Частка_пікового_дня_від_транзакцій_кварталу_(відсоток)"),
        F.col("Peak_vs_Avg_Daily_Ratio").alias("Співвідношення_пікового_дня_до_середнього_денного_рівня"),
    )


# Питання: Які 7 днів тижня демонструють найвищі показники аномальної активності серед клієнтів віком 30-40 років?
# Опис колонок:
# - День_тижня: Назва дня тижня.
# - Кількість_активних_днів: Кількість днів з продажами у вибраному дні тижня.
# - Середня_кількість_транзакцій_за_день: Середня денна кількість транзакцій.
# - Середня_7_денна_ковзна_кількість_транзакцій: Середнє значення 7-денної ковзної кількості транзакцій.
# - Кількість_аномальних_днів: Кількість днів, що перевищили верхню межу аномальності.
# - Частка_аномальних_днів_від_усіх_активних_днів_(відсоток): Частка аномальних днів серед усіх активних днів цього дня тижня.
def query_5_rolling_activity_30_40(
    transactions_df: DataFrame,
    customers_df: DataFrame,
) -> DataFrame:
    """Weekday outlier-activity summary for customers aged 30-40."""
    rolling_window = Window.orderBy("Sale_Timestamp").rangeBetween(-6 * 86400, 0)

    daily_df = (
        transactions_df
        .join(
            customers_df.filter((F.col("age") >= 30) & (F.col("age") <= 40)).select("customer_id"),
            on="customer_id",
            how="inner",
        )
        .filter(F.col("t_dat").isNotNull())
        .groupBy("t_dat")
        .agg(F.count("*").alias("Daily_Transaction_Count"))
        .withColumn("Weekday_Num", F.dayofweek("t_dat"))
        .withColumn("Weekday_Name", build_weekday_label_expression("Weekday_Num"))
        .withColumn("Sale_Timestamp", F.col("t_dat").cast("timestamp").cast("long"))
    )

    activity_df = (
        daily_df
        .withColumn("Rolling_7d_Avg", F.avg("Daily_Transaction_Count").over(rolling_window))
        .withColumn("Rolling_7d_Std", F.stddev_pop("Daily_Transaction_Count").over(rolling_window))
        .withColumn("Upper_Band", F.col("Rolling_7d_Avg") + 2 * F.col("Rolling_7d_Std"))
        .withColumn("Is_Outlier", F.col("Daily_Transaction_Count") > F.col("Upper_Band"))
    )

    result_df = (
        activity_df
        .groupBy("Weekday_Name")
        .agg(
            F.count("*").alias("Active_Day_Count"),
            F.avg("Daily_Transaction_Count").alias("Avg_Daily_Transaction_Count"),
            F.avg("Rolling_7d_Avg").alias("Avg_Rolling_7d_Transactions"),
            F.sum(F.when(F.col("Is_Outlier"), F.lit(1)).otherwise(F.lit(0))).alias("Outlier_Day_Count"),
        )
        .withColumn(
            "Outlier_Share_Pct",
            F.when(F.col("Active_Day_Count") > 0, F.col("Outlier_Day_Count") / F.col("Active_Day_Count") * 100.0).otherwise(F.lit(0.0)),
        )
        .orderBy("Weekday_Name")
        .limit(7)
    )
    return round_existing(
        result_df,
        {
            "Avg_Daily_Transaction_Count": 2,
            "Avg_Rolling_7d_Transactions": 2,
            "Outlier_Share_Pct": 2,
        },
    ).select(
        F.col("Weekday_Name").alias("День_тижня"),
        F.col("Active_Day_Count").alias("Кількість_активних_днів"),
        F.col("Avg_Daily_Transaction_Count").alias("Середня_кількість_транзакцій_за_день"),
        F.col("Avg_Rolling_7d_Transactions").alias("Середня_7_денна_ковзна_кількість_транзакцій"),
        F.col("Outlier_Day_Count").alias("Кількість_аномальних_днів"),
        F.col("Outlier_Share_Pct").alias("Частка_аномальних_днів_від_усіх_активних_днів_(відсоток)"),
    )


# Питання: Яким є розподіл покупок у вибірці з 14 сегментів, сформованих за статусом ACTIVE або PRE-CREATE та днем тижня?
# Опис колонок:
# - Статус_клубного_членства: Статус клубного членства клієнта.
# - День_тижня: Назва дня тижня.
# - Кількість_транзакцій: Кількість транзакцій у цей день тижня.
# - Загальна_кількість_транзакцій_за_статусом: Загальна кількість транзакцій для цього статусу.
# - Частка_від_транзакцій_даного_статусу_(відсоток): Частка транзакцій дня тижня серед усіх транзакцій цього статусу.
# - Індекс_відносно_середнього_рівня_дня_тижня_(відсоток): Індекс відносно середнього рівня транзакцій цього дня тижня між статусами.
def query_6_frequency_loyal_vs_guest(
    transactions_df: DataFrame,
    customers_df: DataFrame,
) -> DataFrame:
    """Weekday purchase-share report for ACTIVE versus PRE-CREATE customers."""
    segment_window = Window.partitionBy("Club_Member_Status")
    day_window = Window.partitionBy("Weekday_Name")

    base_df = (
        transactions_df
        .join(
            customers_df.filter(F.col("club_member_status").isin(["ACTIVE", "PRE-CREATE"])).select("customer_id", "club_member_status"),
            on="customer_id",
            how="inner",
        )
        .filter(F.col("t_dat").isNotNull())
        .withColumn("Weekday_Num", F.dayofweek("t_dat"))
        .withColumn("Weekday_Name", build_weekday_label_expression("Weekday_Num"))
        .withColumn("Club_Member_Status", F.col("club_member_status"))
        .groupBy("Club_Member_Status", "Weekday_Name")
        .agg(F.count("*").alias("Transaction_Count"))
    )

    result_df = (
        base_df
        .withColumn("Segment_Total_Transactions", F.sum("Transaction_Count").over(segment_window))
        .withColumn(
            "Segment_Day_Share_Pct",
            F.when(F.col("Segment_Total_Transactions") > 0, F.col("Transaction_Count") / F.col("Segment_Total_Transactions") * 100.0).otherwise(F.lit(0.0)),
        )
        .withColumn("Weekday_Avg_Transactions", F.avg("Transaction_Count").over(day_window))
        .withColumn(
            "Weekday_Index_100",
            F.when(F.col("Weekday_Avg_Transactions") > 0, F.col("Transaction_Count") / F.col("Weekday_Avg_Transactions") * 100.0).otherwise(F.lit(0.0)),
        )
        .drop("Weekday_Avg_Transactions")
        .orderBy("Club_Member_Status", "Weekday_Name")
        .limit(14)
    )
    return round_existing(result_df, {"Segment_Day_Share_Pct": 2, "Weekday_Index_100": 2}).select(
        F.col("Club_Member_Status").alias("Статус_клубного_членства"),
        F.col("Weekday_Name").alias("День_тижня"),
        F.col("Transaction_Count").alias("Кількість_транзакцій"),
        F.col("Segment_Total_Transactions").alias("Загальна_кількість_транзакцій_за_статусом"),
        F.col("Segment_Day_Share_Pct").alias("Частка_від_транзакцій_даного_статусу_(відсоток)"),
        F.col("Weekday_Index_100").alias("Індекс_відносно_середнього_рівня_дня_тижня_(відсоток)"),
    )


def run(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    output_root: str = "output/dmytro",
) -> None:
    directories = ensure_output_dirs(output_root)
    initialize_explain_log(
        os.path.join(directories["logs"], "explain_logs.txt"),
        "Dmytro transformation query explain logs",
    )

    transactions = load_parquet_dataset(spark, processed_dir, "transactions")
    articles = load_parquet_dataset(spark, processed_dir, "articles")
    customers = load_parquet_dataset(spark, processed_dir, "customers")

    save_report(query_1_weekly_transaction_dynamics(transactions), "Query 1 - Quarterly transaction and revenue report by channel", "query_1", directories)
    save_report(query_2_winter_vs_summer_spike(transactions, articles), "Query 2 - Winter versus summer report by product group and index group", "query_2", directories)
    save_report(query_3_retention_gap_distribution(transactions, customers), "Query 3 - Retention bucket report by age group", "query_3", directories)
    save_report(query_4_top_peak_days_month_share(transactions), "Query 4 - Quarterly peak-day report by channel", "query_4", directories)
    save_report(query_5_rolling_activity_30_40(transactions, customers), "Query 5 - Weekday activity anomaly summary for ages 30-40", "query_5", directories)
    save_report(query_6_frequency_loyal_vs_guest(transactions, customers), "Query 6 - Weekday purchase-share report by club member status", "query_6", directories)


if __name__ == "__main__":
    spark_session = create_spark_session("HM-Fashion-Dmytro-Pipeline")
    try:
        run(spark_session)
    finally:
        spark_session.stop()
