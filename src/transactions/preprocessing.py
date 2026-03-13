# preprocessing for transactions_train.csv
# stage 4

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _step1_general_statistics(df: DataFrame) -> None:
    _section("Step 1 - general stats")

    row_count = df.count()
    print(f"rows: {row_count:,}, cols: {len(df.columns)}")
    print(f"columns: {df.columns}")

    # describe() gives count/mean/stddev/min/max for each col
    print("\ndescribe():")
    df.describe().show(truncate=False)

    # dataset is H&M purchase events - one row = one item bought by one customer
    # spans 2018-2020, ~31M rows, main source for any sales analysis
    print("transactions_train: each row is one purchase (customer, article, date, price, channel)")


def _step2_numerical_analysis(df: DataFrame) -> None:
    _section("Step 2 - numerical analysis")

    # only price and sales_channel_id are numeric here
    quantiles = df.stat.approxQuantile("price", [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
    agg_row = df.select(
        F.mean("price").alias("price_mean"),
        F.stddev("price").alias("price_stddev"),
        F.mean("sales_channel_id").alias("channel_mean"),
        F.countDistinct("sales_channel_id").alias("channel_distinct"),
    ).collect()[0]

    stddev_val = agg_row["price_stddev"]
    channel_mean_val = agg_row["channel_mean"]

    print("price:")
    print(f"  min={quantiles[0]:.4f}  Q1={quantiles[1]:.4f}  median={quantiles[2]:.4f}  Q3={quantiles[3]:.4f}  max={quantiles[4]:.4f}")
    print(f"  mean={agg_row['price_mean']:.4f}  stddev={f'{stddev_val:.4f}' if stddev_val is not None else 'N/A'}")
    # prices are normalized (divided by some constant), most are < 0.06
    # distribution is right-skewed, few expensive items pull the mean up

    print("\nsales_channel_id (1=store, 2=online):")
    print(f"  distinct values: {agg_row['channel_distinct']}")
    print(f"  mean: {f'{channel_mean_val:.4f}' if channel_mean_val is not None else 'N/A'}")
    df.groupBy("sales_channel_id").count().orderBy("sales_channel_id").show()


def _step3_type_casting(df: DataFrame) -> DataFrame:
    _section("Step 3 - type casting and date parsing")

    # schema already sets the types but let's cast explicitly just in case
    df = df.withColumn("t_dat", df["t_dat"].cast(DateType()))
    df = df.withColumn("price", df["price"].cast(DoubleType()))
    df = df.withColumn("sales_channel_id", df["sales_channel_id"].cast(IntegerType()))

    # extract some time features from date - useful for seasonality analysis later
    df = (
        df
        .withColumn("year", F.year("t_dat"))
        .withColumn("month", F.month("t_dat"))
        .withColumn("day_of_week", F.dayofweek("t_dat"))  # 1=Sun, 7=Sat
    )

    print("casts done, added year/month/day_of_week")
    df.printSchema()
    return df


def _step4_feature_informativeness(df: DataFrame) -> DataFrame:
    _section("Step 4 - feature informativeness")

    total = df.count()
    print("{:<22} {:>10} {:>10}  {}".format("column", "distinct", "null%", "verdict"))
    print("-" * 65)

    # don't drop core transaction columns even if they look constant on small samples
    CORE_COLS = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id",
                 "year", "month", "day_of_week"}

    drop_cols = []
    for col_name in df.columns:
        distinct = df.select(col_name).distinct().count()
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = (null_count / total * 100) if total > 0 else 0.0

        if col_name in CORE_COLS:
            verdict = "keep (core)"
        elif distinct == 1:
            verdict = "drop (constant)"
            drop_cols.append(col_name)
        else:
            verdict = "keep"

        print("{:<22} {:>10,} {:>9.2f}%  {}".format(col_name, distinct, null_pct, verdict))

    if drop_cols:
        print(f"\ndropping: {drop_cols}")
        df = df.drop(*drop_cols)
    else:
        print("\nnothing to drop")

    # notes:
    # - customer_id is just a sha256 hash, useless as a feature but needed for joins
    # - sales_channel_id only has 2 values but it's still meaningful (store vs online)
    # - year/month/day_of_week added in step 3, useful for time-based patterns
    return df


def _step5_nulls_and_duplicates(df: DataFrame) -> DataFrame:
    _section("Step 5 - nulls and duplicates")

    total_before = df.count()

    # count nulls per column
    print("nulls per column:")
    df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
    ]).show(truncate=False)

    df_no_nulls = df.dropna()
    print(f"rows dropped (nulls): {total_before - df_no_nulls.count():,}")

    # count how many exact duplicates exist just for info, but DO NOT drop them
    # a customer can legitimately buy 2 of the exactly same item on the same day
    exact_dups = df_no_nulls.count() - df_no_nulls.dropDuplicates().count()
    print(f"\nexact duplicate rows: {exact_dups:,} (kept - these are repeat purchases)")

    total_after = df_no_nulls.count()
    print(f"before: {total_before:,}  after: {total_after:,}  removed: {total_before - total_after:,}")

    return df_no_nulls



def run_preprocessing(df: DataFrame) -> DataFrame:
    """runs all 5 preprocessing steps on the transactions df"""
    print("\n=== PREPROCESSING transactions_train.csv ===")

    _step1_general_statistics(df)
    _step2_numerical_analysis(df)
    df = _step3_type_casting(df)
    df = _step4_feature_informativeness(df)
    df = _step5_nulls_and_duplicates(df)

    print("\ndone.")
    return df
