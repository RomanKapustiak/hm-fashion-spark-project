"""Preprocessing for articles.csv."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType

TEXT_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_code",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]

REQUIRED_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
]

INTEGER_COLUMNS = [
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
]



def _section(title: str) -> None:
    print(f"\n--- {title} ---")



def _step1_general_statistics(df: DataFrame) -> None:
    _section("Step 1 - general stats")
    print(f"rows: {df.count():,}, cols: {len(df.columns)}")
    print(f"columns: {df.columns}")
    df.describe().show(truncate=False)



def _step2_type_casting_and_normalization(df: DataFrame) -> DataFrame:
    _section("Step 2 - type casting and text normalization")

    if "article_id" in df.columns:
        df = df.withColumn("article_id", F.col("article_id").cast(StringType()))

    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast(IntegerType()))

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.lower(F.trim(F.col(column))))

    print("casts and lowercasing complete")
    df.printSchema()
    return df



def _step3_feature_informativeness(df: DataFrame) -> DataFrame:
    _section("Step 3 - feature informativeness")

    total = df.count()
    print("{:<30} {:>10} {:>10}".format("column", "distinct", "null%"))
    print("-" * 56)

    for column in df.columns:
        distinct_count = df.select(column).distinct().count()
        null_count = df.filter(F.col(column).isNull()).count()
        null_pct = (null_count / total * 100) if total else 0.0
        print("{:<30} {:>10,} {:>9.2f}%".format(column, distinct_count, null_pct))

    return df



def _step4_null_handling(df: DataFrame) -> DataFrame:
    _section("Step 4 - null handling for required business fields")

    required_present = [column for column in REQUIRED_COLUMNS if column in df.columns]
    before = df.count()
    cleaned = df.dropna(subset=required_present)
    after = cleaned.count()

    print(f"required columns checked: {required_present}")
    print(f"before: {before:,}  after: {after:,}  removed: {before - after:,}")
    return cleaned



def _step5_deduplicate_by_article_id(df: DataFrame) -> DataFrame:
    _section("Step 5 - deduplicate by article_id")

    if "article_id" not in df.columns:
        print("article_id not found, skipping deduplication")
        return df

    before = df.count()
    candidate_columns = sorted([column for column in df.columns if column != "article_id"])

    # Deterministic tie-break rule:
    # for duplicate article_id groups, keep the row with the lexicographically
    # smallest tuple of all other columns after casting values to string and
    # replacing nulls with empty strings.
    if candidate_columns:
        order_by = [
            F.coalesce(F.col(column).cast(StringType()), F.lit("")).asc()
            for column in candidate_columns
        ]
    else:
        order_by = [F.lit(1).asc()]

    window = Window.partitionBy("article_id").orderBy(*order_by)
    deduped = (
        df.withColumn("_article_row_rank", F.row_number().over(window))
        .filter(F.col("_article_row_rank") == 1)
        .drop("_article_row_rank")
    )
    after = deduped.count()

    print(f"before: {before:,}  after: {after:,}  removed: {before - after:,}")
    return deduped



def run_preprocessing(df: DataFrame) -> DataFrame:
    """Run the five-step preprocessing pipeline for articles data."""
    print("\n=== PREPROCESSING articles.csv ===")

    _step1_general_statistics(df)
    df = _step2_type_casting_and_normalization(df)
    df = _step3_feature_informativeness(df)
    df = _step4_null_handling(df)
    df = _step5_deduplicate_by_article_id(df)

    print("\ndone.")
    return df
