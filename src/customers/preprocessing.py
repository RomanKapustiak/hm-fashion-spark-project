from pyspark.sql.functions import col, isnan, count, when
from pyspark.sql.types import IntegerType, DoubleType

def _step1_general_stats(df):
    print("step 1 - general statistics")
    row_count = df.count()
    col_count = len(df.columns)
    print("Row count: {}".format(row_count))
    print("Column count: {}".format(col_count))
    print("Columns: {}".format(df.columns))
    
    print("Describe:")
    df.describe().show()
    
    print("this dataset holds customer metadata like age and club status.")
    print("it has info on whether they get fashion news and if they are active.")
    return df

def _step2_numerical_analysis(df):
    print("step 2 - numerical analysis")
    num_cols = ["FN", "Active", "age"]
    
    for c in num_cols:
        # approxQuantile returns list of lists if we pass multiple cols, so doing one by one
        quantiles = df.approxQuantile(c, [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
        
        # get mean and stddev
        stats = df.select(c).describe().filter(col("summary").isin("mean", "stddev")).collect()
        
        # handle case where column is all nulls/no stats
        if not stats:
            continue
            
        mean_val = next((row[c] for row in stats if row["summary"] == "mean"), None)
        std_val = next((row[c] for row in stats if row["summary"] == "stddev"), None)
        
        print("Column: {}".format(c))
        print("Min: {}, Q1: {}, Median: {}, Q3: {}, Max: {}".format(*quantiles))
        print("Mean: {}, Stddev: {}".format(mean_val, std_val))
        print("---")
        
    print("age seems to have a fairly normal spread, maybe slightly skewed.")
    print("FN and Active mostly just contain 1.0 or are left null.")
    return df

def _step3_type_casting(df):
    print("step 3 - type casting")
    # explicitly cast columns according to their defined types just in case
    df = df.withColumn("age", col("age").cast(IntegerType()))
    df = df.withColumn("FN", col("FN").cast(DoubleType()))
    df = df.withColumn("Active", col("Active").cast(DoubleType()))
    return df

def _step4_feature_info(df):
    print("step 4 - feature informativeness")
    total_rows = df.count()
    if total_rows == 0:
        return df
        
    for c in df.columns:
        distinct_count = df.select(c).distinct().count()
        null_count = df.filter(col(c).isNull() | isnan(col(c))).count()
        null_pct = (null_count / total_rows) * 100
        
        drop_verdict = "keep"
        if distinct_count == 1 and c not in ["customer_id", "postal_code"]:
            drop_verdict = "drop"
            # actually dropping it from the dataframe if the verdict is drop
            df = df.drop(c)
            
        print("Col: {} - Distinct: {}, Nulls: {}% -> Verdict: {}".format(c, distinct_count, round(null_pct, 2), drop_verdict))
    
    return df

def _step5_nulls_and_dups(df):
    print("step 5 - nulls and duplicates")
    before_count = df.count()
    
    # Count nulls per column
    print("Null counts per column:")
    df.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in df.columns]).show()
    
    # Drop rows with any null
    df = df.dropna()
    after_nulls = df.count()
    
    # keeping duplicates exactly as the instructions say doing nothing about them
    
    print("Count before processing: {}".format(before_count))
    print("Count after dropping nulls: {}".format(after_nulls))
    print("Duplicates were kept on purpose because repeating rows are legitimate for this domain.")
    
    return df

def run_preprocessing(df):
    """Run all preprocessing steps for customers."""
    print("--- Stage 4: Preprocessing (Customers) ---")
    df = _step1_general_stats(df)
    df = _step2_numerical_analysis(df)
    df = _step3_type_casting(df)
    df = _step4_feature_info(df)
    df = _step5_nulls_and_dups(df)
    return df
