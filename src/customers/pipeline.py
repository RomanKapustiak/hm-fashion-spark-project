import os
from src.customers.extraction import run_extraction
from src.customers.preprocessing import run_preprocessing

def run(spark, raw_dir="data/raw", processed_dir="data/processed"):
    """Run the full customers pipeline."""
    print("Starting customers pipeline...")
    
    df = run_extraction(spark, raw_dir)
    df_clean = run_preprocessing(df)
    
    # Save results
    out_path = os.path.join(processed_dir, "customers")
    print("Saving processed data to: {}".format(out_path))
    
    df_clean.write.mode("overwrite").parquet(out_path)
    print("Customers pipeline done!")
