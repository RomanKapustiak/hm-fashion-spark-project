import os
from src.customers.schema import load_customers

def run_extraction(spark, raw_dir="data/raw"):
    """Extract customers from csv."""
    print("--- Stage 3: Extraction (Customers) ---")
    file_path = os.path.join(raw_dir, "customers.csv")
    
    df = load_customers(spark, file_path)
    
    # verify it worked
    print("Schema:")
    df.printSchema()
    
    print("Row count: {}".format(df.count()))
    
    print("First 5 rows:")
    df.show(5, truncate=False)
    
    return df
