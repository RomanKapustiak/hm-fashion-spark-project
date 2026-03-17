from pyspark.sql import Row
from pyspark.sql.types import IntegerType, DoubleType
from src.customers.preprocessing import run_preprocessing, _step3_type_casting, _step5_nulls_and_dups

def test_step3_type_casting(spark):
    """Test that columns are cast correctly."""
    df = spark.createDataFrame([
        Row(customer_id="1", FN=None, Active="1.0", age=22.0)
    ])
    
    res = _step3_type_casting(df)
    dtypes = dict(res.dtypes)
    assert dtypes["age"] == "int"
    assert dtypes["FN"] == "double"
    assert dtypes["Active"] == "double"

def test_step5_nulls_and_dups(spark):
    """Test that nulls are dropped but duplicates are kept."""
    # Note: the actual dataframe created relies on schema from extraction, 
    # but here we just need to ensure the null dropping logic works
    df = spark.createDataFrame([
        Row(customer_id="1", age=20, FN=1.0),
        Row(customer_id="2", age=None, FN=1.0),  # should drop (null age)
        Row(customer_id="1", age=20, FN=1.0),    # duplicate, should keep
        Row(customer_id="3", age=30, FN=None)    # should drop (null FN)
    ])
    
    res = _step5_nulls_and_dups(df)
    
    assert res.count() == 2
    
def test_full_preprocessing_smoke(spark):
    """Smoke test to ensure the whole preprocessing pipeline runs without crash."""
    df = spark.createDataFrame([
        Row(customer_id="1", FN=1.0, Active=1.0, club_member_status="ACTIVE", fashion_news_frequency="NONE", age=20, postal_code="abc"),
        Row(customer_id="2", FN=None, Active=None, club_member_status="ACTIVE", fashion_news_frequency="NONE", age=30, postal_code="xyz")
    ])
    
    res = run_preprocessing(df)
    # Row 2 drops because of NULL in FN/Active. Row 1 remains.
    assert res.count() == 1
