import os
from src.customers.schema import CUSTOMERS_SCHEMA
from src.customers.extraction import run_extraction

def test_customers_schema_fields():
    """Test that the schema has the correct names and types."""
    field_names = [f.name for f in CUSTOMERS_SCHEMA.fields]
    assert "customer_id" in field_names
    assert "age" in field_names
    
    # check type
    customer_id_field = next(f for f in CUSTOMERS_SCHEMA.fields if f.name == "customer_id")
    assert customer_id_field.dataType.typeName() == "string"

def test_extraction_loads_data(spark, tmp_path):
    """Test extraction loads CSV correctly and keeps leading zeros."""
    # mock some data
    test_csv = tmp_path / "customers.csv"
    with open(test_csv, "w") as f:
        f.write("customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n")
        f.write("000123,1.0,1.0,ACTIVE,NONE,22,00xyz\n")
        f.write("abc456,,,,,50,11xyz\n")
        
    df = run_extraction(spark, raw_dir=str(tmp_path))
    
    # check counts
    assert df.count() == 2
    
    # verify leading zeros are kept on IDs
    first_row = df.filter(df.age == 22).collect()[0]
    assert first_row["customer_id"] == "000123"
    assert first_row["postal_code"] == "00xyz"
