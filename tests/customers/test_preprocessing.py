from pyspark.sql import Row
from pyspark.sql.types import IntegerType, DoubleType
from src.customers.preprocessing import run_preprocessing, _step3_type_casting, _step5_nulls_and_dups

def test_step3_type_casting(spark):
    """Test that columns are cast correctly."""
    df = spark.createDataFrame(
        [("1", None, "1.0", 22.0)],
        "customer_id string, FN string, Active string, age double",
    )

    res = _step3_type_casting(df)
    dtypes = dict(res.dtypes)
    assert dtypes["age"] == "int"
    assert dtypes["FN"] == "double"
    assert dtypes["Active"] == "double"

def test_step5_nulls_and_dups(spark):
    """Flags are filled with 0, rows without id are dropped, duplicates are kept."""
    df = spark.createDataFrame(
        [
            ("1", 20, 1.0),
            ("2", None, 1.0),   # null age is kept
            ("1", 20, 1.0),     # duplicate, should keep
            ("3", 30, None),    # null FN -> 0.0, kept
            (None, 40, 1.0),    # no id -> dropped
        ],
        "customer_id string, age int, FN double",
    )

    res = _step5_nulls_and_dups(df)

    assert res.count() == 4
    fn_by_id = {r["customer_id"]: r["FN"] for r in res.collect()}
    assert fn_by_id["3"] == 0.0


def test_step5_normalizes_categories(spark):
    """Category spelling is unified and missing values become UNKNOWN."""
    df = spark.createDataFrame(
        [("1", "NONE", "ACTIVE"), ("2", "None", None), ("3", None, "PRE-CREATE")],
        "customer_id string, fashion_news_frequency string, club_member_status string",
    )

    res = {r["customer_id"]: r for r in _step5_nulls_and_dups(df).collect()}

    assert res["1"]["fashion_news_frequency"] == "NONE"
    assert res["2"]["fashion_news_frequency"] == "NONE"
    assert res["3"]["fashion_news_frequency"] == "UNKNOWN"
    assert res["2"]["club_member_status"] == "UNKNOWN"


def test_full_preprocessing_smoke(spark):
    """Smoke test to ensure the whole preprocessing pipeline runs without crash."""
    df = spark.createDataFrame([
        Row(customer_id="1", FN=1.0, Active=1.0, club_member_status="ACTIVE", fashion_news_frequency="NONE", age=20, postal_code="abc"),
        Row(customer_id="2", FN=None, Active=None, club_member_status="ACTIVE", fashion_news_frequency="NONE", age=30, postal_code="xyz")
    ])

    res = run_preprocessing(df)
    # both customers are kept: empty FN/Active mean "no", not a missing value
    assert res.count() == 2
