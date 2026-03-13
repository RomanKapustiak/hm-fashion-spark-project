# transactions module

Handles **Stage 3 (Extraction)** and **Stage 4 (Preprocessing)** for `transactions_train.csv`.

## Files

| File | What it does |
|---|---|
| `schema.py` | defines `TRANSACTIONS_SCHEMA` and `load_transactions()` |
| `extraction.py` | loads the CSV + prints schema/count/sample to verify |
| `preprocessing.py` | 5-step preprocessing pipeline |
| `pipeline.py` | ties everything together, saves result as Parquet |

## Dataset: transactions_train.csv

~31M rows, 5 columns:

| Column | Type | Notes |
|---|---|---|
| `t_dat` | date | purchase date |
| `customer_id` | string | sha256 hash |
| `article_id` | string | kept as string (leading zeros) |
| `price` | double | normalized, mostly 0.0–0.06 |
| `sales_channel_id` | int | 1 = store, 2 = online |

## What was tricky

**`article_id` must be `StringType`** — looks like a number (e.g. `0663713001`) but has leading zeros.
If you let Spark `inferSchema`, it reads it as `LongType` and silently drops the leading zero.
That's why we use an explicit schema.

**Price is normalized, not actual currency** — values like `0.0508` are not euros/dollars.
H&M divided all prices by some constant before publishing. Distribution is right-skewed:
median ~0.025, but a long tail goes up to ~0.59. Don't treat it as raw price.

**"Duplicates" are not what they look like** — deduplicating on `(t_dat, customer_id, article_id)`
drops 3.2M rows (~10% of the dataset). That sounds like a lot of duplicates but it's wrong —
a customer can legitimately buy 2 of the same item in one transaction. We do NOT drop
any rows (not even exact duplicates), as they all represent valid repeat purchases.

**`customer_id` is a SHA-256 hex hash** — 64 chars, all lowercase hex. Useless as a feature
for ML but essential for joins and groupBy. Do not try to cast it to a number.

**File is 3.5 GB** — `inferSchema=True` would scan the whole file twice (once to infer, once to read).
With an explicit schema it reads in one pass. Makes a noticeable difference at 31M rows.


## Preprocessing steps

1. **General stats** — `describe()`, row/col counts, dataset description
2. **Numerical analysis** — price percentiles, stddev, channel distribution
3. **Type casting + date parsing** — adds `year`, `month`, `day_of_week`
4. **Feature informativeness** — checks distinct/null per column, drops constants
5. **Nulls + duplicates** — drops null rows (exact duplicates are kept)

## Usage

```python
from src.spark_utils import create_spark_session
from src.transactions.pipeline import run

spark = create_spark_session()
run(spark)  # extracts, preprocesses, saves to data/processed/transactions/
spark.stop()
```

Or step by step:

```python
from src.transactions.extraction import run_extraction
from src.transactions.preprocessing import run_preprocessing

raw_df = run_extraction(spark)
clean_df = run_preprocessing(raw_df)
```

## Run

```bash
# full pipeline via main.py
docker compose run spark-app python main.py

# tests only
docker compose run --rm spark-app python -m pytest tests/transactions/ -v

# interactive notebook
docker compose up notebook
# → http://localhost:8888 → src/notebooks/roman_transactions.ipynb
```

## Output

Cleaned data saved to `data/processed/transactions/` as **Parquet**.

Load it in downstream stages:
```python
df = spark.read.parquet("data/processed/transactions")
```
