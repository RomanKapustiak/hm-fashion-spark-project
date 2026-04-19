# H&M Personalized Fashion Recommendations - Big Data Project

This project is developed as part of the Big Data Mining course. It utilizes Apache Spark (PySpark) to process and analyze the H&M dataset within a distributed environment.

## Project Goal
To analyze customer behavior and product trends using PySpark, implementing a full data pipeline from extraction to transformation and analysis.

## Dataset
**Source:** H&M Personalized Fashion Recommendations (Kaggle)
* **articles.csv**: Product metadata (category, color, department).
* **customers.csv**: Customer demographics (age, club status).
* **transactions_train.csv**: Purchase history (date, price, channel).

## Prerequisites
* **Docker Desktop** installed and running:
  * Windows: [Download for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
  * macOS: [Download for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)

> **Note:** All `docker compose` commands in this guide work identically on Windows (PowerShell / CMD) and macOS (Terminal / zsh).

## Quick Start

### 1. Place Your Data
Put the Kaggle CSV files into the `data/raw/` directory:
```
data/raw/articles.csv
data/raw/customers.csv
data/raw/transactions_train.csv
```

### 2. Build the Docker Image
```bash
docker compose build
```
This builds once. Rebuild only when you change `Dockerfile` or `requirements.txt`.

### 3. Verify Everything Works
```bash
docker compose run spark-app python src/scripts/verify_data.py
```

### 4. Run the Preprocessing Pipelines
```bash
docker compose run spark-app
```
This runs `main.py`, which saves cleaned parquet outputs to:
```text
data/processed/articles
data/processed/customers
data/processed/transactions
```

### 5. Run a Team-Member Transformation Pipeline
Example:
Roman's standalone financial performance pipeline is available at:
```text
src/transformations/roman_pipeline.py
```

Run it with:
```bash
docker compose run --rm spark-app python src/transformations/roman_pipeline.py
```

It reads the processed parquet data and writes outputs to:
```text
output/roman/csv/
output/roman/plots/
output/roman/logs/
```

The `output/` directory is mounted into the container, so generated CSV files, plots, and explain logs are persisted on your local machine.
---

## Two Modes of Work

### Mode 1: Run PySpark Scripts
Use `spark-app` service to execute any Python script inside the container.

```bash
# Run the default entry point (main.py)
docker compose run spark-app

# Run a specific script
docker compose run spark-app python src/scripts/verify_data.py

# Run Roman's transformation pipeline
docker compose run --rm spark-app python src/transformations/roman_pipeline.py

# Run Artem's transformation pipeline
docker compose run --rm spark-app python src/transformations/artem_pipeline.py
```

Data is mounted at `/app/data` inside the container, so in your scripts use:
```python
df = spark.read.csv("data/raw/articles.csv", header=True, inferSchema=True)
```

### Mode 2: Jupyter Notebook (Interactive)
Use `notebook` service to start Jupyter Lab in your browser.

```bash
docker compose up notebook
```

Then open **http://localhost:8888** in your browser. The `notebooks/` folder is mounted, so any notebooks you create or edit are saved to your local disk.

In notebooks, use absolute container paths for data:
```python
df = spark.read.csv("/app/data/raw/articles.csv", header=True, inferSchema=True)
```

Generated transformation outputs are also available inside the container at:
```text
/app/output
```

To stop the notebook server, press `Ctrl+C` in the terminal or run:
```bash
docker compose down
```

---

## Useful Docker Commands

| Command | Description |
|---------|-------------|
| `docker compose build` | Build (or rebuild) the image |
| `docker compose run spark-app` | Run `main.py` |
| `docker compose run spark-app python <script>` | Run a specific script |
| `docker compose run --rm spark-app python src/transformations/roman_pipeline.py` | Run Roman's finance transformation pipeline |
| `docker compose run --rm spark-app python src/transformations/artem_pipeline.py` | Run Artem's customer transformation pipeline |
| `docker compose up notebook` | Start Jupyter Lab on port 8888 |
| `docker compose down` | Stop all services and remove containers |
| `docker compose down --remove-orphans` | Stop all and clean up orphan containers |
| `docker compose build --no-cache` | Full rebuild (after Dockerfile changes) |
| `docker compose logs notebook` | View notebook service logs |
