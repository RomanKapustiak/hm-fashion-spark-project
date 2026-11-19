# Roles & Responsibilities — H&M Personalized Fashion Recommendations

---

## Team Members
- Taras
- Yura
- Artem
- Roman
- Dmytro

---

## Overview
This document records the initial distribution of roles and the project stages for the H&M Personalized Fashion Recommendations course project. It is the canonical starting point for task assignment and should be updated via pull request as responsibilities evolve.

---

## Project Stages and Role Distribution

### 1) Preparation Stage
- Roman: Repository initialization, main/dev branch configuration.
- Yura: Local project structure setup and `.gitignore` configuration.
- Artem: Dataset preview — data type identification and initial feature selection.
- Taras: Dataset acquisition and local data integrity verification.
- Dmytro: Initial documentation and `roles.md` file preparation.

### 2) Configuration Stage
All team members install and verify Docker locally. Individual contributions:
- Roman: `Dockerfile` and Docker Compose setup, guide for running PySpark scripts and notebooks.
- Yura: Automation — Kaggle API script for downloading and unpacking the data into `data/raw`.
- Taras: Presentation — template, domain description, progress of stages 1–2.
- Artem: Presentation — content, progress visualization and analytics.
- Dmytro: Report update, Docker verification (screenshots), documentation of roles and technical description.

### 3) Extraction Stage
- Roman: StructType schema and loading module for `transactions_train.csv`.
- Yura: StructType schema and correctly typed loading of `customers.csv`.
- Dmytro: StructType schema and correctly typed loading of `articles.csv`.
- Taras: Testing of the extraction flow, verification that all three tables load into memory.
- Artem: Code review of the loading logic, data integrity checks after loading into PySpark.

### 4) Data Preprocessing Stage
- Roman: Transactions cleaning pipeline — type casting (dates, identifiers), missing values, duplicate analysis of purchase history.
- Yura: Customers — text normalization, informativeness analysis of demographic features, null handling (age, club status).
- Dmytro: Articles pipeline — statistics, type casting, text normalization, deduplication by `article_id`.
- Taras: Analysis of cleaning results, structuring the report according to the project requirements.
- Artem: Analysis of cleaning results, adding preprocessing results to the presentation.

### 5) Transformation Stage
Each member implements 6 business queries and provides an execution plan analysis (`.explain()`):
- Taras: Product hierarchy & department queries
- Yura: Visual trends & design feature queries
- Artem: Customer demographics & behavior queries
- Roman: Financial performance & revenue queries
- Dmytro: Time-series dynamics & retention queries

Requirements per member: at least 3 filters, 2 joins, 2 group-bys, 2 window functions, plus query plan analysis.

### 6) Data Analysis Stage (Regression & Classification Models)
Two shared tasks on shared feature tables with a shared train/test split (`src/analysis/`):
- Classification: will a customer buy anything during the next 4 weeks?
- Regression: average price of an article from its characteristics.

Each member owns one algorithm family and applies it to both tasks:
- Roman: Linear models — `LinearRegression`, `LogisticRegression`. Shared foundation: feature tables, split, preprocessing, tuning and evaluation utilities, fix of customer null handling, model comparison.
- Taras: Decision tree — `DecisionTreeRegressor`, `DecisionTreeClassifier`.
- Artem: Random forest — `RandomForestRegressor`, `RandomForestClassifier`. Review of customer features.
- Dmytro: Gradient-boosted trees — `GBTRegressor`, `GBTClassifier`.
- Yura: Factorization machines — `FMRegressor`, `FMClassifier`.

Requirements per member: hyper-parameter tuning, analysis of the training process, test metrics (RMSE, R² / Accuracy, Precision, Recall, F1), comparison against a trivial baseline.

### 7) Results Writing Stage
- Taras: Export product analytics to CSV
- Yura: Export design/trend analytics to CSV
- Artem: Export customer segmentation to CSV
- Roman: Export financial reports and sales-channel analytics to CSV; model comparison tables (`output/ml/comparison/`)
- Dmytro: Validate pipeline outputs and ensure CSV storage in Docker
- All members: model metrics, tuning results and training plots to `output/ml/<member>/`

### 8) Presentation Stage
- Team presentation of findings and pipeline demonstration (shared responsibility).

---

## Development & Coding Standards (brief)
- Follow PEP 8 and include type hints for public functions.
- Use `pytest` for unit tests and place tests in `tests/`.
- Run code and tests in Docker: `docker compose run --rm spark-app python -m pytest`.
- Work in `feature/*` branches created from `dev`; open pull requests into `dev`, merge `dev` into `main` at the end of a stage.
- Keep `requirements.txt` updated (`pip freeze > requirements.txt`).

---

