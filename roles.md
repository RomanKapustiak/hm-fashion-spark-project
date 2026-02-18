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
- Roman: Repository initialization, master/develop branch configuration.
- Yura: Local project structure setup and `.gitignore` configuration.
- Artem: Dataset preview — data type identification and initial feature selection.
- Taras: Dataset acquisition and local data integrity verification.
- Dmytro: Initial documentation and `roles.md` file preparation.

### 2) Configuration Stage (Common Tasks)
All team members share responsibility for environment setup and verification:
- Install Python 3.8 and required IDEs
- Configure PySpark and Java 11
- Install and verify Docker
- Build project Docker image and run the basic SparkSession smoke test

### 3) Extraction Stage
- Taras: StructType schema for article identification/hierarchy (`articles.csv`).
- Yura: StructType schema for visual/descriptive attributes (`articles.csv`).
- Artem: StructType schema for customer metadata (`customers.csv`).
- Roman: StructType schema for transaction records (`transactions_train.csv`).
- Dmytro: Implement `data_loader` module and centralized DataFrame readers.

### 4) Data Preprocessing Stage
- Taras: Article cleaning (duplicates, case normalization).
- Yura: Description cleaning (null handling, color normalization).
- Artem: Customer cleaning (age imputation, status normalization).
- Roman: Financial validation (price outliers, type casting).
- Dmytro: Statistical analysis and removal of non-informative features.

### 5) Transformation Stage
Each member implements 6 business queries and provides an execution plan analysis (`.explain()`):
- Taras: Product hierarchy & department queries
- Yura: Visual trends & design feature queries
- Artem: Customer demographics & behavior queries
- Roman: Financial performance & revenue queries
- Dmytro: Time-series dynamics & retention queries

Requirements per member: at least 3 filters, 2 joins, 2 group-bys, 2 window functions, plus query plan analysis.

### 6) Results Writing Stage
- Taras: Export product analytics to CSV
- Yura: Export design/trend analytics to CSV
- Artem: Export customer segmentation to CSV
- Roman: Export financial reports and sales-channel analytics to CSV
- Dmytro: Validate pipeline outputs and ensure CSV storage in Docker

### 7) Presentation Stage
- Team presentation of findings and pipeline demonstration (shared responsibility).

---

## Development & Coding Standards (brief)
- Follow PEP 8 and include type hints for public functions.
- Use `pytest` for unit tests and place tests in `tests/`.
- Use the local virtual environment: `source venv/bin/activate`.
- Use `./venv/bin/python` if activation is not persistent.
- Keep `requirements.txt` updated (`pip freeze > requirements.txt`).

---

