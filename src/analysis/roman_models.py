"""Roman: linear models (baseline family) for both tasks.

- classification: LogisticRegression  (will a customer buy within the next 4 weeks?)
- regression:     LinearRegression    (article price from its characteristics)

This file is also the reference template for the other members' model scripts.

Run:
    docker compose run --rm spark-app python src/analysis/roman_models.py [classification|regression]
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from pyspark.ml.classification import LogisticRegression  # noqa: E402
from pyspark.ml.regression import LinearRegression  # noqa: E402
from pyspark.ml.tuning import ParamGridBuilder  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from src.analysis.common import (  # noqa: E402
    ARTICLES,
    CUSTOMERS,
    FEATURES_COL,
    LABEL_COL,
    WEIGHT_COL,
    best_threshold,
    evaluate,
    params_to_dict,
    plot_top_weights,
    plot_tuning,
    prepare_data,
    save_metrics,
    save_plot,
    save_table,
    timed_fit,
    tune,
    weights_table,
)
from src.spark_utils import create_spark_session  # noqa: E402

MEMBER = "roman"
REG_PARAMS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
ELASTIC_NET = [0.0, 0.5, 1.0]


def _linear_grid(estimator):
    """regParam x elasticNetParam; with regParam=0 the mixing parameter has no effect,
    so only one such candidate is kept."""
    grid = (
        ParamGridBuilder()
        .addGrid(estimator.regParam, REG_PARAMS)
        .addGrid(estimator.elasticNetParam, ELASTIC_NET)
        .build()
    )
    return [p for p in grid if p[estimator.regParam] > 0 or p[estimator.elasticNetParam] == 0.0]


def _plot_loss(history, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, marker=".")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (loss + regularization)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


def run_classification(spark: SparkSession) -> None:
    print("\n=== Logistic regression: will the customer buy in the next 4 weeks? ===")
    task, model_name = CUSTOMERS.task, "logistic_regression"
    data = prepare_data(spark, CUSTOMERS, kind="linear", sample_fraction=0.3)

    estimator = LogisticRegression(
        featuresCol=FEATURES_COL, labelCol=LABEL_COL, weightCol=WEIGHT_COL, maxIter=200, tol=1e-6,
    )

    # 1. hyper-parameter search on the fit part, compared on the validation part
    tuning = tune(estimator, _linear_grid(estimator), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(tuning.table, "regParam", tuning.metric, hue="elasticNetParam", log_x=True,
                          title="LogisticRegression: PR AUC vs regularization"),
              MEMBER, f"{task}__tuning")

    # 2. decision threshold: classes are imbalanced, so 0.5 is not optimal for F1
    threshold, threshold_table = best_threshold(tuning.best_model.transform(data.valid))
    save_table(threshold_table, MEMBER, f"threshold__{task}__{model_name}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("precision", "recall", "f1"):
        ax.plot(threshold_table["threshold"], threshold_table[column], label=column)
    ax.axvline(threshold, color="grey", linestyle="--", label=f"best = {threshold:.2f}")
    ax.set_xlabel("threshold on P(buy)")
    ax.set_title("LogisticRegression: metrics vs threshold (validation)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, MEMBER, f"{task}__threshold")

    # 3. final model on the whole train part with the best parameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    history = model.summary.objectiveHistory
    print(f"final fit: {seconds:.1f}s, {model.summary.totalIterations} iterations")
    save_plot(_plot_loss(history, "LogisticRegression: training objective"), MEMBER, f"{task}__loss")

    # 4. evaluation on test (once)
    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task, threshold=threshold)
    default_metrics = evaluate(predictions, task)
    print(f"test @ {threshold:.2f}: {metrics}")
    print(f"test @ 0.50: {default_metrics}")
    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, threshold=threshold, num_features=len(data.feature_names))

    # 5. interpretation: the largest coefficients (numeric features are standardized, categories are 0/1)
    coefficients = model.coefficients.toArray()
    save_table(weights_table(data.feature_names, coefficients, "coefficient"), MEMBER, f"coefficients__{task}")
    save_plot(plot_top_weights(data.feature_names, coefficients, "LogisticRegression: top coefficients"),
              MEMBER, f"{task}__coefficients")
    data.unpersist()


def run_regression(spark: SparkSession) -> None:
    print("\n=== Linear regression: article price from its characteristics ===")
    task, model_name = ARTICLES.task, "linear_regression"
    data = prepare_data(spark, ARTICLES, kind="linear")

    estimator = LinearRegression(featuresCol=FEATURES_COL, labelCol=LABEL_COL, maxIter=200, tol=1e-6)

    tuning = tune(estimator, _linear_grid(estimator), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(tuning.table, "regParam", tuning.metric, hue="elasticNetParam", log_x=True,
                          title="LinearRegression: RMSE vs regularization"),
              MEMBER, f"{task}__tuning")

    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, {model.summary.totalIterations} iterations")
    if len(model.summary.objectiveHistory) > 1:
        save_plot(_plot_loss(model.summary.objectiveHistory, "LinearRegression: training objective"),
                  MEMBER, f"{task}__loss")

    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task)
    train_metrics = evaluate(model.transform(data.train), task)
    print(f"train: {train_metrics}")
    print(f"test:  {metrics}")
    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, num_features=len(data.feature_names))

    # predicted vs actual on test
    sample = predictions.select(LABEL_COL, "prediction").toPandas()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(sample[LABEL_COL], sample["prediction"], s=3, alpha=0.3)
    upper = float(sample[LABEL_COL].quantile(0.995))
    ax.plot([0, upper], [0, upper], color="grey", linestyle="--")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("actual average price")
    ax.set_ylabel("predicted")
    ax.set_title(f"LinearRegression on test: R2={metrics['r2']:.3f}")
    save_plot(fig, MEMBER, f"{task}__predicted_vs_actual")

    coefficients = model.coefficients.toArray()
    save_table(weights_table(data.feature_names, coefficients, "coefficient"), MEMBER, f"coefficients__{task}")
    save_plot(plot_top_weights(data.feature_names, coefficients, "LinearRegression: top coefficients"),
              MEMBER, f"{task}__coefficients")
    data.unpersist()


TASKS = {"classification": run_classification, "regression": run_regression}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(TASKS)
    session = create_spark_session("HM-Models-Roman")
    try:
        for name in selected:
            TASKS[name](session)
    finally:
        session.stop()
