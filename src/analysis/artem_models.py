"""Artem: Random forest models for both tasks.

- classification: RandomForestClassifier (will a customer buy within the next 4 weeks?)
- regression:     RandomForestRegressor  (article price from its characteristics)

Run:
    docker compose run --rm spark-app python src/analysis/artem_models.py
    docker compose run --rm spark-app python src/analysis/artem_models.py [classification|regression]
"""

from __future__ import annotations

import gc
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from pyspark.ml.classification import RandomForestClassifier  # noqa: E402
from pyspark.ml.regression import RandomForestRegressor  # noqa: E402
from pyspark.ml.tuning import ParamGridBuilder  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from src.analysis.common import (  # noqa: E402
    ARTICLES,
    CUSTOMERS,
    FEATURES_COL,
    LABEL_COL,
    SEED,
    WEIGHT_COL,
    best_threshold,
    evaluate,
    params_to_dict,
    plot_confusion,
    plot_roc_pr,
    plot_top_weights,
    plot_tuning,
    prepare_data,
    regression_diagnostics,
    save_curves,
    save_metrics,
    save_plot,
    save_table,
    timed_fit,
    tune,
    weights_table,
)
from src.spark_utils import create_spark_session  # noqa: E402

MEMBER = "artem"
MODEL_NAME = "random_forest"

# Hyperparameter search grid: numTrees x maxDepth with subsamplingRate=0.8
NUM_TREES = [20, 50]
MAX_DEPTHS = [6, 8, 10]
SUBSAMPLING_RATE = 0.8

# Quick grid for fast smoke tests or debugging (activated via FAST_DEV=1)
FAST_NUM_TREES = [10, 20]
FAST_MAX_DEPTHS = [4, 6]


def _forest_grid(estimator, max_categories, fast: bool = False):
    """Grid over numTrees x maxDepth with subsamplingRate and maxBins properly configured."""
    bins = max(32, max_categories)
    trees = FAST_NUM_TREES if fast else NUM_TREES
    depths = FAST_MAX_DEPTHS if fast else MAX_DEPTHS

    return (
        ParamGridBuilder()
        .addGrid(estimator.numTrees, trees)
        .addGrid(estimator.maxDepth, depths)
        .addGrid(estimator.subsamplingRate, [SUBSAMPLING_RATE])
        .addGrid(estimator.maxBins, [bins])
        .build()
    )


def _subsample_filtered_table(tuning, estimator):
    """Filter tuning table to rows matching the best subsamplingRate,
    ensuring each (maxDepth, numTrees) coordinate is unique for plotting."""
    if estimator.subsamplingRate in tuning.best_params:
        best_rate = tuning.best_params[estimator.subsamplingRate]
        return tuning.table[tuning.table["subsamplingRate"] == best_rate]
    return tuning.table


def _total_nodes(model) -> int:
    """Safely return total number of nodes in the forest across PySpark versions."""
    return getattr(model, "totalNumNodes", sum(t.numNodes for t in model.trees))


def run_classification(spark: SparkSession, fast: bool = False) -> None:
    print("\n=== Random Forest: will the customer buy in the next 4 weeks? ===")
    task, model_name = CUSTOMERS.task, MODEL_NAME
    data = prepare_data(spark, CUSTOMERS, kind="tree", sample_fraction=0.2)

    estimator = RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxBins=max(32, data.max_categories),
        subsamplingRate=SUBSAMPLING_RATE,
        seed=SEED,
    )

    # 1. Hyper-parameter search on fit subset, evaluated on validation subset
    tuning = tune(estimator, _forest_grid(estimator, data.max_categories, fast=fast), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")

    filtered_table = _subsample_filtered_table(tuning, estimator)
    save_plot(
        plot_tuning(
            filtered_table, "maxDepth", tuning.metric, hue="numTrees",
            title="RandomForestClassifier: PR AUC vs depth",
        ),
        MEMBER, f"{task}__tuning_depth",
    )
    save_plot(
        plot_tuning(
            filtered_table, "numTrees", tuning.metric, hue="maxDepth",
            title="RandomForestClassifier: PR AUC vs numTrees",
        ),
        MEMBER, f"{task}__tuning_trees",
    )

    # 2. Decision threshold: optimize threshold on validation for maximum F1
    valid_preds = tuning.best_model.transform(data.valid).select(LABEL_COL, "probability").localCheckpoint(eager=True)
    threshold, thresholds = best_threshold(valid_preds)
    save_table(thresholds, MEMBER, f"threshold__{task}__{model_name}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("precision", "recall", "f1"):
        ax.plot(thresholds["threshold"], thresholds[column], label=column)
    ax.axvline(threshold, color="grey", linestyle="--", label=f"best = {threshold:.2f}")
    ax.set_xlabel("threshold on P(buy)")
    ax.set_title("RandomForestClassifier: metrics vs threshold (validation)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, MEMBER, f"{task}__threshold")

    # 3. Final model on the full train split with the best hyperparameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, {model.getNumTrees} trees, {_total_nodes(model)} total nodes")

    # 4. Evaluation on test set (once)
    predictions = model.transform(data.test).localCheckpoint(eager=True)
    metrics = evaluate(predictions, task, threshold=threshold)
    default_metrics = evaluate(predictions, task)
    print(f"test @ {threshold:.2f}: {metrics}")
    print(f"test @ 0.50: {default_metrics}")
    save_metrics(
        MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
        fit_seconds=seconds, threshold=threshold, num_features=len(data.feature_names),
    )

    # 5. Diagnostic curves & confusion matrix on test
    curves = save_curves(MEMBER, model_name, predictions)
    save_plot(plot_roc_pr(curves, metrics, "RandomForestClassifier on test", threshold), MEMBER, f"{task}__roc_pr")
    save_plot(
        plot_confusion(metrics, f"RandomForestClassifier on test (threshold {threshold:.2f})"),
        MEMBER, f"{task}__confusion_matrix",
    )

    # 6. Feature importances
    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(
        plot_top_weights(data.feature_names, importances, "RandomForestClassifier: feature importances"),
        MEMBER, f"{task}__importances",
    )
    data.unpersist()
    gc.collect()


def run_regression(spark: SparkSession, fast: bool = False) -> None:
    print("\n=== Random Forest: article price from its characteristics ===")
    task, model_name = ARTICLES.task, MODEL_NAME
    data = prepare_data(spark, ARTICLES, kind="tree")

    estimator = RandomForestRegressor(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxBins=max(32, data.max_categories),
        subsamplingRate=SUBSAMPLING_RATE,
        seed=SEED,
    )

    # 1. Hyper-parameter search on fit subset, evaluated on validation subset
    tuning = tune(estimator, _forest_grid(estimator, data.max_categories, fast=fast), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")

    filtered_table = _subsample_filtered_table(tuning, estimator)
    save_plot(
        plot_tuning(
            filtered_table, "maxDepth", tuning.metric, hue="numTrees",
            title="RandomForestRegressor: RMSE vs depth",
        ),
        MEMBER, f"{task}__tuning_depth",
    )
    save_plot(
        plot_tuning(
            filtered_table, "numTrees", tuning.metric, hue="maxDepth",
            title="RandomForestRegressor: RMSE vs numTrees",
        ),
        MEMBER, f"{task}__tuning_trees",
    )

    # 2. Final model on the full train split with the best hyperparameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, {model.getNumTrees} trees, {_total_nodes(model)} total nodes")

    # 3. Evaluation on test set (once)
    predictions = model.transform(data.test).localCheckpoint(eager=True)
    metrics = evaluate(predictions, task)
    train_preds = model.transform(data.train).localCheckpoint(eager=True)
    train_metrics = evaluate(train_preds, task)
    print(f"train: {train_metrics}")
    print(f"test:  {metrics}")
    save_metrics(
        MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
        fit_seconds=seconds, num_features=len(data.feature_names),
    )

    # 4. Residual diagnostics (predicted vs actual & residual distribution)
    save_plot(
        regression_diagnostics(
            predictions,
            f"RandomForestRegressor on test: R2 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.4f}",
        ),
        MEMBER, f"{task}__diagnostics",
    )

    # 5. Feature importances
    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(
        plot_top_weights(data.feature_names, importances, "RandomForestRegressor: feature importances"),
        MEMBER, f"{task}__importances",
    )
    data.unpersist()
    gc.collect()


TASKS = {"classification": run_classification, "regression": run_regression}

if __name__ == "__main__":
    is_fast = os.environ.get("FAST_DEV", "0").lower() in ("1", "true", "yes")
    args = sys.argv[1:]
    selected = [a for a in args if a in TASKS] or list(TASKS)

    session = create_spark_session("HM-Models-Artem")
    try:
        for name in selected:
            TASKS[name](session, fast=is_fast)
    finally:
        session.stop()
