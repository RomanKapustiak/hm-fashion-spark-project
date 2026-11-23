"""Taras: decision trees for both tasks.

- classification: DecisionTreeClassifier  (will a customer buy within the next 4 weeks?)
- regression:     DecisionTreeRegressor   (article price from its characteristics)

Run:
    docker compose run --rm spark-app python src/analysis/taras_models.py [classification|regression]
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from pyspark.ml.classification import DecisionTreeClassifier  # noqa: E402
from pyspark.ml.regression import DecisionTreeRegressor  # noqa: E402
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

MEMBER = "taras"
MAX_DEPTHS = [4, 8, 12, 16, 20]
MIN_INSTANCES = [1, 200]


def _tree_grid(estimator, max_categories):
    bins = max(32, max_categories)
    return (
        ParamGridBuilder()
        .addGrid(estimator.maxDepth, MAX_DEPTHS)
        .addGrid(estimator.minInstancesPerNode, MIN_INSTANCES)
        .addGrid(estimator.maxBins, [bins, 2 * bins])
        .build()
    )


def _depth_table(tuning, estimator):
    """Rows for the depth plot: maxBins fixed at its best value."""
    return tuning.table[tuning.table["maxBins"] == tuning.best_params[estimator.maxBins]]


def run_classification(spark: SparkSession) -> None:
    print("\n=== Decision tree: will the customer buy in the next 4 weeks? ===")
    task, model_name = CUSTOMERS.task, "decision_tree"
    data = prepare_data(spark, CUSTOMERS, kind="tree", sample_fraction=0.3)

    estimator = DecisionTreeClassifier(
        featuresCol=FEATURES_COL, labelCol=LABEL_COL, weightCol=WEIGHT_COL,
        maxBins=max(32, data.max_categories), seed=SEED,
    )

    # 1. hyper-parameter search on the fit part, compared on the validation part
    tuning = tune(estimator, _tree_grid(estimator, data.max_categories), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(_depth_table(tuning, estimator), "maxDepth", tuning.metric, hue="minInstancesPerNode",
                          title="DecisionTreeClassifier: PR AUC vs depth"),
              MEMBER, f"{task}__tuning")

    # 2. decision threshold: classes are imbalanced, so 0.5 is not optimal for F1
    threshold, thresholds = best_threshold(tuning.best_model.transform(data.valid))
    save_table(thresholds, MEMBER, f"threshold__{task}__{model_name}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("precision", "recall", "f1"):
        ax.plot(thresholds["threshold"], thresholds[column], label=column)
    ax.axvline(threshold, color="grey", linestyle="--", label=f"best = {threshold:.2f}")
    ax.set_xlabel("threshold on P(buy)")
    ax.set_title("DecisionTreeClassifier: metrics vs threshold (validation)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, MEMBER, f"{task}__threshold")

    # 3. final model on the whole train part with the best parameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, depth {model.depth}, {model.numNodes} nodes")

    # 4. evaluation on test (once)
    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task, threshold=threshold)
    print(f"test @ {threshold:.2f}: {metrics}")
    print(f"test @ 0.50: {evaluate(predictions, task)}")
    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, threshold=threshold, num_features=len(data.feature_names))

    # 5. result analysis on test: ROC / precision-recall curves and the confusion matrix
    curves = save_curves(MEMBER, model_name, predictions)
    save_plot(plot_roc_pr(curves, metrics, "DecisionTreeClassifier on test", threshold), MEMBER, f"{task}__roc_pr")
    save_plot(plot_confusion(metrics, f"DecisionTreeClassifier on test (threshold {threshold:.2f})"),
              MEMBER, f"{task}__confusion_matrix")

    # 6. interpretation: feature importances
    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(plot_top_weights(data.feature_names, importances, "DecisionTreeClassifier: feature importances"),
              MEMBER, f"{task}__importances")
    data.unpersist()


def run_regression(spark: SparkSession) -> None:
    print("\n=== Decision tree: article price from its characteristics ===")
    task, model_name = ARTICLES.task, "decision_tree"
    data = prepare_data(spark, ARTICLES, kind="tree")

    # maxBins >= number of categories (248 departments), otherwise Spark fails
    estimator = DecisionTreeRegressor(featuresCol=FEATURES_COL, labelCol=LABEL_COL,
                                      maxBins=max(32, data.max_categories), seed=SEED)

    tuning = tune(estimator, _tree_grid(estimator, data.max_categories), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(_depth_table(tuning, estimator), "maxDepth", tuning.metric, hue="minInstancesPerNode",
                          title="DecisionTreeRegressor: RMSE vs depth"),
              MEMBER, f"{task}__tuning")

    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, depth {model.depth}, {model.numNodes} nodes")

    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task)
    print(f"train: {evaluate(model.transform(data.train), task)}")
    print(f"test:  {metrics}")
    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, num_features=len(data.feature_names))

    # result analysis on test: predicted vs actual and residuals
    save_plot(regression_diagnostics(predictions, f"DecisionTreeRegressor on test: R2 = {metrics['r2']:.3f}, "
                                                  f"RMSE = {metrics['rmse']:.4f}"),
              MEMBER, f"{task}__diagnostics")

    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(plot_top_weights(data.feature_names, importances, "DecisionTreeRegressor: feature importances"),
              MEMBER, f"{task}__importances")
    data.unpersist()


TASKS = {"classification": run_classification, "regression": run_regression}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(TASKS)
    session = create_spark_session("HM-Models-Taras")
    try:
        for name in selected:
            TASKS[name](session)
    finally:
        session.stop()
