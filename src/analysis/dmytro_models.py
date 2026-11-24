"""Dmytro: gradient-boosted trees for both tasks.

- classification: GBTClassifier  (will a customer buy within the next 4 weeks?)
- regression:     GBTRegressor   (article price from its characteristics)

Run:
    docker compose run --rm spark-app python src/analysis/dmytro_models.py [classification|regression]
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml.classification import GBTClassificationModel, GBTClassifier  # noqa: E402
from pyspark.ml.regression import GBTRegressionModel, GBTRegressor  # noqa: E402
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

MEMBER = "dmytro"
MAX_DEPTHS = [3, 5]
STEP_SIZES = [0.1, 0.2]
MAX_ITER = {"classification": [20, 40], "regression": [20, 40]}
SUBSAMPLING_RATE = 0.8


def _gbt_grid(estimator, task):
    return (
        ParamGridBuilder()
        .addGrid(estimator.maxDepth, MAX_DEPTHS)
        .addGrid(estimator.stepSize, STEP_SIZES)
        .addGrid(estimator.maxIter, MAX_ITER[task])
        .build()
    )


def _best_iter_table(tuning, estimator):
    """Rows for the depth / step plot: maxIter fixed at its best value."""
    return tuning.table[tuning.table["maxIter"] == tuning.best_params[estimator.maxIter]]


def _plot_iterations(curve: pd.DataFrame, title: str, ylabel: str):
    """Training process of boosting: error after each added tree, train vs validation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["iteration"], curve["train"], label="train (fit part)")
    ax.plot(curve["iteration"], curve["valid"], label="validation")
    best = curve.loc[curve["valid"].idxmin()]
    ax.axvline(best["iteration"], color="grey", linestyle="--", label=f"min valid at {int(best['iteration'])}")
    ax.set_xlabel("number of trees (iteration)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def _iteration_curve(model, fit_df, valid_df, loss=None) -> pd.DataFrame:
    args = (loss,) if loss else ()
    train = model.evaluateEachIteration(fit_df, *args)
    valid = model.evaluateEachIteration(valid_df, *args)
    return pd.DataFrame({"iteration": range(1, len(train) + 1), "train": train, "valid": valid})


def run_classification(spark: SparkSession) -> None:
    print("\n=== Gradient-boosted trees: will the customer buy in the next 4 weeks? ===")
    task, model_name = CUSTOMERS.task, "gbt"
    data = prepare_data(spark, CUSTOMERS, kind="tree", sample_fraction=0.2)

    estimator = GBTClassifier(
        featuresCol=FEATURES_COL, labelCol=LABEL_COL, weightCol=WEIGHT_COL,
        maxBins=max(32, data.max_categories), subsamplingRate=SUBSAMPLING_RATE, seed=SEED,
    )

    # 1. hyper-parameter search on the fit part, compared on the validation part
    tuning = tune(estimator, _gbt_grid(estimator, task), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(_best_iter_table(tuning, estimator), "maxDepth", tuning.metric, hue="stepSize",
                          title="GBTClassifier: PR AUC vs depth"),
              MEMBER, f"{task}__tuning")

    candidate = tuning.best_model
    if not isinstance(candidate, GBTClassificationModel):
        raise TypeError("Expected a fitted GBT classifier")

    # 2. training process: Spark logistic loss after every tree (fit vs valid)
    curve = _iteration_curve(candidate, data.fit, data.valid)
    save_table(curve, MEMBER, f"iterations__{task}__{model_name}")
    save_plot(_plot_iterations(curve, "GBTClassifier: logistic loss by iteration", "logistic loss"),
              MEMBER, f"{task}__iterations")

    # 3. Pick the F1 threshold on validation, before evaluating test predictions.
    threshold, thresholds = best_threshold(candidate.transform(data.valid))
    save_table(thresholds, MEMBER, f"threshold__{task}__{model_name}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("precision", "recall", "f1"):
        ax.plot(thresholds["threshold"], thresholds[column], label=column)
    ax.axvline(threshold, color="grey", linestyle="--", label=f"best = {threshold:.2f}")
    ax.set_xlabel("threshold on P(buy)")
    ax.set_title("GBTClassifier: metrics vs threshold (validation)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, MEMBER, f"{task}__threshold")

    # 4. final model on the whole train part with the best parameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, {model.getNumTrees} trees")

    # 5. evaluation on test (once)
    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task, threshold=threshold)
    print(f"test @ {threshold:.2f}: {metrics}")

    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, threshold=threshold, num_features=len(data.feature_names))

    # 6. result analysis on test: ROC / precision-recall curves and the confusion matrix
    curves = save_curves(MEMBER, model_name, predictions)
    save_plot(plot_roc_pr(curves, metrics, "GBTClassifier on test", threshold), MEMBER, f"{task}__roc_pr")
    save_plot(plot_confusion(metrics, f"GBTClassifier on test (threshold {threshold:.2f})"),
              MEMBER, f"{task}__confusion_matrix")

    # 7. interpretation: feature importances
    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(plot_top_weights(data.feature_names, importances, "GBTClassifier: feature importances"),
              MEMBER, f"{task}__importances")
    data.unpersist()


def run_regression(spark: SparkSession) -> None:
    print("\n=== Gradient-boosted trees: article price from its characteristics ===")
    task, model_name = ARTICLES.task, "gbt"
    data = prepare_data(spark, ARTICLES, kind="tree")

    # maxBins >= number of categories (248 departments), otherwise Spark fails
    estimator = GBTRegressor(featuresCol=FEATURES_COL, labelCol=LABEL_COL,
                             maxBins=max(32, data.max_categories), subsamplingRate=SUBSAMPLING_RATE, seed=SEED)

    tuning = tune(estimator, _gbt_grid(estimator, task), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{model_name}")
    save_plot(plot_tuning(_best_iter_table(tuning, estimator), "maxDepth", tuning.metric, hue="stepSize",
                          title="GBTRegressor: RMSE vs depth"),
              MEMBER, f"{task}__tuning")

    candidate = tuning.best_model
    if not isinstance(candidate, GBTRegressionModel):
        raise TypeError("Expected a fitted GBT regressor")
    curve = _iteration_curve(candidate, data.fit, data.valid, loss="squared")
    save_table(curve, MEMBER, f"iterations__{task}__{model_name}")
    save_plot(_plot_iterations(curve, "GBTRegressor: squared error by iteration", "squared error"),
              MEMBER, f"{task}__iterations")

    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    print(f"final fit: {seconds:.1f}s, {model.getNumTrees} trees")

    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task)
    print(f"train: {evaluate(model.transform(data.train), task)}")
    print(f"test:  {metrics}")
    save_metrics(MEMBER, task, model_name, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, num_features=len(data.feature_names))

    # result analysis on test: predicted vs actual and residuals
    save_plot(regression_diagnostics(predictions, f"GBTRegressor on test: R2 = {metrics['r2']:.3f}, "
                                                  f"RMSE = {metrics['rmse']:.4f}"),
              MEMBER, f"{task}__diagnostics")

    importances = model.featureImportances.toArray()
    save_table(weights_table(data.feature_names, importances, "importance"), MEMBER, f"importances__{task}")
    save_plot(plot_top_weights(data.feature_names, importances, "GBTRegressor: feature importances"),
              MEMBER, f"{task}__importances")
    data.unpersist()


TASKS = {"classification": run_classification, "regression": run_regression}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(TASKS)
    session = create_spark_session("HM-Models-Dmytro")
    try:
        for name in selected:
            TASKS[name](session)
    finally:
        session.stop()
