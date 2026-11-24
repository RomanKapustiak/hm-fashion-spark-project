"""Yura: factorization machines for both tasks.

- classification: FMClassifier  (will a customer buy within the next 4 weeks?)
- regression:     FMRegressor   (article price from its characteristics)

A factorization machine is a linear model plus all pairwise feature interactions,
where the weight of an interaction (i, j) is the dot product of two learned factor
vectors <v_i, v_j> of length `factorSize`. So interactions are learned even for
feature pairs that rarely occur together (sparse one-hot / TF-IDF features).

Notes specific to Spark's FM implementation (Spark 3.2):
- FMClassifier does not support weightCol, so the class imbalance is handled only
  by choosing the decision threshold on the validation part;
- the default solver is adamW: every step moves the parameters by roughly `stepSize`
  regardless of the gradient scale. The regression target (price) is tiny (~0.03),
  so it is standardized on train before fitting and predictions are converted back
  to the price scale before any evaluation;
- only FMClassificationModel has a training summary (objective history), so for
  both tasks the training process is also shown as train/valid metric vs maxIter.

Run:
    docker compose run --rm spark-app python src/analysis/yura_models.py [classification|regression]
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml.classification import FMClassifier  # noqa: E402
from pyspark.ml.regression import FMRegressor  # noqa: E402
from pyspark.ml.tuning import ParamGridBuilder  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.analysis.common import (  # noqa: E402
    ARTICLES,
    CUSTOMERS,
    FEATURES_COL,
    LABEL_COL,
    SEED,
    PreparedData,
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
)
from src.spark_utils import create_spark_session  # noqa: E402

MEMBER = "yura"
MODEL_NAME = "factorization_machines"

FACTOR_SIZES = [4, 8, 16]
# The regression has 4.7k mostly sparse TF-IDF features and overfits quickly
# (train RMSE ~3x lower than valid without regularization), and steps >= 0.05
# make the pairwise terms diverge there. So it gets smaller steps and stronger L2.
GRIDS = {
    "classification": {"stepSize": [0.01, 0.05, 0.1], "regParam": [0.0, 0.01, 0.1]},
    "regression": {"stepSize": [0.001, 0.003, 0.01], "regParam": [0.01, 0.1, 0.3]},
}
MAX_ITER = 100
# learning curve: the same best parameters trained for a different number of iterations
ITERATIONS = [10, 25, 50, 100, 200]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fm_grid(estimator, task: str):
    """factorSize x stepSize x regParam = 27 candidates."""
    return (
        ParamGridBuilder()
        .addGrid(estimator.factorSize, FACTOR_SIZES)
        .addGrid(estimator.stepSize, GRIDS[task]["stepSize"])
        .addGrid(estimator.regParam, GRIDS[task]["regParam"])
        .build()
    )


def _rows_with(table: pd.DataFrame, **fixed) -> pd.DataFrame:
    """Rows of the tuning table where the given parameters have the given values."""
    mask = np.ones(len(table), dtype=bool)
    for name, value in fixed.items():
        mask &= np.isclose(table[name].astype(float), float(value))
    return table[mask]


def _save_tuning_plots(tuning, estimator, task: str, label: str) -> None:
    """Training analysis on the grid: metric vs factorSize (one line per stepSize)
    and metric vs regParam (one line per stepSize), other parameters at their best values."""
    best = params_to_dict(tuning.best_params)
    by_factor = _rows_with(tuning.table, regParam=best["regParam"])
    save_plot(plot_tuning(by_factor, "factorSize", tuning.metric, hue="stepSize",
                          title=f"{label}: {tuning.metric} vs factorSize (regParam={best['regParam']})"),
              MEMBER, f"{task}__tuning_factor_size")
    by_reg = _rows_with(tuning.table, factorSize=best["factorSize"])
    save_plot(plot_tuning(by_reg, "regParam", tuning.metric, hue="stepSize",
                          title=f"{label}: {tuning.metric} vs regParam (factorSize={best['factorSize']})"),
              MEMBER, f"{task}__tuning_reg_param")


def _iteration_curve(estimator, tuning, fit_df: DataFrame, valid_df: DataFrame, task: str):
    """Train/valid metric vs maxIter for the best parameters. Full-batch adamW with a fixed
    seed is deterministic, so a shorter run is a prefix of a longer one: this is a learning curve."""
    grid = [{**tuning.best_params, estimator.maxIter: n} for n in ITERATIONS]
    return tune(estimator, grid, fit_df, valid_df, task, metric=tuning.metric)


def _plot_loss(history, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, marker=".")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (log loss)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


def _plot_threshold(thresholds: pd.DataFrame, threshold: float, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("precision", "recall", "f1"):
        ax.plot(thresholds["threshold"], thresholds[column], label=column)
    ax.axvline(threshold, color="grey", linestyle="--", label=f"best = {threshold:.2f}")
    ax.set_xlabel("threshold on P(buy)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def _save_interpretation(model, data: PreparedData, task: str, label: str) -> None:
    """Linear weights w_i and the norms of factor vectors ||v_i||.

    w_i is the feature's own (first order) effect; ||v_i|| shows how strongly the
    feature takes part in pairwise interactions. Numeric features are standardized
    and categories are 0/1, so the values are comparable between features."""
    names = data.feature_names
    linear = model.linear.toArray()
    factor_norms = np.linalg.norm(model.factors.toArray(), axis=1)
    table = pd.DataFrame({"feature": names, "linear_weight": linear, "factor_norm": factor_norms})
    table = table.reindex(table["linear_weight"].abs().sort_values(ascending=False).index)
    save_table(table, MEMBER, f"weights__{task}")
    save_plot(plot_top_weights(names, linear, f"{label}: top linear weights"), MEMBER, f"{task}__linear_weights")
    save_plot(plot_top_weights(names, factor_norms, f"{label}: interaction strength ||v_i||"),
              MEMBER, f"{task}__factor_norms")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(spark: SparkSession) -> None:
    print("\n=== Factorization machine: will the customer buy in the next 4 weeks? ===")
    task, label = CUSTOMERS.task, "FMClassifier"
    data = prepare_data(spark, CUSTOMERS, kind="linear", sample_fraction=0.25)

    # no weightCol in FMClassifier: imbalance is handled by the threshold below
    estimator = FMClassifier(featuresCol=FEATURES_COL, labelCol=LABEL_COL, maxIter=MAX_ITER, seed=SEED)

    # 1. hyper-parameter search on the (sampled) fit part, compared on the validation part
    tuning = tune(estimator, _fm_grid(estimator, task), data.fit, data.valid, task)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{MODEL_NAME}")
    _save_tuning_plots(tuning, estimator, task, label)

    # 2. learning curve: metric vs number of iterations
    curve = _iteration_curve(estimator, tuning, data.fit, data.valid, task)
    save_table(curve.table, MEMBER, f"iterations__{task}__{MODEL_NAME}")
    save_plot(plot_tuning(curve.table, "maxIter", curve.metric, title=f"{label}: {curve.metric} vs iterations"),
              MEMBER, f"{task}__iterations")

    # 3. decision threshold on validation (0.5 is far from optimal for 15.7% positives)
    threshold, thresholds = best_threshold(tuning.best_model.transform(data.valid))
    save_table(thresholds, MEMBER, f"threshold__{task}__{MODEL_NAME}")
    save_plot(_plot_threshold(thresholds, threshold, f"{label}: metrics vs threshold (validation)"),
              MEMBER, f"{task}__threshold")

    # 4. final model on the whole train part with the best parameters
    model, seconds = timed_fit(estimator, data.train, tuning.best_params)
    summary = model.summary()
    print(f"final fit: {seconds:.1f}s, {summary.totalIterations} iterations")
    save_plot(_plot_loss(summary.objectiveHistory, f"{label}: training objective (final model)"),
              MEMBER, f"{task}__loss")

    # 5. evaluation on test (once)
    predictions = model.transform(data.test)
    metrics = evaluate(predictions, task, threshold=threshold)
    print(f"test @ {threshold:.2f}: {metrics}")
    print(f"test @ 0.50: {evaluate(predictions, task)}")
    save_metrics(MEMBER, task, MODEL_NAME, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, threshold=threshold, num_features=len(data.feature_names))

    # 6. result analysis on test: ROC / precision-recall curves and the confusion matrix
    curves = save_curves(MEMBER, MODEL_NAME, predictions)
    save_plot(plot_roc_pr(curves, metrics, f"{label} on test", threshold), MEMBER, f"{task}__roc_pr")
    save_plot(plot_confusion(metrics, f"{label} on test (threshold {threshold:.2f})"),
              MEMBER, f"{task}__confusion_matrix")

    # 7. interpretation
    _save_interpretation(model, data, task, label)
    data.unpersist()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _label_scaler(train: DataFrame) -> Tuple[float, float, Callable[[DataFrame], DataFrame], Callable[[DataFrame], DataFrame]]:
    """Standardization of the target, fitted on train: (y - mean) / std.

    Returns mean, std, a function that scales the label of a frame and a function that
    converts the `prediction` column back to the price scale."""
    row = train.select(F.avg(LABEL_COL).alias("mean"), F.stddev(LABEL_COL).alias("std")).first()
    mean, std = float(row["mean"]), float(row["std"])

    def scale(df: DataFrame) -> DataFrame:
        return df.withColumn(LABEL_COL, (F.col(LABEL_COL) - F.lit(mean)) / F.lit(std))

    def unscale(predictions: DataFrame) -> DataFrame:
        return predictions.withColumn("prediction", F.col("prediction") * F.lit(std) + F.lit(mean))

    return mean, std, scale, unscale


def _rmse_to_price_scale(table: pd.DataFrame, metric: str, std: float) -> pd.DataFrame:
    """RMSE / MAE on the standardized target times std = the same metric in price units."""
    table = table.copy()
    if metric in ("rmse", "mae"):
        for column in (f"train_{metric}", f"valid_{metric}"):
            table[column] = table[column] * std
    return table


def run_regression(spark: SparkSession) -> None:
    print("\n=== Factorization machine: article price from its characteristics ===")
    task, label = ARTICLES.task, "FMRegressor"
    data = prepare_data(spark, ARTICLES, kind="linear")

    mean, std, scale, unscale = _label_scaler(data.train)
    print(f"target on train: mean={mean:.5f} std={std:.5f} (standardized for fitting)")
    fit_df, valid_df, train_df = scale(data.fit), scale(data.valid), scale(data.train)

    estimator = FMRegressor(featuresCol=FEATURES_COL, labelCol=LABEL_COL, maxIter=MAX_ITER, seed=SEED)

    # 1. hyper-parameter search (RMSE on the standardized target ranks candidates
    #    exactly like RMSE in price units; the saved table is converted to price units)
    tuning = tune(estimator, _fm_grid(estimator, task), fit_df, valid_df, task)
    tuning.table = _rmse_to_price_scale(tuning.table, tuning.metric, std)
    save_table(tuning.table, MEMBER, f"tuning__{task}__{MODEL_NAME}")
    _save_tuning_plots(tuning, estimator, task, label)

    # 2. learning curve: RMSE vs number of iterations
    curve = _iteration_curve(estimator, tuning, fit_df, valid_df, task)
    curve_table = _rmse_to_price_scale(curve.table, curve.metric, std)
    save_table(curve_table, MEMBER, f"iterations__{task}__{MODEL_NAME}")
    save_plot(plot_tuning(curve_table, "maxIter", curve.metric, title=f"{label}: {curve.metric} vs iterations"),
              MEMBER, f"{task}__iterations")

    # 3. final model on the whole train part
    model, seconds = timed_fit(estimator, train_df, tuning.best_params)
    print(f"final fit: {seconds:.1f}s")

    # 4. evaluation on test (once), predictions converted back to the price scale
    predictions = unscale(model.transform(data.test))
    metrics = evaluate(predictions, task)
    print(f"train: {evaluate(unscale(model.transform(data.train)), task)}")
    print(f"test:  {metrics}")
    save_metrics(MEMBER, task, MODEL_NAME, metrics, params_to_dict(tuning.best_params),
                 fit_seconds=seconds, num_features=len(data.feature_names))

    # 5. result analysis on test: predicted vs actual and residuals
    save_plot(regression_diagnostics(predictions, f"{label} on test: R2 = {metrics['r2']:.3f}, "
                                                  f"RMSE = {metrics['rmse']:.4f}"),
              MEMBER, f"{task}__diagnostics")

    # 6. interpretation
    _save_interpretation(model, data, task, label)
    data.unpersist()


TASKS = {"classification": run_classification, "regression": run_regression}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(TASKS)
    session = create_spark_session("HM-Models-Yura")
    try:
        for name in selected:
            TASKS[name](session)
    finally:
        session.stop()
