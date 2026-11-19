"""Shared building blocks for the data analysis (modelling) stage.

Every member's model script uses these helpers, so that all models are trained
on the same feature tables, with the same train/test split, and are evaluated
with the same metrics. This is what makes the final comparison fair.

Typical flow inside a member script:

    data = prepare_data(spark, CUSTOMERS, kind="linear", sample_fraction=0.25)
    tuning = tune(estimator, grid, data.fit, data.valid, data.spec.task)
    final_model = estimator.copy(tuning.best_params).fit(data.train)
    metrics = evaluate(final_model.transform(data.test), data.spec.task)
    save_metrics(MEMBER, data.spec.task, "logistic_regression", metrics, ...)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark import StorageLevel  # noqa: E402
from pyspark.ml import Estimator, Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator  # noqa: E402
from pyspark.ml.feature import (  # noqa: E402
    HashingTF,
    IDF,
    Imputer,
    OneHotEncoder,
    RegexTokenizer,
    StandardScaler,
    StopWordsRemover,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.param import Param  # noqa: E402
from pyspark.sql import Column, DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

SEED = 42
ML_DIR = "data/ml"
OUTPUT_DIR = "output/ml"

FEATURES_COL = "features"
LABEL_COL = "label"
WEIGHT_COL = "class_weight"
SPLIT_COL = "split"

CLASSIFICATION = "classification"
REGRESSION = "regression"

TEST_FRACTION = 0.2
VALID_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Dataset descriptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Describes one feature table in data/ml/.

    Numeric feature columns are not listed: every column that is not an id,
    label, split, group, categorical, text or excluded column is treated as a
    numeric feature. So a new feature added to the builder is picked up by all
    models automatically.
    """

    name: str
    task: str
    id_col: str
    group_col: str
    categorical: Tuple[str, ...]
    text: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return os.path.join(ML_DIR, self.name)

    def non_numeric(self) -> set:
        return {
            self.id_col, self.group_col, LABEL_COL, SPLIT_COL, WEIGHT_COL,
            *self.categorical, *self.text, *self.exclude,
        }


# Classification: will the customer buy anything during the next 4 weeks?
CUSTOMERS = DatasetSpec(
    name="customers_features",
    task=CLASSIFICATION,
    id_col="customer_id",
    group_col="customer_id",
    categorical=("club_member_status", "fashion_news_frequency"),
)

# Regression: average selling price of an article from its attributes.
ARTICLES = DatasetSpec(
    name="articles_features",
    task=REGRESSION,
    id_col="article_id",
    group_col="product_code",
    categorical=(
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    ),
    text=("text",),
)


# ---------------------------------------------------------------------------
# Deterministic splitting
# ---------------------------------------------------------------------------

def hash_bucket(key_col: str, salt: str, buckets: int = 1000) -> Column:
    """Stable bucket in [0, buckets) for a key. Same key -> same bucket on every run,
    independent of partitioning, unlike randomSplit."""
    hashed = F.xxhash64(F.col(key_col).cast("string"), F.lit(salt))
    return ((hashed % buckets) + buckets) % buckets  # non-negative modulo (F.pmod appears only in Spark 3.4)


def assign_split(df: DataFrame, key_col: str, test_fraction: float = TEST_FRACTION) -> DataFrame:
    """Adds the shared train/test split column. Used by the feature builders only."""
    return df.withColumn(
        SPLIT_COL,
        F.when(hash_bucket(key_col, "test") < int(test_fraction * 1000), F.lit("test")).otherwise(F.lit("train")),
    )


# ---------------------------------------------------------------------------
# Loading and feature preparation
# ---------------------------------------------------------------------------

def load_dataset(spark: SparkSession, spec: DatasetSpec) -> DataFrame:
    return spark.read.parquet(spec.path)


def numeric_columns(df: DataFrame, spec: DatasetSpec) -> List[str]:
    skip = spec.non_numeric()
    return [name for name, dtype in df.dtypes if name not in skip and dtype in ("double", "float", "int", "bigint")]


def class_weights(train: DataFrame) -> Dict[int, float]:
    """Balanced class weights: n / (2 * n_class). Computed on train only."""
    counts = {int(r[LABEL_COL]): r["count"] for r in train.groupBy(LABEL_COL).count().collect()}
    total = sum(counts.values())
    return {label: total / (len(counts) * n) for label, n in counts.items()}


def build_preprocessing(
    spec: DatasetSpec,
    numeric_cols: Sequence[str],
    kind: str = "linear",
    include_text: Optional[bool] = None,
    text_features: int = 1 << 12,
) -> Pipeline:
    """Feature preparation pipeline.

    kind="linear": median imputation -> standard scaling for numeric columns,
                   one-hot encoding for categories, TF-IDF for text.
                   For LinearRegression / LogisticRegression / FM models.
    kind="tree":   median imputation, categories as indices (trees split on them
                   natively), no scaling, no text by default.
                   For DecisionTree / RandomForest / GBT models.
    """
    if kind not in ("linear", "tree"):
        raise ValueError(f"unknown kind: {kind}")
    if include_text is None:
        include_text = kind == "linear"

    stages = []
    imputed = [f"{c}__imp" for c in numeric_cols]
    stages.append(Imputer(inputCols=list(numeric_cols), outputCols=imputed, strategy="median"))

    indexed = [f"{c}__idx" for c in spec.categorical]
    if spec.categorical:
        stages.append(StringIndexer(
            inputCols=list(spec.categorical), outputCols=indexed,
            handleInvalid="keep", stringOrderType="frequencyDesc",
        ))

    if kind == "linear":
        stages.append(VectorAssembler(inputCols=imputed, outputCol="numeric_vec"))
        stages.append(StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled", withMean=True, withStd=True))
        parts = ["numeric_scaled"]
        if spec.categorical:
            one_hot = [f"{c}__ohe" for c in spec.categorical]
            stages.append(OneHotEncoder(inputCols=indexed, outputCols=one_hot, handleInvalid="keep"))
            parts += one_hot
    else:
        parts = imputed + indexed

    if include_text:
        for column in spec.text:
            stages += [
                RegexTokenizer(inputCol=column, outputCol=f"{column}__tokens", pattern="[^a-z]+", minTokenLength=2),
                StopWordsRemover(inputCol=f"{column}__tokens", outputCol=f"{column}__words"),
                HashingTF(inputCol=f"{column}__words", outputCol=f"{column}__tf", numFeatures=text_features),
                IDF(inputCol=f"{column}__tf", outputCol=f"{column}__tfidf", minDocFreq=5),
            ]
            parts.append(f"{column}__tfidf")

    stages.append(VectorAssembler(inputCols=parts, outputCol=FEATURES_COL))
    return Pipeline(stages=stages)


@dataclass
class PreparedData:
    """Everything a member script needs. All frames have columns:
    id, label, [class_weight], features."""

    spec: DatasetSpec
    kind: str
    preprocessing: PipelineModel
    train: DataFrame   # full train part (final fit)
    test: DataFrame    # test part (final evaluation only!)
    fit: DataFrame     # part of train used to fit candidates during tuning (optionally sampled)
    valid: DataFrame   # part of train used to compare candidates / pick a threshold
    feature_names: List[str]
    weights: Optional[Dict[int, float]] = None

    @property
    def max_categories(self) -> int:
        """Largest number of categories among indexed columns. Tree models need maxBins >= this."""
        sizes = [len(labels) + 1 for stage in self.preprocessing.stages
                 if hasattr(stage, "labelsArray") for labels in stage.labelsArray]
        return max(sizes) if sizes else 2

    def unpersist(self) -> None:
        for frame in (self.train, self.test):
            frame.unpersist()


def prepare_data(
    spark: SparkSession,
    spec: DatasetSpec,
    kind: str = "linear",
    sample_fraction: Optional[float] = None,
    valid_fraction: float = VALID_FRACTION,
    include_text: Optional[bool] = None,
    text_features: int = 1 << 12,
) -> PreparedData:
    """Loads a feature table, applies the shared split and the shared preprocessing.

    The preprocessing is fitted on train only (no information from test leaks in).
    sample_fraction limits the `fit` part used for hyper-parameter search on heavy
    models; the final model should still be trained on `train`.
    """
    df = load_dataset(spark, spec)
    num_cols = numeric_columns(df, spec)
    train_raw = df.filter(F.col(SPLIT_COL) == "train")
    test_raw = df.filter(F.col(SPLIT_COL) == "test")

    weights = None
    keep = [spec.id_col, spec.group_col, LABEL_COL, FEATURES_COL]
    if spec.task == CLASSIFICATION:
        weights = class_weights(train_raw)
        weight_expr = F.when(F.col(LABEL_COL) == 1, F.lit(weights[1])).otherwise(F.lit(weights[0]))
        train_raw = train_raw.withColumn(WEIGHT_COL, weight_expr)
        test_raw = test_raw.withColumn(WEIGHT_COL, weight_expr)
        keep.append(WEIGHT_COL)
    keep = list(dict.fromkeys(keep))

    pipeline = build_preprocessing(spec, num_cols, kind=kind, include_text=include_text, text_features=text_features)
    preprocessing = pipeline.fit(train_raw)

    train = preprocessing.transform(train_raw).select(*keep).persist(StorageLevel.MEMORY_AND_DISK)
    test = preprocessing.transform(test_raw).select(*keep).persist(StorageLevel.MEMORY_AND_DISK)

    in_valid = hash_bucket(spec.group_col, "valid") < int(valid_fraction * 1000)
    valid = train.filter(in_valid)
    fit = train.filter(~in_valid)
    if sample_fraction is not None and sample_fraction < 1.0:
        fit = fit.filter(hash_bucket(spec.group_col, "sample") < int(sample_fraction * 1000))

    names = _readable_names(feature_names(train), num_cols, preprocessing, train_raw)
    print(f"[{spec.name}] kind={kind} features={len(names)} "
          f"train={train.count():,} test={test.count():,} fit={fit.count():,} valid={valid.count():,}")
    if weights:
        print(f"[{spec.name}] class weights: {weights}")

    return PreparedData(spec, kind, preprocessing, train, test, fit, valid, names, weights)


def feature_names(df: DataFrame, column: str = FEATURES_COL) -> List[str]:
    """Human readable names of the assembled vector slots (from ML attribute metadata)."""
    meta = df.schema[column].metadata.get("ml_attr", {})
    attrs = meta.get("attrs", {})
    named = sorted((a["idx"], a.get("name", f"f{a['idx']}")) for group in attrs.values() for a in group)
    size = meta.get("num_attrs", len(named))
    names = [f"f{i}" for i in range(size)]
    for idx, name in named:
        names[idx] = name.replace("__imp", "")
    return names


def _readable_names(
    names: List[str], numeric_cols: Sequence[str], preprocessing: PipelineModel, raw: DataFrame
) -> List[str]:
    """Replaces technical slot names with column names / words.

    - the scaled numeric vector has no metadata: slot i -> numeric_cols[i];
    - hashed text slots -> the words that fall into that hash bucket.
    """
    result = []
    for name in names:
        if name.startswith("numeric_scaled_") and name.rsplit("_", 1)[-1].isdigit():
            name = numeric_cols[int(name.rsplit("_", 1)[-1])]
        result.append(name)

    hashers = [s for s in preprocessing.stages if isinstance(s, HashingTF)]
    if hashers:
        transformed = preprocessing.transform(raw)
        for hasher in hashers:
            words_col, prefix = hasher.getInputCol(), hasher.getOutputCol().replace("__tf", "__tfidf")
            vocabulary = [r[0] for r in transformed.select(F.explode(words_col)).distinct().collect()]
            buckets: Dict[int, List[str]] = {}
            for word in vocabulary:
                buckets.setdefault(hasher.indexOf(word), []).append(word)
            for i, name in enumerate(result):
                if name.startswith(prefix + "_") and name.rsplit("_", 1)[-1].isdigit():
                    words = sorted(buckets.get(int(name.rsplit("_", 1)[-1]), []))
                    result[i] = f"word: {'/'.join(words[:3])}" if words else name
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def positive_probability(column: str = "probability") -> Column:
    return vector_to_array(F.col(column)).getItem(1)


def classification_metrics(predictions: DataFrame, threshold: Optional[float] = None) -> Dict[str, float]:
    """Accuracy / Precision / Recall / F1 for the positive class (label 1),
    plus threshold-independent ROC AUC and PR AUC.

    threshold=None uses the model's own `prediction` column.
    """
    if threshold is None:
        predicted = F.col("prediction")
    else:
        predicted = (positive_probability() >= F.lit(threshold)).cast("double")
    label = F.col(LABEL_COL)
    row = predictions.select(
        F.sum(((predicted == 1) & (label == 1)).cast("long")).alias("tp"),
        F.sum(((predicted == 1) & (label == 0)).cast("long")).alias("fp"),
        F.sum(((predicted == 0) & (label == 1)).cast("long")).alias("fn"),
        F.sum(((predicted == 0) & (label == 0)).cast("long")).alias("tn"),
    ).collect()[0]
    tp, fp, fn, tn = (row[k] or 0 for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    score_col = score_column(predictions)
    if score_col:
        for name in ("areaUnderROC", "areaUnderPR"):
            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol=score_col, metricName=name)
            metrics["roc_auc" if name == "areaUnderROC" else "pr_auc"] = evaluator.evaluate(predictions)
    return metrics


def score_column(predictions: DataFrame) -> Optional[str]:
    """Column used to rank examples for ROC/PR AUC.

    `probability` is preferred: for DecisionTreeClassifier `rawPrediction` holds raw
    (weighted) class counts of the leaf, which do not rank examples correctly.
    """
    for column in ("probability", "rawPrediction"):
        if column in predictions.columns:
            return column
    return None


def regression_metrics(predictions: DataFrame) -> Dict[str, float]:
    return {
        name: RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName=name).evaluate(predictions)
        for name in ("rmse", "r2", "mae")
    }


def evaluate(predictions: DataFrame, task: str, threshold: Optional[float] = None) -> Dict[str, float]:
    if task == CLASSIFICATION:
        return classification_metrics(predictions, threshold)
    return regression_metrics(predictions)


def best_threshold(predictions: DataFrame, step: float = 0.01) -> Tuple[float, pd.DataFrame]:
    """Threshold on P(label=1) that maximises F1. Use it on the validation part,
    never on test. Returns (threshold, table with precision/recall/f1 per threshold)."""
    thresholds = [round(step * i, 4) for i in range(1, int(1 / step))]
    p1 = positive_probability()
    label = F.col(LABEL_COL)
    exprs = []
    for i, t in enumerate(thresholds):
        exprs += [
            F.sum(((p1 >= t) & (label == 1)).cast("long")).alias(f"tp{i}"),
            F.sum(((p1 >= t) & (label == 0)).cast("long")).alias(f"fp{i}"),
        ]
    exprs.append(F.sum((label == 1).cast("long")).alias("pos"))
    row = predictions.select(*exprs).collect()[0]
    records = []
    for i, t in enumerate(thresholds):
        tp, fp, pos = row[f"tp{i}"] or 0, row[f"fp{i}"] or 0, row["pos"] or 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / pos if pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        records.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    table = pd.DataFrame(records)
    best = float(table.loc[table["f1"].idxmax(), "threshold"])
    return best, table


def baseline_metrics(train: DataFrame, test: DataFrame, task: str) -> Dict[str, float]:
    """Trivial reference model: majority class for classification, train mean for regression.
    Every real model must beat it."""
    if task == CLASSIFICATION:
        majority = train.groupBy(LABEL_COL).count().orderBy(F.desc("count")).first()[LABEL_COL]
        preds = test.withColumn("prediction", F.lit(float(majority)))
        return classification_metrics(preds)
    mean_value = train.select(F.avg(LABEL_COL)).first()[0]
    preds = test.withColumn("prediction", F.lit(float(mean_value)))
    return regression_metrics(preds)


# ---------------------------------------------------------------------------
# Hyper-parameter search
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    best_params: Dict[Param, object]
    best_model: object
    table: pd.DataFrame   # one row per candidate: params, train/valid metric, fit time
    metric: str


def tune(
    estimator: Estimator,
    param_grid: List[Dict[Param, object]],
    fit_df: DataFrame,
    valid_df: DataFrame,
    task: str,
    metric: Optional[str] = None,
) -> TuningResult:
    """Train-validation search over a ParamGridBuilder grid.

    Works like pyspark.ml.tuning.TrainValidationSplit, but also records the metric
    on the fit part and the fit time of every candidate. Comparing train vs valid
    metric shows under/over-fitting, which is needed for the training analysis.
    """
    if metric is None:
        metric = "areaUnderPR" if task == CLASSIFICATION else "rmse"
    larger_is_better = metric not in ("rmse", "mae", "mse")
    if task == CLASSIFICATION:
        # probability, not rawPrediction: see score_column()
        evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="probability", metricName=metric)
    else:
        evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName=metric)

    rows, best = [], None
    for i, params in enumerate(param_grid, start=1):
        started = time.perf_counter()
        model = estimator.fit(fit_df, params)
        seconds = time.perf_counter() - started
        train_score = evaluator.evaluate(model.transform(fit_df))
        valid_score = evaluator.evaluate(model.transform(valid_df))
        readable = {p.name: _plain(v) for p, v in params.items()}
        rows.append({**readable, f"train_{metric}": train_score, f"valid_{metric}": valid_score, "fit_seconds": seconds})
        print(f"  [{i}/{len(param_grid)}] {readable} train={train_score:.4f} valid={valid_score:.4f} ({seconds:.1f}s)")
        better = best is None or (valid_score > best[0] if larger_is_better else valid_score < best[0])
        if better:
            best = (valid_score, params, model)

    table = pd.DataFrame(rows)
    print(f"  best: {{{', '.join(f'{p.name}: {_plain(v)}' for p, v in best[1].items())}}} valid {metric}={best[0]:.4f}")
    return TuningResult(best_params=best[1], best_model=best[2], table=table, metric=metric)


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def params_to_dict(params: Dict[Param, object]) -> Dict[str, object]:
    return {p.name: _plain(v) for p, v in params.items()}


def timed_fit(estimator: Estimator, df: DataFrame, params: Optional[Dict[Param, object]] = None):
    """Fits a model and returns (model, seconds)."""
    started = time.perf_counter()
    model = estimator.fit(df, params or {})
    return model, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Writing results (output/ml/<member>/...)
# ---------------------------------------------------------------------------

def output_dir(member: str, *sub: str) -> str:
    path = os.path.join(OUTPUT_DIR, member, *sub)
    os.makedirs(path, exist_ok=True)
    return path


def save_table(table: pd.DataFrame, member: str, name: str) -> str:
    path = os.path.join(output_dir(member), f"{name}.csv")
    table.to_csv(path, index=False)
    return path


def save_plot(fig, member: str, name: str) -> str:
    path = os.path.join(output_dir(member, "plots"), f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_metrics(
    member: str,
    task: str,
    model_name: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, object]] = None,
    fit_seconds: Optional[float] = None,
    threshold: Optional[float] = None,
    num_features: Optional[int] = None,
) -> str:
    """One row per model; compare.py collects these files from all members."""
    row = {
        "member": member,
        "task": task,
        "model": model_name,
        **{k: v for k, v in metrics.items()},
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "num_features": num_features,
        "params": json.dumps(params or {}, default=str),
    }
    return save_table(pd.DataFrame([row]), member, f"metrics__{task}__{model_name}")


def plot_tuning(table: pd.DataFrame, x: str, metric: str, hue: Optional[str] = None, title: str = "", log_x: bool = False):
    """Train vs validation metric against one hyper-parameter (optionally one line per `hue` value)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = table.groupby(hue) if hue else [(None, table)]
    for key, part in groups:
        part = part.sort_values(x)
        suffix = f" ({hue}={key})" if hue else ""
        ax.plot(part[x], part[f"valid_{metric}"], marker="o", label=f"valid{suffix}")
        ax.plot(part[x], part[f"train_{metric}"], marker="x", linestyle="--", alpha=0.6, label=f"train{suffix}")
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


def plot_top_weights(names: Sequence[str], values: Sequence[float], title: str, top: int = 20):
    """Horizontal bar chart of the largest |coefficients| or feature importances."""
    pairs = sorted(zip(names, values), key=lambda p: abs(p[1]), reverse=True)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(pairs) + 1.5))
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=["#2a9d8f" if p[1] >= 0 else "#e76f51" for p in pairs])
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    return fig


def weights_table(names: Sequence[str], values: Sequence[float], column: str = "weight") -> pd.DataFrame:
    table = pd.DataFrame({"feature": list(names), column: list(values)})
    table["abs"] = table[column].abs()
    return table.sort_values("abs", ascending=False).drop(columns="abs")
