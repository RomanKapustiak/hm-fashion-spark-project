import pytest
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, StructField, StructType

from src.analysis.common import (
    CLASSIFICATION,
    LABEL_COL,
    SPLIT_COL,
    DatasetSpec,
    assign_split,
    baseline_metrics,
    best_threshold,
    build_preprocessing,
    classification_metrics,
    feature_names,
    numeric_columns,
    regression_metrics,
    threshold_table,
)

SPEC = DatasetSpec(
    name="tiny", task=CLASSIFICATION, id_col="id", group_col="id",
    categorical=("colour",), text=("text",),
)


@pytest.fixture
def tiny(spark):
    rows = [
        (str(i), float(i % 7), None if i % 5 == 0 else float(i), ["red", "blue", "green"][i % 3],
         "soft cotton jersey top" if i % 2 else "denim jeans", float(i % 2))
        for i in range(60)
    ]
    return spark.createDataFrame(rows, "id string, a double, b double, colour string, text string, label double")


def test_assign_split_is_deterministic(tiny):
    first = {r["id"]: r[SPLIT_COL] for r in assign_split(tiny, "id").collect()}
    second = {r["id"]: r[SPLIT_COL] for r in assign_split(tiny.repartition(7), "id").collect()}
    assert first == second
    assert set(first.values()) == {"train", "test"}


def test_numeric_columns_skip_service_columns(tiny):
    assert numeric_columns(assign_split(tiny, "id"), SPEC) == ["a", "b"]


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessing_builds_feature_vector(tiny, kind):
    model = build_preprocessing(SPEC, ["a", "b"], kind=kind, text_features=16).fit(tiny)
    out = model.transform(tiny)
    size = out.first()["features"].size
    expected = 2 + 4 + 16 if kind == "linear" else 2 + 1  # numeric + one-hot(3 colours + "unseen" slot) + text
    assert size == expected
    assert len(feature_names(out)) == size


def test_classification_metrics_from_confusion_matrix(spark):
    df = spark.createDataFrame(
        [(1.0, 1.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "label double, prediction double",
    )
    m = classification_metrics(df)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_best_threshold_separable(spark):
    df = spark.createDataFrame(
        [(1.0, Vectors.dense([0.2, 0.8])), (1.0, Vectors.dense([0.3, 0.7])),
         (0.0, Vectors.dense([0.6, 0.4])), (0.0, Vectors.dense([0.9, 0.1]))],
        StructType([StructField("label", DoubleType()), StructField("probability", VectorUDT())]),
    )
    threshold, table = best_threshold(df, step=0.1)
    assert 0.4 < threshold <= 0.7
    assert table["f1"].max() == pytest.approx(1.0)


def test_baselines(spark):
    train = spark.createDataFrame([(0.0,), (0.0,), (1.0,)], "label double")
    test = spark.createDataFrame([(0.0,), (1.0,)], "label double")
    m = baseline_metrics(train, test, CLASSIFICATION)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["f1"] == 0.0
    r = regression_metrics(spark.createDataFrame([(1.0, 1.0), (3.0, 3.0)], "label double, prediction double"))
    assert r["rmse"] == pytest.approx(0.0)


def test_logistic_regression_runs_on_prepared_features(tiny):
    prepared = build_preprocessing(SPEC, ["a", "b"], kind="linear", text_features=16).fit(tiny).transform(tiny)
    model = LogisticRegression(labelCol=LABEL_COL, maxIter=5).fit(prepared)
    assert classification_metrics(model.transform(prepared))["accuracy"] > 0.5


def test_threshold_table_roc_endpoints(spark):
    df = spark.createDataFrame(
        [(1.0, Vectors.dense([0.2, 0.8])), (0.0, Vectors.dense([0.6, 0.4])), (1.0, Vectors.dense([0.7, 0.3]))],
        StructType([StructField("label", DoubleType()), StructField("probability", VectorUDT())]),
    )
    table = threshold_table(df, step=0.1)
    first, last = table.iloc[0], table.iloc[-1]
    assert (first["recall"], first["fpr"]) == (1.0, 1.0)  # threshold 0: everything is positive
    assert (last["recall"], last["fpr"]) == (0.0, 0.0)    # threshold > 1: nothing is positive
    assert (table["tp"] + table["fn"] == 2).all() and (table["fp"] + table["tn"] == 1).all()
