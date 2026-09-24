from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.tools import guess_task_spec, profile_file
from automl_agent.verification import verify_request


def test_taskspec_normalizes_invalid_metric():
    spec = TaskSpec(task_type=TaskType.tabular_regression, target_column="price", metric="accuracy")
    assert spec.metric == "rmse"
    assert not spec.higher_is_better
    assert spec.meets_target(1.0)


def test_meets_target_direction():
    clf = TaskSpec(
        task_type=TaskType.tabular_classification, target_column="y", metric="f1_macro", metric_target=0.8
    )
    assert clf.meets_target(0.85) and not clf.meets_target(0.7)
    reg = TaskSpec(task_type=TaskType.tabular_regression, target_column="y", metric="mae", metric_target=10)
    assert reg.meets_target(8) and not reg.meets_target(12)


def test_profiler_detects_kinds(sample_csvs):
    churn = profile_file(sample_csvs["churn"])
    assert churn.guessed_target == "churn"
    assert churn.column("customer_id").kind == "identifier"
    assert churn.column("contract").kind == "categorical"
    assert churn.column("monthly_charges").n_missing > 0

    reviews = profile_file(sample_csvs["reviews"])
    assert reviews.text_columns == ["review_text"]


def test_heuristic_parser(sample_csvs):
    houses = profile_file(sample_csvs["houses"])
    spec = guess_task_spec("Predict the price of a house, optimise MAE", houses)
    assert spec.task_type == TaskType.tabular_regression
    assert spec.target_column == "price"
    assert spec.metric == "mae"

    reviews = profile_file(sample_csvs["reviews"])
    spec = guess_task_spec("Classify review sentiment with at least 80% accuracy", reviews)
    assert spec.task_type == TaskType.text_classification
    assert spec.text_column == "review_text"
    assert spec.target_column == "sentiment"
    assert spec.metric == "accuracy" and spec.metric_target == 0.8


def test_request_verification_rejects_bad_columns(sample_csvs):
    churn = profile_file(sample_csvs["churn"])
    bad = TaskSpec(task_type=TaskType.tabular_classification, target_column="nope", metric="accuracy")
    result = verify_request(bad, churn)
    assert not result.ok and "nope" in result.feedback
