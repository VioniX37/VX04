"""Tests for the inference module and the REST inference endpoints.

All tests run offline with the fake LLM backend, no API key required.
The test_pipeline_e2e.py suite verifies that model.joblib exists after a
successful run; these tests pick up from there and verify the inference layer.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from automl_agent.execution.inference import (
    InferenceError,
    build_schema,
    load_bundle,
    predict_dataframe,
    score_file,
    validate_input,
)
from automl_agent.main import create_app


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _make_tabular_bundle(tmp_path: Path, is_clf: bool = True) -> tuple[Path, dict]:
    """Create a minimal model.joblib bundle for unit tests without running a full pipeline."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame({"age": [30.0, 45.0, 25.0], "income": [50000.0, 80000.0, 30000.0]})
    if is_clf:
        y = np.array([0, 1, 0])
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ])
        model.fit(X, y)
        classes = np.array(["no", "yes"], dtype=object)
        task_type = "tabular_classification"
    else:
        y = np.array([100.0, 200.0, 80.0])
        model = Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge()),
        ])
        model.fit(X, y)
        classes = None
        task_type = "tabular_regression"

    bundle = {
        "model": model,
        "classes": classes,
        "features": ["age", "income"],
        "task_type": task_type,
        "target": "churn",
        "drop_columns": [],
        "config": {"task_type": task_type, "model_family": "logistic_regression"},
    }
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(bundle, bundle_path)
    return bundle_path, bundle


# --------------------------------------------------------------------------- #
# Unit tests – load_bundle                                                     #
# --------------------------------------------------------------------------- #


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(InferenceError, match="No trained model"):
        load_bundle(tmp_path / "does_not_exist.joblib")


def test_load_bundle_wrong_format(tmp_path):
    p = tmp_path / "bad.joblib"
    joblib.dump([1, 2, 3], p)  # a list, not a dict
    with pytest.raises(InferenceError, match="unexpected format"):
        load_bundle(p)


def test_load_bundle_missing_keys(tmp_path):
    p = tmp_path / "bad.joblib"
    joblib.dump({"model": None}, p)
    with pytest.raises(InferenceError, match="missing keys"):
        load_bundle(p)


def test_load_bundle_ok(tmp_path):
    bundle_path, bundle = _make_tabular_bundle(tmp_path)
    loaded = load_bundle(bundle_path)
    assert set(loaded.keys()) >= {"model", "features", "task_type"}


# --------------------------------------------------------------------------- #
# Unit tests – validate_input                                                  #
# --------------------------------------------------------------------------- #


def test_validate_input_missing_column(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pd.DataFrame({"age": [30.0]})  # income missing
    with pytest.raises(InferenceError) as exc_info:
        validate_input(df, bundle)
    assert exc_info.value.column == "income"
    assert "income" in str(exc_info.value)


def test_validate_input_extra_column(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pd.DataFrame({"age": [30.0], "income": [50000.0], "extra": ["x"]})
    with pytest.raises(InferenceError) as exc_info:
        validate_input(df, bundle)
    assert exc_info.value.column == "extra"


def test_validate_input_drops_target(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pd.DataFrame({"age": [30.0], "income": [50000.0], "churn": ["yes"]})
    validated = validate_input(df, bundle)
    assert "churn" not in validated.columns
    assert list(validated.columns) == ["age", "income"]


def test_validate_input_reorders_columns(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pd.DataFrame({"income": [50000.0], "age": [30.0]})  # wrong order
    validated = validate_input(df, bundle)
    assert list(validated.columns) == ["age", "income"]  # training order


# --------------------------------------------------------------------------- #
# Unit tests – predict_dataframe                                               #
# --------------------------------------------------------------------------- #


def test_predict_classification_returns_labels_and_probas(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path, is_clf=True)
    df = pd.DataFrame({"age": [30.0], "income": [50000.0]})
    validated = validate_input(df, bundle)
    result = predict_dataframe(validated, bundle)
    assert "predictions" in result
    assert result["predictions"][0] in ["no", "yes"]
    assert "probabilities" in result
    assert len(result["probabilities"][0]) == 2  # binary
    assert "classes" in result


def test_predict_regression_returns_floats(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path, is_clf=False)
    df = pd.DataFrame({"age": [30.0], "income": [50000.0]})
    validated = validate_input(df, bundle)
    result = predict_dataframe(validated, bundle)
    assert "predictions" in result
    assert isinstance(result["predictions"][0], float)
    assert "probabilities" not in result


# --------------------------------------------------------------------------- #
# Unit tests – score_file (batch)                                              #
# --------------------------------------------------------------------------- #


def test_score_file_parquet_output(tmp_path):
    bundle_path, bundle = _make_tabular_bundle(tmp_path)
    # Create a small Parquet file to score.
    df = pl.DataFrame({"age": [30.0, 45.0], "income": [50000.0, 80000.0]})
    src = tmp_path / "input.parquet"
    df.write_parquet(src)
    out_bytes = score_file(src, bundle, output_format="parquet")
    result = pl.read_parquet(io.BytesIO(out_bytes))
    assert "prediction" in result.columns
    assert len(result) == 2


def test_score_file_csv_output(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pl.DataFrame({"age": [30.0], "income": [50000.0]})
    src = tmp_path / "input.parquet"
    df.write_parquet(src)
    out_bytes = score_file(src, bundle, output_format="csv")
    text = out_bytes.decode()
    assert "prediction" in text


def test_score_file_missing_column_raises(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    df = pl.DataFrame({"age": [30.0]})  # income missing
    src = tmp_path / "bad.parquet"
    df.write_parquet(src)
    with pytest.raises(InferenceError) as exc_info:
        score_file(src, bundle)
    assert exc_info.value.column == "income"


# --------------------------------------------------------------------------- #
# Unit tests – build_schema                                                    #
# --------------------------------------------------------------------------- #


def test_build_schema_returns_all_features(tmp_path):
    _, bundle = _make_tabular_bundle(tmp_path)
    schema = build_schema(bundle)
    assert set(schema.keys()) == {"age", "income"}


# --------------------------------------------------------------------------- #
# API integration tests (offline fake LLM)                                    #
# --------------------------------------------------------------------------- #


def _run_until_done(client, dataset_id: str, prompt: str) -> dict:
    """Start a run and poll until it finishes; return the run dict."""
    import time

    run = client.post("/api/runs", json={"dataset_id": dataset_id, "prompt": prompt}).json()
    deadline = time.time() + 120
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run['id']}").json()
        if status["status"] in ("succeeded", "failed"):
            return status
        time.sleep(0.5)
    raise TimeoutError("Run did not finish in time")


def test_predict_json_on_succeeded_run(sample_csvs):
    """POST /runs/{id}/predict returns predictions for valid JSON records."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()
        run = _run_until_done(client, dataset["id"], "Predict churn")
        if run["status"] != "succeeded":
            pytest.skip("Run failed in fake LLM; skipping inference assertions")

        # Read a few rows from the CSV for the request body.
        sample = pd.read_csv(sample_csvs["churn"]).head(3)
        task_spec = run.get("task_spec") or {}
        target = task_spec.get("target_column", "churn")
        drop = task_spec.get("drop_columns", [])
        feature_cols = [c for c in sample.columns if c != target and c not in drop]
        records = sample[feature_cols].to_dict(orient="records")

        resp = client.post(f"/api/runs/{run['id']}/predict", json=records)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 3


def test_predict_json_schema_mismatch_returns_422(sample_csvs):
    """POST /runs/{id}/predict with a missing column returns 422 with the column name."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()
        run = _run_until_done(client, dataset["id"], "Predict churn")
        if run["status"] != "succeeded":
            pytest.skip("Run failed; skipping schema mismatch test")

        resp = client.post(f"/api/runs/{run['id']}/predict", json=[{"nonsense_col": 42}])
        assert resp.status_code == 422
        body = resp.json()
        # The detail should be our InferenceError dict, not FastAPI's validation error.
        detail = body.get("detail", {})
        assert "error" in detail or "column" in detail


def test_predict_json_on_failed_run(sample_csvs):
    """POST /runs/{id}/predict on a failed/missing run returns 422, not 500."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()
        run = client.post("/api/runs", json={"dataset_id": dataset["id"], "prompt": "Predict churn"}).json()
        # Don't wait for it: try to predict immediately; run is "pending" or "running".
        resp = client.post(f"/api/runs/{run['id']}/predict", json=[{"x": 1}])
        # Must be 422 (not succeeded) or 404 (not found) — never 500.
        assert resp.status_code in (422, 404)


def test_predict_nonexistent_run():
    """GET/POST on a run that doesn't exist returns 404."""
    with TestClient(create_app()) as client:
        resp = client.post("/api/runs/doesnotexist99/predict", json=[{"x": 1}])
        assert resp.status_code == 404
        resp2 = client.get("/api/runs/doesnotexist99/artifacts/bundle")
        assert resp2.status_code == 404


def test_predict_empty_body(sample_csvs):
    """POST /runs/{id}/predict with an empty list returns 422."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()
        run = _run_until_done(client, dataset["id"], "Predict churn")
        if run["status"] != "succeeded":
            pytest.skip("Run failed; skipping empty body test")
        resp = client.post(f"/api/runs/{run['id']}/predict", json=[])
        assert resp.status_code == 422


def test_bundle_download_on_succeeded_run(sample_csvs):
    """GET /runs/{id}/artifacts/bundle returns a valid zip with required files."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()
        run = _run_until_done(client, dataset["id"], "Predict churn")
        if run["status"] != "succeeded":
            pytest.skip("Run failed; skipping bundle download test")

        resp = client.get(f"/api/runs/{run['id']}/artifacts/bundle")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        for required in ("model.joblib", "predict.py", "requirements.txt", "schema.json"):
            assert required in names, f"{required} missing from bundle zip"
        schema = json.loads(zf.read("schema.json"))
        assert "features" in schema and "task_type" in schema and "dtypes" in schema
