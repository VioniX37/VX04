import time

from fastapi.testclient import TestClient

from automl_agent.main import create_app


def test_health_upload_and_run(sample_csvs):
    with TestClient(create_app()) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["llm_provider"] == "fake"

        with sample_csvs["churn"].open("rb") as f:
            resp = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")})
        assert resp.status_code == 201, resp.text
        dataset = resp.json()
        assert dataset["profile"]["guessed_target"] == "churn"

        bad = client.post("/api/datasets", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        assert bad.status_code == 400

        run = client.post("/api/runs", json={"dataset_id": dataset["id"], "prompt": "Predict churn"}).json()
        deadline = time.time() + 120
        while time.time() < deadline:
            status = client.get(f"/api/runs/{run['id']}").json()
            if status["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        assert status["status"] == "succeeded", status.get("error")
        assert status["metrics"]["score"] is not None
        assert "metrics.json" in status["code"]

        events = client.get(f"/api/runs/{run['id']}/events/history").json()
        assert events[-1]["stage"] == "done"

        with client.stream("GET", f"/api/runs/{run['id']}/events") as stream:
            body = "".join(stream.iter_text())
        assert "agent_event" in body and "event: end" in body


def test_register_by_path_and_errors(sample_csvs):
    with TestClient(create_app()) as client:
        ok = client.post("/api/datasets/register", json={"path": str(sample_csvs["houses"])})
        assert ok.status_code == 201, ok.text
        body = ok.json()
        assert body["source"] == "path" and body["profile"]["n_rows"] == 400
        assert body["profile"]["scale_tier"] == "small"

        missing = client.post("/api/datasets/register", json={"path": "does/not/exist.csv"})
        assert missing.status_code == 400
        both = client.post("/api/datasets/register", json={"path": "a.csv", "url": "http://x/a.csv"})
        assert both.status_code == 422


def test_observations_endpoint(sample_csvs):
    with TestClient(create_app()) as client:
        with sample_csvs["reviews"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("r.csv", f, "text/csv")}).json()
        run = client.post(
            "/api/runs", json={"dataset_id": dataset["id"], "prompt": "Classify sentiment"}
        ).json()
        deadline = time.time() + 120
        while time.time() < deadline and client.get(f"/api/runs/{run['id']}").json()["status"] not in (
            "succeeded",
            "failed",
        ):
            time.sleep(0.5)
        rows = client.get(f"/api/runs/{run['id']}/observations").json()
        assert any(r["final"] for r in rows) and any(not r["final"] for r in rows)
        assert client.get(f"/api/runs/{run['id']}").json()["config"]["verification_mode"] == "grounded"


def test_uploads_buffer_on_workspace_drive():
    import tempfile

    from automl_agent.config import get_settings

    with TestClient(create_app()):
        assert tempfile.gettempdir() == str(get_settings().tmp_dir)


def test_disk_full_during_upload_gives_clear_error():
    from fastapi import HTTPException

    app = create_app()

    @app.get("/boom")
    def boom():
        raise HTTPException(400, "There was an error parsing the body") from OSError(
            28, "No space left on device"
        )

    with TestClient(app) as client:
        resp = client.get("/boom")
    assert resp.status_code == 507
    assert "register the file by path" in resp.json()["detail"]


def _wait_for_job(client, job):
    deadline = time.time() + 60
    while job["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        job = client.get(f"/api/datasets/jobs/{job['id']}").json()
    return job


def test_background_registration_reports_progress(sample_csvs, tmp_path):
    with TestClient(create_app()) as client:
        started = client.post("/api/datasets/jobs/register", json={"path": str(sample_csvs["houses"])})
        assert started.status_code == 202, started.text
        assert started.json()["steps"] == ["convert", "profile"]
        job = _wait_for_job(client, started.json())
        assert job["status"] == "succeeded", job["error"]
        assert job["phase"] == "ready" and job["fraction"] == 1.0
        assert job["dataset"]["profile"]["n_rows"] == 400

        with sample_csvs["churn"].open("rb") as f:
            started = client.post("/api/datasets/jobs", files={"file": ("churn.csv", f, "text/csv")})
        assert started.status_code == 202 and started.json()["steps"] == ["upload", "convert", "profile"]
        assert _wait_for_job(client, started.json())["dataset"]["profile"]["guessed_target"] == "churn"

        # Bad input fails at once; unreadable content fails inside the job with a message.
        assert client.post("/api/datasets/jobs/register", json={"path": "nope.csv"}).status_code == 400
        empty = tmp_path / "empty.csv"
        empty.write_text("a,b\n")
        started = client.post("/api/datasets/jobs/register", json={"path": str(empty)})
        job = _wait_for_job(client, started.json())
        assert job["status"] == "failed" and "no rows" in job["error"]
        assert client.get("/api/datasets/jobs/missing").status_code == 404
