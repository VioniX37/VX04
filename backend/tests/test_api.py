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
