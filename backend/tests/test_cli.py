import json

from automl_agent.cli import main


def test_cli_ingest_and_run(sample_csvs, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    from automl_agent.config import get_settings

    get_settings.cache_clear()
    try:
        assert main(["ingest", "--data", str(sample_csvs["churn"])]) == 0
        dataset_id = json.loads(capsys.readouterr().out)["dataset_id"]

        code = main(["--set", "N_PLANS=2", "run", "--dataset-id", dataset_id, "--prompt", "Predict churn"])
        out = capsys.readouterr().out
        assert code == 0, out
        summary = json.loads(out[out.index("{\n") :])
        assert summary["success"] and summary["score"] is not None
        assert "[       prepare]" in out

        assert main(["datasets"]) == 0
        assert dataset_id in capsys.readouterr().out

        scored = tmp_path / "scored.csv"
        assert (
            main(
                [
                    "predict",
                    "--run",
                    summary["run_id"],
                    "--data",
                    str(sample_csvs["churn"]),
                    "--out",
                    str(scored),
                ]
            )
            == 0
        )
        assert scored.read_text().splitlines()[0].startswith("prediction")
        assert len(scored.read_text().splitlines()) == 401
        assert (
            main(
                ["predict", "--run", "no-such-run", "--data", str(sample_csvs["churn"]), "--out", str(scored)]
            )
            == 2
        )
    finally:
        get_settings.cache_clear()
