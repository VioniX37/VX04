import asyncio

from automl_agent.execution import run_script, static_check


def test_static_check():
    assert static_check("def (") is not None
    assert "metrics.json" in static_check("print(1)")
    assert static_check("open('metrics.json','w')") is None


def test_run_script_success(tmp_path):
    code = "import json\njson.dump({'score': 0.9, 'metric': 'accuracy'}, open('metrics.json', 'w'))\n"
    result = asyncio.run(run_script(code, tmp_path))
    assert result.ok and result.metrics["score"] == 0.9


def test_run_script_failure_captures_stderr(tmp_path):
    code = "raise RuntimeError('boom')  # metrics.json\n"
    result = asyncio.run(run_script(code, tmp_path))
    assert not result.ok and result.returncode != 0 and "boom" in result.stderr


def test_run_script_timeout(tmp_path):
    code = "import time\ntime.sleep(5)  # metrics.json\n"
    result = asyncio.run(run_script(code, tmp_path, timeout_s=1))
    assert result.timed_out and not result.ok
