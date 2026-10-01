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
    assert result.peak_memory_mb is not None


def test_run_script_failure_captures_stderr(tmp_path):
    code = "raise RuntimeError('boom')  # metrics.json\n"
    result = asyncio.run(run_script(code, tmp_path))
    assert not result.ok and result.returncode != 0 and "boom" in result.stderr


def test_run_script_timeout(tmp_path):
    code = "import time\ntime.sleep(10)  # metrics.json\n"
    result = asyncio.run(run_script(code, tmp_path, timeout_s=1))
    assert result.timed_out and not result.ok and result.duration_s < 8


def test_run_script_memory_limit(tmp_path):
    code = "import time\nblob = bytearray(400 * 1024 * 1024)\ntime.sleep(5)  # metrics.json\n"
    result = asyncio.run(run_script(code, tmp_path, max_mem_mb=150))
    assert result.memory_exceeded and not result.ok
    assert "memory limit" in result.stderr


def test_progress_lines_are_streamed(tmp_path):
    code = (
        "import json, time\n"
        "for i in range(3):\n"
        "    print('PROGRESS ' + json.dumps({'stage': 'step', 'i': i}), flush=True)\n"
        "    time.sleep(0.3)\n"
        "print('regular output')\n"
        "json.dump({'score': 1}, open('metrics.json', 'w'))\n"
    )
    seen = []

    async def on_progress(item):
        if not item.get("telemetry"):
            seen.append(item["i"])

    result = asyncio.run(run_script(code, tmp_path, on_progress=on_progress))
    assert result.ok and seen == [0, 1, 2]
    assert "regular output" in result.stdout and "PROGRESS" not in result.stdout


def test_resource_telemetry_is_streamed(tmp_path):
    code = (
        "import json, time\n"
        "blob = bytearray(80 * 1024 * 1024)\n"
        "end = time.time() + 5\n"
        "while time.time() < end:\n"
        "    sum(range(10000))\n"
        "json.dump({'score': 1}, open('metrics.json', 'w'))\n"
    )
    samples = []

    async def on_progress(item):
        if item.get("telemetry"):
            samples.append(item)

    result = asyncio.run(run_script(code, tmp_path, on_progress=on_progress))
    assert result.ok and len(samples) >= 2
    assert all(s["rss_mb"] > 50 for s in samples)
    assert max(s["cores"] for s in samples) > 0.3  # the busy loop shows up as CPU use
