"""Documentation stays in sync with the code."""

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_configuration_reference_is_up_to_date():
    cmd = [sys.executable, "scripts/gen_config_reference.py", "--check"]
    result = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_setting_is_documented():
    from automl_agent.config import Settings

    text = (BACKEND.parent / "docs" / "reference" / "configuration.md").read_text(encoding="utf-8")
    missing = [name for name in Settings.model_fields if f"`{name.upper()}" not in text]
    assert not missing, f"undocumented settings: {missing}"
