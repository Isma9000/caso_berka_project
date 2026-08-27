import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def test_dvc_lock_has_no_merge_markers():
    lock_text = (PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8")
    for marker in ("<<<<<<<", "=======", ">>>>>>>"):
        assert marker not in lock_text


def test_dvc_yaml_and_params_are_valid():
    dvc_yaml = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))

    assert "stages" in dvc_yaml
    assert {"preprocess", "train"}.issubset(dvc_yaml["stages"])
    assert "prepare" in params
    assert "train" in params
    assert "stages" in lock


@pytest.mark.slow
def test_dvc_status_succeeds():
    if shutil.which("dvc") is None:
        pytest.skip("dvc no está instalado en PATH")

    result = subprocess.run(
        ["dvc", "status"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
