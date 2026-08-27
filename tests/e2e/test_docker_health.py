import shutil
import subprocess

import pytest

pytestmark = pytest.mark.slow


def test_docker_health_if_available():
    if shutil.which("docker") is None:
        pytest.skip("docker no está instalado")

    result = subprocess.run(
        ["docker", "image", "inspect", "caso-berka-api"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("imagen caso-berka-api no construida; ejecuta make docker-build")

    up = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "18000:8000",
            "--env-file",
            "models/docker_meta.env",
            "-e",
            "ENVIRONMENT=docker",
            "caso-berka-api",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if up.returncode != 0:
        pytest.skip(f"no se pudo levantar contenedor: {up.stderr}")

    container_id = up.stdout.strip()
    try:
        health = subprocess.run(
            ["curl", "-sf", "http://127.0.0.1:18000/health"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert health.returncode == 0, health.stderr or health.stdout
    finally:
        subprocess.run(["docker", "stop", container_id], check=False)
