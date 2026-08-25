from __future__ import annotations

from pathlib import Path
import subprocess

from caso_berka_model.config import PROJ_ROOT


class DataLineage:
    """Extrae hashes de Git y estado DVC para tags de trazabilidad en MLflow."""

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root) if project_root else PROJ_ROOT

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def get_git_commit(self) -> str:
        try:
            result = self._run(["git", "rev-parse", "--short", "HEAD"])
            commit = (result.stdout or "").strip()
            if result.returncode == 0 and commit:
                return commit
            return "standalone_execution"
        except OSError:
            return "git_unavailable"

    def get_dvc_status(self) -> str:
        try:
            result = self._run(["dvc", "status"])
            if result.returncode != 0:
                return "dvc_unavailable"
            status = (result.stdout or "").strip()
            if not status or "Data and pipelines are up to date" in status:
                return "synced"
            return "uncommitted_changes"
        except OSError:
            return "dvc_unavailable"

    def get_lineage_metadata(self) -> dict[str, str]:
        return {
            "git_commit": self.get_git_commit(),
            "dvc_status": self.get_dvc_status(),
        }
