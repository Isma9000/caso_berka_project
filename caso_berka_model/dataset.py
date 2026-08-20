from pathlib import Path

import pandas as pd
import yaml

from caso_berka_model.config import (
    PARAMS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TABLA_MINABLE,
)
from caso_berka_model.features import FeatureEngineer


class DataProcessor:
    """Ingesta y preprocesamiento de datos Berka."""

    RAW_TABLES = ("client", "disp", "card", "account", "loan", "trans")

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        params_path: str | Path | None = None,
    ):
        self.raw_dir = Path(input_path) if input_path else RAW_DATA_DIR
        self.output_path = Path(output_path) if output_path else TABLA_MINABLE
        self.params_path = Path(params_path) if params_path else PARAMS_PATH
        self.engineer = FeatureEngineer()

    def _load_params(self) -> dict:
        if not self.params_path.exists():
            return {"prepare": {"null_threshold": 0.60}}
        with open(self.params_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _resolve_raw_file(self, filename: str) -> Path:
        candidates = [
            self.raw_dir / filename,
            self.raw_dir / "data" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        matches = list(self.raw_dir.rglob(filename))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"No se encontró {filename} bajo {self.raw_dir}"
        )

    def load_data(self) -> dict[str, pd.DataFrame]:
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"El directorio {self.raw_dir} no existe.")

        tables = {}
        for name in self.RAW_TABLES:
            path = self._resolve_raw_file(f"{name}.asc")
            tables[name] = pd.read_csv(path, sep=";", low_memory=False)
        return tables

    def clean_data(self, datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
        params = self._load_params()
        null_threshold = params.get("prepare", {}).get("null_threshold", 0.60)
        return self.engineer.clean_data(datos, null_threshold=null_threshold)

    def save_data(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"[DataProcessor] Datos guardados en: {self.output_path}")

    def run(self) -> None:
        datos = self.load_data()
        tabla = self.clean_data(datos)
        self.save_data(tabla)


def main():
    DataProcessor().run()


if __name__ == "__main__":
    main()
