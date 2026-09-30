from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["ExperimentConfig", "load_config"]


@dataclass
class ExperimentConfig:
    seed: int = 20240930
    valid_size: float = 0.4
    n_folds: int = 10
    n_trials: int = 50
    missing_threshold: float = 0.20
    contamination: float = 0.05
    select_features: bool = True
    min_features: int = 5
    rfe_step: float = 0.1
    output_dir: str | None = "results"

    def __post_init__(self) -> None:
        if not 0.0 < self.valid_size < 1.0:
            raise ValueError("valid_size must lie in (0, 1)")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0.0 < self.missing_threshold <= 1.0:
            raise ValueError("missing_threshold must lie in (0, 1]")
        if not 0.0 <= self.contamination < 0.5:
            raise ValueError("contamination must lie in [0, 0.5)")

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text)

    known = {f for f in ExperimentConfig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return ExperimentConfig(**data)
