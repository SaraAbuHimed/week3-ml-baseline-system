from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

'''
@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_cache_dir: Path
    data_external_dir: Path
    reports_dir: Path
    models_runs_dir: Path
    models_registry_dir: Path

    @staticmethod
    def from_repo_root() -> "Paths":
        root = Path.cwd()
        data = root / "data"
        return Paths(
            root=root,
            data_dir=data,
            data_raw_dir=data / "raw",
            data_processed_dir=data / "processed",
            data_cache_dir=data / "cache",
            data_external_dir=data / "external",
            reports_dir=root / "reports",
            models_runs_dir=root / "models" / "runs",
            models_registry_dir=root / "models" / "registry",
        )
'''




@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_cache_dir: Path
    data_external_dir: Path
    reports_dir: Path
    models_runs_dir: Path
    models_registry_dir: Path

    @staticmethod
    def from_repo_root() -> "Paths":
        root = Path.cwd()
        data = root / "data"
        models = root / "models"

        return Paths(
            root=root,
            data_dir=data,
            data_raw_dir=data / "raw",
            data_processed_dir=data / "processed",
            data_cache_dir=data / "cache",
            data_external_dir=data / "external",
            reports_dir=root / "reports",
            models_runs_dir=models / "runs",
            models_registry_dir=models / "registry",
        )


'''
@dataclass(frozen=True)
class TrainConfig:
    target: str
    seed: int
    test_size: float = 0.2
'''





@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str

    # task type: "classification" | "regression" (keep simple for now)
    task: str = "classification"

    # columns
    id_cols: tuple[str, ...] = ("id",)          # optional passthrough identifiers
    time_col: str | None = None                # if set, sort by time before splitting
    group_col: str | None = None               # (not implemented yet)

    # splitting / reproducibility
    session_id: int = 42
    train_size: float = 0.8
    fold: int = 5

