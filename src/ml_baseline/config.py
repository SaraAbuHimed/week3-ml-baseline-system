from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_cache_dir: Path
    data_external_dir: Path
    reports_dir: Path

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
        )


