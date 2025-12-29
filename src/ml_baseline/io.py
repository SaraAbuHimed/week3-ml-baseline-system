from pathlib import Path
import pandas as pd


def write_tabular(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False
