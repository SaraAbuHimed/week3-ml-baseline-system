from pathlib import Path
import pandas as pd


def write_tabular(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path

    raise ValueError(
        f"Unsupported file type: {path.suffix} (expected .csv or .parquet)"
    )



def read_tabular(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path.suffix} (expected .csv or .parquet)")


def parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False
