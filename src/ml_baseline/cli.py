from __future__ import annotations

from pathlib import Path
import typer

from .sample_data import make_sample_feature_table
#from .train import train_once run_train
from .train import run_train
from .config import TrainCfg

app = typer.Typer(help="ML Baseline System CLI")

@app.command("hello")
def hello() -> None:
    """Sanity command."""
    typer.echo("Hello ✅")

@app.command("make_sample_data")
def make_sample_data(
    n_users: int = typer.Option(50, help="Number of users"),
    seed: int = typer.Option(42, help="Random seed"),
    root: Path | None = typer.Option(None, help="Repo root"),
) -> None:
    """Generate sample data and write features.csv."""
    out_path = make_sample_feature_table(
        root=root,
        n_users=n_users,
        seed=seed,
    )
    typer.echo(f"Wrote: {out_path}")


@app.command("train")
def train(
    target: str = typer.Option(..., "--target", help="Target column name"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """
    Train once and save a run folder under models/runs/
    """

    cfg = TrainCfg(
        features_path=Path("data/processed/features.parquet"),
        target=target,
        session_id=seed,
    )

    run_dir = run_train(cfg, root=Path.cwd(), run_tag="clf")
    typer.echo(f"Saved run: {run_dir}")

'''
@app.command("train")
def train(
    target: str = typer.Option(..., "--target", help="Target column name (e.g., is_high_value)"),
    seed: int = typer.Option(42, help="Random seed"),
    root: Path | None = typer.Option(None, help="Repo root"),
) -> None:
    """Train once and save a run folder under models/runs/."""
    run_dir = train_once(target=target, seed=seed, root=root)
    typer.echo(f"Saved run: {run_dir}")
'''