from __future__ import annotations

from pathlib import Path
import typer

from .sample_data import make_sample_feature_table

app = typer.Typer(help="ML Baseline System CLI")

@app.command()
def hello() -> None:
    """Sanity command."""
    typer.echo("Hello ✅")

@app.command()
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
