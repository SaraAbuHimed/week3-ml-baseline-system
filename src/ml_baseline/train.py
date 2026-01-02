'''
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Paths, TrainConfig
from .io import read_tabular

def _utc_run_id(task: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return f"{ts}__{task}__seed{seed}"


def _compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return out


def train_once(*, target: str, seed: int = 42, root: Path | None = None) -> Path:
    """
    1) Load data/processed/features.*
    2) Split train/holdout
    3) Fit dummy baseline + metrics
    4) Fit real sklearn pipeline
    5) Save run folder: models/runs/<run_id>/
    6) Write models/registry/latest.txt
    7) Print run dir (caller prints it)
    """
    paths = Paths.from_repo_root() if root is None else Paths(root=root)

    # ---- load features table (csv/parquet) ----
    features_path = paths.data_processed_dir / "features.csv"
    if not features_path.exists():
        # allow parquet fallback if you want
        parquet_path = paths.data_processed_dir / "features.parquet"
        if parquet_path.exists():
            features_path = parquet_path
        else:
            raise FileNotFoundError(
                "Missing features table. Run: uv run ml-baseline make-sample-data"
            )

    df = read_tabular(features_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in columns: {list(df.columns)}")

    # ---- separate X / y ----
    y = df[target].astype(int)
    drop_cols = [target]
    # common passthrough ID columns
    for col in ["user_id"]:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)

    # ---- split ----
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ---- baseline dummy ----
    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_hold)
    baseline_metrics = _compute_metrics(y_hold, dummy_pred)

    # ---- real pipeline ----
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_hold)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_hold)[:, 1]

    model_metrics = _compute_metrics(y_hold, pred, y_proba=proba)

    # ---- save run ----
    run_id = _utc_run_id("classification", seed)
    run_dir = paths.models_runs_dir / run_id
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(target=target, seed=seed)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    (run_dir / "metrics" / "baseline_holdout.json").write_text(
        json.dumps(baseline_metrics, indent=2)
    )
    (run_dir / "metrics" / "model_holdout.json").write_text(
        json.dumps(model_metrics, indent=2)
    )

    joblib.dump(model, run_dir / "model" / "model.joblib")

    # ---- registry/latest.txt ----
    paths.models_registry_dir.mkdir(parents=True, exist_ok=True)
    (paths.models_registry_dir / "latest.txt").write_text(str(run_dir))

    return run_dir
'''


from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainCfg

log = logging.getLogger("ml_baseline.train")


def _pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze failed: {e!r}\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_run_id(run_tag: str, session_id: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}__{run_tag}__session{session_id}"


def _load_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported features file: {path} (use .csv or .parquet)")


def _compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return out


def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    """
    Executable version of the instructor structure:
    - creates models/runs/<run_id>/*
    - loads features
    - trains a baseline classifier
    - saves model + metrics + schema + env snapshot
    - writes models/registry/latest.txt
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_id = _utc_run_id(run_tag=run_tag, session_id=cfg.session_id)
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "tables", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    log.info("Run dir: %s", run_dir)

    # --- load data ---
    df = _load_features(cfg.features_path)
    if cfg.target not in df.columns:
        raise ValueError(f"Missing target: {cfg.target}. Columns: {list(df.columns)}")

    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    if cfg.time_col:
        if cfg.time_col not in df.columns:
            raise ValueError(f"Missing time_col: {cfg.time_col}")
        df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # --- schema contract ---
    id_cols_present = [c for c in cfg.id_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in {cfg.target, *id_cols_present}]

    schema = {
        "task": cfg.task,
        "target": cfg.target,
        "features_path": str(cfg.features_path),
        "features_sha256": _sha256(cfg.features_path) if cfg.features_path.exists() else None,
        "required_feature_columns": feature_cols,
        "optional_id_columns": id_cols_present,
        "feature_dtypes": {c: str(df[c].dtype) for c in feature_cols},
        "policy_unknown_categories": "tolerant (OneHotEncoder handle_unknown=ignore)",
        "forbidden_columns": [cfg.target],
    }
    (run_dir / "schema" / "input_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    # --- env capture ---
    (run_dir / "env" / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    env_meta = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
    }
    (run_dir / "env" / "env_meta.json").write_text(
        json.dumps(env_meta, indent=2), encoding="utf-8"
    )

    # --- build X/y ---
    y = df[cfg.target].astype(int)
    X = df[feature_cols]

    # --- split ---
    test_size = 1.0 - cfg.train_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=cfg.session_id,
        stratify=y if cfg.task == "classification" else None,
    )

    # --- preprocess ---
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )

    # --- baseline model ---
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=cfg.session_id)),
        ]
    )

    # dummy baseline (floor)
    dummy = DummyClassifier(strategy="most_frequent", random_state=cfg.session_id)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_hold)
    dummy_metrics = _compute_metrics(y_hold, dummy_pred)

    # train real model
    model.fit(X_train, y_train)
    pred = model.predict(X_hold)
    proba = model.predict_proba(X_hold)[:, 1] if hasattr(model, "predict_proba") else None
    model_metrics = _compute_metrics(y_hold, pred, y_proba=proba)

    # --- tables: holdout_predictions ---
    holdout_predictions = pd.DataFrame({
        "y_true": y_hold.reset_index(drop=True).astype(int),
        "y_pred": pd.Series(pred).reset_index(drop=True).astype(int),
    })

    if proba is not None:
        holdout_predictions["y_score"] = (
            pd.Series(proba).reset_index(drop=True).astype(float)
        )

    holdout_predictions.to_csv(
        run_dir / "tables" / "holdout_predictions.csv",
        index=False,
    )

    # --- tables: holdout_input (features-only; NO target) ---
    holdout_input = X_hold.reset_index(drop=True).copy()

    holdout_input.to_csv(
        run_dir / "tables" / "holdout_input.csv",
        index=False,
    )

    # --- save artifacts ---
    (run_dir / "metrics" / "baseline_holdout.json").write_text(
        json.dumps(dummy_metrics, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(model_metrics, indent=2) + "\n", encoding="utf-8",
    )



    (run_dir / "tables" / "holdout_preview.csv").write_text(
        pd.concat([X_hold.reset_index(drop=True), y_hold.reset_index(drop=True)], axis=1).head(25).to_csv(index=False),
        encoding="utf-8",
    )

    joblib.dump(model, run_dir / "model" / "model.joblib")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    # --- registry/latest.txt ---
    registry_dir = root / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "latest.txt").write_text(str(run_dir), encoding="utf-8")

    log.info("Saved run: %s", run_dir)
    return run_dir
