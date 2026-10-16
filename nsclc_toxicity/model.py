"""Prediction with numerical model parameters; no executable pickle files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .schema import C_FEATURES, FOLDS, HORIZONS, Q_FEATURES, dl_columns, validate


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Weights not found at {path}. Run: python -m nsclc_toxicity download-weights")
    model = json.loads(path.read_text())
    if (model.get("format_version") != 1 or model.get("horizons_days") != list(HORIZONS)
            or model.get("q_features") != list(Q_FEATURES) or model.get("clinical_features") != list(C_FEATURES)
            or [f["fold"] for f in model.get("folds", [])] != list(FOLDS)):
        raise ValueError("The weights do not match this model/input format.")
    return model


def logistic_predict(parameters: dict, matrix: np.ndarray) -> np.ndarray:
    if "constant" in parameters:
        return np.full(len(matrix), parameters["constant"], dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != parameters["input_dim"]:
        raise ValueError("Incorrect number of model features.")
    missing = np.isnan(matrix)
    # Match SimpleImputer's learned retained features and missingness indicators.
    kept = parameters["retained_columns"]
    filled = np.where(missing, np.asarray(parameters["median"]), matrix)[:, kept]
    indicators = missing[:, parameters["indicator_columns"]].astype(float)
    transformed = np.concatenate([filled, indicators], axis=1)
    scaled = (transformed - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])
    return expit(scaled @ np.asarray(parameters["coefficient"]) + parameters["intercept"])


def accumulate(hazards: np.ndarray, scan_days: np.ndarray) -> np.ndarray:
    """Remaining-bin risk; a partially elapsed bin is included in full."""
    eligible = np.asarray(HORIZONS)[None, :] > np.asarray(scan_days)[:, None]
    return 1.0 - np.cumprod(np.where(eligible, 1.0 - hazards, 1.0), axis=1)


def percentile(reference: list[float], values: np.ndarray) -> np.ndarray:
    ordered = np.asarray(reference, dtype=float)
    if len(ordered) == 0 or not np.isfinite(ordered).all() or (np.diff(ordered) < 0).any():
        raise ValueError("Invalid learned DL normalization parameters.")
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (0.5 * (left + right) + 0.5) / (len(ordered) + 1.0)


def predict_fold(frame: pd.DataFrame, fold_model: dict) -> dict[str, np.ndarray]:
    n = len(frame)
    q = np.column_stack([
        logistic_predict(p, frame[list(Q_FEATURES)].to_numpy(float)) for p in fold_model["q"]
    ])
    clinical = np.column_stack([
        logistic_predict(fold_model["clinical"], np.concatenate([
            frame[list(C_FEATURES)].to_numpy(float), np.tile(np.eye(4)[i], (n, 1))
        ], axis=1)) for i in range(4)
    ])
    q = accumulate(q, frame["scan_day"].to_numpy(float))
    clinical = accumulate(clinical, frame["scan_day"].to_numpy(float))
    dl_raw = accumulate(frame[dl_columns(fold_model["fold"])].to_numpy(float), frame["scan_day"].to_numpy(float))
    dl = np.column_stack([percentile(ref, dl_raw[:, i]) for i, ref in enumerate(fold_model["dl_normalizer"])])
    imaging = (dl + 2.0 * q) / 3.0
    mass = np.array([0.2, 0.4, 0.6, 0.8])
    return {"DL": dl, "Q": q, "Clinical": clinical, "Imaging": imaging,
            "Multimodal": (1.0 - mass) * imaging + mass * clinical}


def predict(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    frame = validate(frame)
    # If outcomes are supplied, prevent predictions for already known events/progression.
    if "event_days" in frame and (frame["event_days"].notna() & frame["scan_day"].ge(frame["event_days"])).any():
        raise ValueError("Inference requires event-free scans. Remove scans at/after toxicity.")
    if "progression_event_by_365d" in frame and frame["progression_event_by_365d"].eq(1).any():
        raise ValueError("This model excludes patients with progression within 365 days.")
    parts = [predict_fold(frame, fold) for fold in model["folds"]]
    rows = []
    for branch in parts[0]:
        score = np.mean([part[branch] for part in parts], axis=0)
        for i, horizon in enumerate(HORIZONS):
            valid = frame["scan_day"].lt(horizon)
            out = frame.loc[valid, ["patient_id", "scan_id", "scan_day"]].copy()
            out["horizon_days"] = horizon
            out["model"] = branch
            out["score"] = score[valid, i]
            rows.append(out)
    return pd.concat(rows, ignore_index=True)
