"""Fit the selected Q7 and clinical branches and evaluate held-out patients."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import predict_fold
from .schema import C_FEATURES, FOLDS, HORIZONS, Q_FEATURES, dl_columns, eligible_cohort, targets, validate

SEED = 20260720


def patient_weights(ids: pd.Series, target: np.ndarray) -> np.ndarray:
    counts = ids.map(ids.value_counts()).to_numpy(float)
    weights = 1.0 / counts
    positives = weights[target == 1].sum()
    negatives = weights[target == 0].sum()
    if positives > 0:
        weights[target == 1] *= min(8.0, negatives / positives)
    return weights


def export_logistic(model: Pipeline, input_dim: int) -> dict:
    imputer = model.named_steps["impute"]
    scale = model.named_steps["scale"]
    estimator = model.named_steps["model"]
    stats = imputer.statistics_
    return {
        "input_dim": input_dim,
        "median": np.nan_to_num(stats, nan=0.0).tolist(),
        "retained_columns": np.flatnonzero(~np.isnan(stats)).tolist(),
        "indicator_columns": imputer.indicator_.features_.tolist(),
        "mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
        "coefficient": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
    }


def fit_logistic(matrix: np.ndarray, target: np.ndarray, ids: pd.Series, c: float) -> dict:
    if not len(target):
        return {"constant": 0.5}
    if len(np.unique(target)) < 2:
        return {"constant": float(np.mean(target))}
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000)),
    ])
    model.fit(matrix, target.astype(int), model__sample_weight=patient_weights(ids, target))
    return export_logistic(model, matrix.shape[1])


def fit_fold(train: pd.DataFrame, fold: int) -> dict:
    _, hazard = targets(train)
    q = []
    matrices, ys, identifiers = [], [], []
    for i in range(4):
        observed = np.isfinite(hazard[:, i])
        current = train.loc[observed]
        q.append(fit_logistic(current[list(Q_FEATURES)].to_numpy(float), hazard[observed, i], current["patient_id"], 0.30))
        matrices.append(np.concatenate([current[list(C_FEATURES)].to_numpy(float), np.tile(np.eye(4)[i], (len(current), 1))], axis=1))
        ys.append(hazard[observed, i])
        identifiers.append(current["patient_id"])
    clinical = fit_logistic(np.concatenate(matrices), np.concatenate(ys), pd.concat(identifiers, ignore_index=True), 0.03)
    baseline = train.loc[train["is_baseline"].eq(1)]
    # This is the fitted empirical-CDF transform, with no patient linkage or outcomes.
    # As in development, its reference is direct cumulative risk on training baselines.
    dl_risk = 1.0 - np.cumprod(1.0 - baseline[dl_columns(fold)].to_numpy(float), axis=1)
    normalizer = [np.sort(dl_risk[:, i]).tolist() for i in range(4)]
    return {"fold": fold, "q": q, "clinical": clinical, "dl_normalizer": normalizer}


def assign_folds(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    frame = frame.copy()
    if "fold" in frame:
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise")
        if not frame["fold"].isin(FOLDS).all() or set(frame["fold"]) != set(FOLDS):
            raise ValueError("fold must contain all of 1, 2, 3.")
        if frame.groupby("patient_id")["fold"].nunique().gt(1).any():
            raise ValueError("A patient must stay in one fold across all scans.")
        return frame
    patients = frame.drop_duplicates("patient_id").sort_values("patient_id").reset_index(drop=True)
    event = (patients["event_days"].notna() & patients["event_days"].le(365)).astype(int)
    if event.value_counts().reindex([0, 1], fill_value=0).min() < 3:
        raise ValueError("At least three patients with and without a one-year event are needed for three folds.")
    mapping = {}
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(patients, event), 1):
        mapping.update({patients.loc[i, "patient_id"]: fold for i in test})
    frame["fold"] = frame["patient_id"].map(mapping)
    return frame


def train_ensemble(frame: pd.DataFrame) -> dict:
    return {
        "format_version": 1,
        "model_id": "nsclc_sbrt_toxicity_q7_c6_three_fold_ensemble",
        "horizons_days": list(HORIZONS), "q_features": list(Q_FEATURES),
        "clinical_features": list(C_FEATURES),
        "fusion": {"imaging_dl_to_q": [1, 2], "clinical_mass": [0.2, 0.4, 0.6, 0.8]},
        "folds": [fit_fold(frame.loc[frame["fold"].ne(fold)], fold) for fold in FOLDS],
    }


def cross_validate(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    rows = []
    for fold in model["folds"]:
        test = frame.loc[frame["fold"].eq(fold["fold"])].copy()
        cumulative, _ = targets(test)
        for branch, values in predict_fold(test, fold).items():
            for i, horizon in enumerate(HORIZONS):
                observed = np.isfinite(cumulative[:, i])
                out = test.loc[observed, ["patient_id", "scan_id", "scan_day", "is_baseline", "fold"]].copy()
                out["horizon_days"] = horizon
                out["model"] = branch
                out["target"] = cumulative[observed, i].astype(int)
                out["score"] = values[observed, i]
                rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all_scans", "baseline", "follow_up"):
        current = predictions if scope == "all_scans" else predictions.loc[predictions["is_baseline"].eq(1 if scope == "baseline" else 0)]
        for (branch, fold, horizon), group in current.groupby(["model", "fold", "horizon_days"], sort=True):
            weights = 1.0 / group["patient_id"].map(group["patient_id"].value_counts()).to_numpy(float)
            auc = float(roc_auc_score(group["target"], group["score"], sample_weight=weights)) if group["target"].nunique() == 2 else np.nan
            rows.append({"scope": scope, "model": branch, "fold": fold, "horizon_days": horizon,
                         "patients": group["patient_id"].nunique(), "scans": len(group), "td_auc": auc})
    return pd.DataFrame(rows)


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")


def run_experiment(frame: pd.DataFrame, output: Path, seed: int = SEED) -> dict:
    frame = assign_folds(eligible_cohort(validate(frame, outcomes=True)), seed)
    if frame.empty:
        raise ValueError("No eligible patients remain after exclusions.")
    model = train_ensemble(frame)
    predictions = cross_validate(frame, model)
    metrics = summarize(predictions)
    output.mkdir(parents=True, exist_ok=True)
    save_model(model, output / "model.json")
    frame[["patient_id", "fold"]].drop_duplicates().to_csv(output / "patient_folds.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    # Require support in all three folds at each horizon before calculating iAUC.
    complete = metrics.groupby(["scope", "model", "horizon_days"])["td_auc"].agg(["count", "mean"])
    complete["td_auc"] = complete["mean"].where(complete["count"].eq(3))
    summary = complete[["td_auc"]].unstack("horizon_days").reindex(columns=pd.MultiIndex.from_product([["td_auc"], HORIZONS]))
    summary.columns = [f"td_auc_{h}d" for h in HORIZONS]
    summary["integrated_auc"] = np.trapz(summary.to_numpy(float), HORIZONS, axis=1) / (HORIZONS[-1] - HORIZONS[0])
    summary.reset_index().to_csv(output / "performance.csv", index=False)
    report = {"patients": int(frame["patient_id"].nunique()), "scans": len(frame),
              "seed": seed, "folds": 3, "dl_training": "provided fold-specific CT hazards; see docs/CT_WORKFLOW.md",
              "estimates": "development; not an independent external validation"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_performance(summary.reset_index(), output / "performance.png")
    return report


def plot_performance(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"DL": "#3E6F9E", "Q": "#4D9387", "Clinical": "#B45F64", "Imaging": "#91A9BE", "Multimodal": "#75519A"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, scope in zip(axes, ("all_scans", "baseline", "follow_up")):
        for _, row in summary.loc[summary["scope"].eq(scope)].iterrows():
            ax.plot([3, 6, 9, 12], row[[f"td_auc_{h}d" for h in HORIZONS]].to_numpy(float), marker="o", label=row["model"], color=colors[row["model"]])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.set(title=scope.replace("_", " ").capitalize(), xlabel="Months after treatment", ylim=(0, 1.02), xticks=[3, 6, 9, 12])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean fold time-dependent AUC")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
