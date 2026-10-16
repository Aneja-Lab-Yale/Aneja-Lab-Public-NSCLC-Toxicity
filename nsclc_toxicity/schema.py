"""Input validation shared by inference, training, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (90, 183, 274, 365)
LOWER_EDGES = (0, 90, 183, 274)
FOLDS = (1, 2, 3)
Q_FEATURES = (
    "lung_volume_ml", "lung_mean", "lung_density_frac_lt_m950_nonoverlap",
    "lung_density_frac_m700_to_m300_nonoverlap", "target_minus_other_p95_hu",
    "target_minus_other_frac_ge_m300", "target_minus_other_signed_all_slice_2d_sobel_mean",
)
C_FEATURES = (
    "clinical_tumor_size_mm", "clinical_t_stage_ordinal",
    "clinical_total_dose_reconciled_gy", "clinical_lower_lobe_corrected",
    "clinical_histology_adenocarcinoma_corrected", "clinical_histology_squamous_corrected",
)
IDENTITY = ("patient_id", "scan_id", "scan_day", "is_baseline")
OUTCOMES = ("event_days", "censor_days", "progression_event_by_365d")


def dl_columns(fold: int) -> list[str]:
    return [f"dl_fold{fold}_hazard_{h}d" for h in HORIZONS]


def require_columns(frame: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))


def validate(frame: pd.DataFrame, *, outcomes: bool = False, dl: bool = True) -> pd.DataFrame:
    columns = [*IDENTITY, *Q_FEATURES, *C_FEATURES]
    if dl:
        columns += [c for fold in FOLDS for c in dl_columns(fold)]
    if outcomes:
        columns += list(OUTCOMES)
    else:
        columns += [col for col in OUTCOMES if col in frame]
    require_columns(frame, columns)
    frame = frame.copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("CSV has no rows.")
    for col in ("patient_id", "scan_id"):
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} must contain a nonempty study identifier.")
        frame[col] = frame[col].astype(str)
    if frame.duplicated(["patient_id", "scan_id"]).any():
        raise ValueError("Each patient_id / scan_id pair must be unique.")
    for col in set(columns).difference(("patient_id", "scan_id")):
        try:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{col} must be numeric; leave missing values blank.") from exc
        if np.isinf(frame[col].to_numpy(float)).any():
            raise ValueError(f"{col} contains infinity.")
    if not frame["is_baseline"].isin([0, 1]).all():
        raise ValueError("is_baseline must be 0 or 1.")
    if not frame.groupby("patient_id")["is_baseline"].sum().eq(1).all():
        raise ValueError("Include exactly one baseline row per patient.")
    if frame["scan_day"].isna().any() or frame["scan_day"].gt(365).any():
        raise ValueError("scan_day must be finite and at most 365 days from treatment start.")
    baseline = frame["is_baseline"].eq(1)
    if frame.loc[baseline, "scan_day"].gt(0).any() or frame.loc[~baseline, "scan_day"].lt(0).any():
        raise ValueError("Baseline scan_day must be <= 0; follow-up scan_day must be >= 0.")
    for col in (*Q_FEATURES, *C_FEATURES):
        if frame.groupby("patient_id")[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"{col} must repeat the baseline value on every row of a patient.")
    for col in C_FEATURES[3:]:
        if not frame[col].dropna().isin([0, 1]).all():
            raise ValueError(f"{col} must be 0, 1, or blank.")
    for col in Q_FEATURES[2:4]:
        if not frame[col].dropna().between(0, 1).all():
            raise ValueError(f"{col} must be a fraction between 0 and 1.")
    if not frame[Q_FEATURES[5]].dropna().between(-1, 1).all():
        raise ValueError("target_minus_other_frac_ge_m300 must be between -1 and 1.")
    if dl:
        for fold in FOLDS:
            for col in dl_columns(fold):
                if frame[col].isna().any() or not frame[col].between(0, 1).all():
                    raise ValueError(f"{col} must contain probabilities from 0 to 1 on every row.")
    if outcomes:
        for col in OUTCOMES:
            if frame.groupby("patient_id")[col].nunique(dropna=False).gt(1).any():
                raise ValueError(f"{col} must be identical within each patient.")
        if not frame["progression_event_by_365d"].isin([0, 1]).all():
            raise ValueError("progression_event_by_365d must be 0 or 1 (no missing values).")
        if frame["censor_days"].isna().any() or frame["censor_days"].lt(0).any():
            raise ValueError("censor_days must be a nonnegative observation duration.")
        events = frame["event_days"].dropna()
        if events.le(0).any():
            raise ValueError("event_days must be > 0, or blank when no toxicity was observed.")
        if (frame["event_days"].notna() & frame["event_days"].gt(frame["censor_days"])).any():
            raise ValueError("event_days cannot be later than censor_days.")
    return frame


def eligible_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply progression exclusion and remove scans after toxicity or observation."""
    return frame.loc[
        frame["progression_event_by_365d"].eq(0)
        & (frame["event_days"].isna() | frame["scan_day"].lt(frame["event_days"]))
        & frame["scan_day"].le(frame["censor_days"])
    ].copy().reset_index(drop=True)


def targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Treatment-origin cumulative and interval targets; NaN means unobserved."""
    scan = frame["scan_day"].to_numpy(float)
    event = frame["event_days"].to_numpy(float)
    censor = frame["censor_days"].to_numpy(float)
    cumulative = np.full((len(frame), 4), np.nan)
    hazards = cumulative.copy()
    for i, (lower, horizon) in enumerate(zip(LOWER_EDGES, HORIZONS)):
        control = (event > horizon) | (np.isnan(event) & (censor >= horizon))
        case = (event > scan) & (event <= horizon)
        observed = (scan < horizon) & (case | control)
        cumulative[observed, i] = case[observed].astype(float)
        at_risk = np.isnan(event) | (event > lower)
        interval_case = (event > np.maximum(scan, lower)) & (event <= horizon)
        observed = (scan < horizon) & at_risk & (interval_case | control)
        hazards[observed, i] = interval_case[observed].astype(float)
    return cumulative, hazards
