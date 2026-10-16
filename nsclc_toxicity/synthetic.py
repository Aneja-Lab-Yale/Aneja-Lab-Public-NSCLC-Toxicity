"""Generate the public example without reading any source patient data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .schema import C_FEATURES, FOLDS, HORIZONS, Q_FEATURES, dl_columns


def make_sample() -> pd.DataFrame:
    rng = np.random.default_rng(716)
    rows = []
    for i in range(72):
        event = [45.0, 140.0, 230.0, 320.0][i % 12] if i % 12 < 4 else np.nan
        censor = 400.0 if np.isfinite(event) or i % 7 else 250.0
        latent = (0.8 if np.isfinite(event) else -0.3) + rng.normal(0, 0.7)
        histology = int(rng.integers(0, 3))
        clinical = [float(rng.integers(12, 47)), float(rng.integers(1, 4)), float(rng.choice([48, 50, 54, 60])),
                    float(rng.integers(0, 2)), float(histology == 0), float(histology == 1)]
        q = [float(rng.uniform(2800, 6500)), float(rng.uniform(-850, -650)), float(rng.uniform(0.01, 0.22)),
             float(rng.uniform(0.04, 0.25)), float(rng.normal(50 + 20 * latent, 100)),
             float(rng.uniform(-0.1, 0.2)), float(rng.normal(0, 8))]
        # Demonstrate train-only missing-value imputation.
        if i % 11 == 0:
            q[4:] = [np.nan] * 3
        for visit, day in enumerate([-30.0, 100.0, 220.0]):
            if day >= min(event if np.isfinite(event) else np.inf, censor):
                continue
            row = {"patient_id": f"SYN{i+1:03}", "scan_id": f"SYN{i+1:03}_V{visit}",
                   "scan_day": day, "is_baseline": int(visit == 0),
                   **dict(zip(Q_FEATURES, q)), **dict(zip(C_FEATURES, clinical)),
                   "event_days": event, "censor_days": censor,
                   "progression_event_by_365d": 0, "fold": (i // 12) % 3 + 1}
            for fold in FOLDS:
                # Simulated model outputs; no CT or learned private parameters enter here.
                hazard = expit(-2.7 + 0.5 * latent + 0.15 * visit + rng.normal(0, 0.18, 4))
                row.update(dict(zip(dl_columns(fold), hazard)))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    destination = Path("data/sample.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_sample().to_csv(destination, index=False, float_format="%.10g")
    print(f"Wrote {destination}")
