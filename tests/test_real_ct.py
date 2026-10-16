"""Run after download-demo-ct and infer-ct; this exercises real public voxels."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("NSCLC_RUN_PUBLIC_CT_TEST") != "1", reason="Real public CT integration is opt-in; see public-ct CI job")


def test_real_public_ct_features_are_measured_from_the_images():
    from nsclc_toxicity.download import sha256
    from nsclc_toxicity.imaging import LOBES, load_ct
    from nsclc_toxicity.public_ct import CT_SHA256
    from nsclc_toxicity.schema import Q_FEATURES, FOLDS, dl_columns
    assert sha256(Path("data/public_ct/CT-chest.nrrd")) == CT_SHA256
    prepared = pd.read_csv("results/public_ct/prepared.csv")
    assert len(prepared) == 1
    assert prepared[list(Q_FEATURES)].notna().all().all()
    assert prepared[[c for f in FOLDS for c in dl_columns(f)]].notna().all().all()
    ct = load_ct(next(Path("results/public_ct/converted_ct").glob("*.nii.gz")))
    values = ct.get_fdata(dtype=np.float32)
    mask_dir = next(Path("results/public_ct/lobe_masks").glob("*/lung_upper_lobe_right.nii.gz")).parent
    masks = [load_ct(mask_dir / f"{lobe}.nii.gz").get_fdata() > 0 for lobe in LOBES.values()]
    assert all(np.count_nonzero(mask) >= 10 for mask in masks)
    lung = np.logical_or.reduce(masks)
    assert prepared.iloc[0]["lung_volume_ml"] == pytest.approx(np.count_nonzero(lung) * np.prod(ct.header.get_zooms()) / 1000)
    assert prepared.iloc[0]["lung_mean"] == pytest.approx(float(values[lung].mean()))
    predictions = pd.read_csv("results/public_ct/predictions.csv")
    final = predictions.loc[predictions["model"].eq("Multimodal")]
    assert len(final) == 4
    assert final["score"].between(0, 1).all()
    assert set(final["horizon_days"]) == {90, 183, 274, 365}
