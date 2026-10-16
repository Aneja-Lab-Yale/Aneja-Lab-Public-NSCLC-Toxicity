import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from nsclc_toxicity.imaging import LOBES, preprocess_ct, qct_features, select_slabs


def test_q7_uses_original_hu_and_exact_masks(tmp_path):
    array = np.full((10, 10, 6), -800, dtype=np.float32)
    array[:2] = -200
    array[2:4] = -1000
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    for i, name in enumerate(LOBES.values()):
        mask = np.zeros_like(array, dtype=np.uint8)
        mask[2*i:2*i+2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / f"{name}.nii.gz")
    q = qct_features(path, tmp_path, "RUL")
    assert q["lung_volume_ml"] == pytest.approx(0.6)
    assert q["lung_mean"] == pytest.approx(-720)
    assert q["lung_density_frac_lt_m950_nonoverlap"] == pytest.approx(0.2)
    assert q["target_minus_other_p95_hu"] == pytest.approx(600)
    assert q["target_minus_other_frac_ge_m300"] == pytest.approx(1)
    mask_path = tmp_path / "lung_upper_lobe_right.nii.gz"
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.Nifti1Image(np.ones_like(array), affine), mask_path)
    with pytest.raises(ValueError, match="exact CT voxel grid"):
        qct_features(path, tmp_path, "RUL")


def test_volume_preprocessing_and_slab_boundaries(tmp_path):
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.full((12, 14, 16), -300, dtype=np.float32), np.eye(4)), path)
    volume = preprocess_ct(path)
    assert volume.shape == (48, 96, 96)
    np.testing.assert_allclose(volume, 0.5)
    x = torch.arange(48, dtype=torch.float32).reshape(1, 1, 48, 1, 1)
    slabs = select_slabs(x)
    assert slabs.shape == (16, 3, 1, 1)
    np.testing.assert_equal(slabs[0, :, 0, 0].numpy(), [0, 0, 1])
    np.testing.assert_equal(slabs[-1, :, 0, 0].numpy(), [46, 47, 47])
