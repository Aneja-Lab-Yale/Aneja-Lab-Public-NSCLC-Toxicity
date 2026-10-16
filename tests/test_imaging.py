import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from nsclc_toxicity.imaging import LOBES, clinical_features, convert_ct, load_ct, preprocess_ct, qct_features, select_slabs


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


def test_physical_orientation_matches_training_las(tmp_path):
    array = np.arange(12 * 14 * 16, dtype=np.float32).reshape(12, 14, 16) - 1000
    ras = nib.Nifti1Image(array, np.eye(4))
    transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")),
                                               nib.orientations.axcodes2ornt(("L", "P", "S")))
    lps = ras.as_reoriented(transform)
    nib.save(ras, tmp_path / "ras.nii.gz")
    nib.save(lps, tmp_path / "lps.nii.gz")
    assert nib.aff2axcodes(load_ct(tmp_path / "ras.nii.gz").affine) == ("L", "A", "S")
    np.testing.assert_array_equal(preprocess_ct(tmp_path / "ras.nii.gz"), preprocess_ct(tmp_path / "lps.nii.gz"))
    expected = ((np.clip(array[::-1], -1000, 400) + 1000) / 1400).transpose(2, 1, 0)
    from scipy.ndimage import zoom
    np.testing.assert_allclose(preprocess_ct(tmp_path / "ras.nii.gz"), zoom(expected, [48/16, 96/14, 96/12], order=1))


def test_clinical_manifest_encodes_real_inputs_and_unknowns():
    import pandas as pd
    from nsclc_toxicity.schema import C_FEATURES
    frame = pd.DataFrame({"treated_lobe": ["RLL", "LUL", ""], "tumor_size_mm": [24, 35, np.nan],
                          "t_stage": ["cT1c", "T2a", ""], "total_dose_gy": [50, 60, np.nan],
                          "histology": ["adenocarcinoma", "SCC", ""]})
    result = clinical_features(frame)
    np.testing.assert_equal(result[list(C_FEATURES)].iloc[0].to_numpy(float), [24, 1, 50, 1, 1, 0])
    np.testing.assert_equal(result[list(C_FEATURES)].iloc[1].to_numpy(float), [35, 2, 60, 0, 0, 1])
    assert result[list(C_FEATURES)].iloc[2].isna().all()
    frame.loc[0, "treated_lobe"] = "wrong"
    with pytest.raises(ValueError, match="treated_lobe"):
        clinical_features(frame)


def test_nrrd_conversion_preserves_hu_and_geometry(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.GetImageFromArray(np.arange(6 * 8 * 10, dtype=np.int16).reshape(6, 8, 10) - 1000)
    image.SetSpacing((0.7, 0.8, 2.5))
    image.SetOrigin((10, -20, 30))
    source = tmp_path / "ct.nrrd"
    sitk.WriteImage(image, str(source))
    target = convert_ct(source, tmp_path / "ct.nii.gz")
    converted = sitk.ReadImage(str(target))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(converted))
    np.testing.assert_allclose(converted.GetSpacing(), image.GetSpacing())
    np.testing.assert_allclose(converted.GetOrigin(), image.GetOrigin())


def test_dicom_conversion_orders_slices_and_applies_hu_scaling(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    study, series = generate_uid(), generate_uid()
    folder = tmp_path / "dicom"
    folder.mkdir()
    stored = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    # Reverse filenames to ensure physical slice order, not filename order, wins.
    for z in range(6):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        data = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        data.SOPClassUID, data.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        data.StudyInstanceUID, data.SeriesInstanceUID = study, series
        data.Modality = "CT"
        data.Rows, data.Columns = 8, 10
        data.SamplesPerPixel = 1
        data.PhotometricInterpretation = "MONOCHROME2"
        data.BitsAllocated, data.BitsStored, data.HighBit, data.PixelRepresentation = 16, 16, 15, 0
        data.PixelSpacing = [0.8, 0.7]
        data.SliceThickness = 2.5
        data.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        data.ImagePositionPatient = [10, -20, 30 + 2.5*z]
        data.InstanceNumber = z + 1
        data.RescaleSlope, data.RescaleIntercept = 2, -1024
        data.PixelData = stored[z].tobytes()
        data.save_as(folder / f"{5-z}.dcm", enforce_file_format=True)
    target = convert_ct(folder, tmp_path / "converted.nii.gz")
    converted = sitk.ReadImage(str(target))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(converted), stored.astype(np.int32) * 2 - 1024)
    np.testing.assert_allclose(converted.GetSpacing(), [0.7, 0.8, 2.5])
