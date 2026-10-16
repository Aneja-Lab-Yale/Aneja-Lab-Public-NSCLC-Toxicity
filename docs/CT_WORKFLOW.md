# CT inputs and retraining

## CT → masks → features → prediction

The main environment includes image processing and automatic lung-lobe segmentation. To add them to a previous installation (Python ≥3.10):

```bash
python -m pip install -e ".[imaging,segmentation,notebook]"
```

Use [`data/clinical_manifest_example.csv`](../data/clinical_manifest_example.csv) as your template. Required columns are `patient_id`, `scan_id`, `scan_day`, `is_baseline`, `ct_path`, `tumor_size_mm`, `t_stage`, `total_dose_gy`, `treated_lobe`, and `histology`. The earlier six encoded clinical columns are also accepted. Include exactly one selected pretreatment baseline per patient. Repeat baseline clinical values on follow-up rows; use each follow-up CT's day from treatment start. No toxicity outcomes are needed for inference.

```bash
python -m nsclc_toxicity infer-ct --manifest private_data/clinical.csv --output results/my_case
```

This automatically downloads verified prediction weights, reads/converts the CT, creates all five baseline lobe masks with the free TotalSegmentator `total` task, measures Q7, computes all 12 DINOv3 hazard outputs, and runs the final three-fold ensemble. It saves `prepared.csv`, `predictions.csv`, and `run.json`. Images, masks, model weights, and clinical data stay local; only downloads require internet. Segmentation usage reporting is disabled by this wrapper.

The default segmentation uses the development protocol’s 6 mm (`fastest`) resolution, then returns masks to the exact input CT grid. Add `--segmentation-resolution 3` for 3 mm or `--full-resolution` for the slower 1.5 mm model, or `--device cuda` for a CUDA GPU. Mask resolution affects measured features; review masks and use a consistent, validated segmentation protocol for a study. This preprocessing does not determine tumor location, prescribe treatment, or validate a CT's clinical suitability.

## Image formats and orientation

- **NIfTI:** `.nii` / `.nii.gz` with intensities in HU and correct physical geometry.
- **NRRD/MHA:** scalar 3D HU images are converted to NIfTI without windowing/resampling.
- **DICOM:** `ct_path` is a folder containing exactly one CT image series. SimpleITK orders slices and applies rescale slope/intercept. Separate other series/localizers first. Multi-series folders are rejected.

Use full lung coverage. The volume must be a diagnostic CT in HU, not a screenshot, projection, contrast-normalized array, or MR image. Compression of HU values into 8-bit display pixels loses information the model needs.

The development CT arrays were **LAS**. The loader flips/permutates voxel axes into LAS without interpolation before image features and neural preprocessing. This handles physically equivalent RAS/LPS files consistently. Oblique acquisition and clinical protocol differences still require review; this is not a clinical harmonization or quality-assurance system.

## Use externally reviewed masks

To supply your own masks, retain `mask_dir` in the older [`ct_manifest_example.csv`](../data/ct_manifest_example.csv), then run:

```bash
python -m nsclc_toxicity download-weights --imaging
python -m nsclc_toxicity prepare-ct --manifest private_data/ct_manifest.csv --output results/prepared.csv
python -m nsclc_toxicity predict --input results/prepared.csv --output results/predictions.csv
```

Each baseline mask folder contains five binary NIfTI masks: `lung_upper_lobe_right`, `lung_middle_lobe_right`, `lung_lower_lobe_right`, `lung_upper_lobe_left`, and `lung_lower_lobe_left`, each ending in `.nii.gz`. After axis alignment, their voxel grids and affines must match the CT (1e-3 affine tolerance). Mismatches are rejected. `prepare-ct --auto-masks` generates masks instead. Follow-up masks are unnecessary because Q7 is baseline-only.

A blank `treated_lobe` leaves the three lobe-contrast features missing and uses the fitted imputer; a known location is preferable. Unknown clinical values stay blank. The code computes location/histology encodings from the manifest, not from the CT. Invalid location codes and contradictory encodings are rejected.

## Retrain every branch

Add `event_days`, `censor_days`, and `progression_event_by_365d` to the manifest. These durations start at treatment, not the scan. An optional `fold` column locks patient folds 1–3. At least three event and three non-event patients are needed for automatic three-fold assignment.

```bash
python -m nsclc_toxicity train-ct --manifest private_data/clinical.csv --auto-masks --output results/ct_training --device cuda
```

Each CT model starts from the public [timm DINOv3 ViT-S/16 backbone](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), trains adapters/head on baseline patients in the other two folds, scores every scan, and runs the selected tabular/fusion experiment with the same folds. `--initial-backbone /path/to/timm_state.pt` supplies a matching pretrained backbone offline. Reviewed masks can be used by omitting `--auto-masks` and supplying `mask_dir`.

Outputs include `prepared.csv`, three `dino_fold*.pt` checkpoints, training losses, and `experiment/` with fitted tabular weights, held-out predictions, metrics, and a figure. For your trained CT models, use `prepare-ct --weights results/ct_training`; for your fitted ensemble, use `predict --weights results/ct_training/experiment/model.json`. `infer-ct` is the shortcut for the published weights. Training stores volumes in memory and can take hours; GPU training is recommended.

## Exact feature definitions

DL clips HU to [-1000, 400], scales to [0, 1], transposes LAS XYZ to ZYX, and linearly resizes to 48 × 96 × 96 without antialiasing. Sixteen uniformly spaced three-slice slabs pass through a learned 1×1 adapter, bilinear resize to 224 × 224, DINOv3, equal-weight pooling, and the hazard head. No ImageNet normalization is added.

Q7 uses baseline CT values on the exact mask grid. Whole lung is the five-lobe union. Volume uses voxel spacing; mean HU and density fractions use original HU. Fractions are HU < -950 and -700 ≤ HU < -300. Lobe contrasts are target minus other-lobe union, with target voxels removed from the other-lobe mask: P95 HU, fraction HU ≥ -300, and signed Sobel mean. Sobel uses clipped HU and `scipy.ndimage.sobel` with default axis/mode on every occupied axial slice in LAS orientation, then equal-weight averaging of within-mask slice means.
