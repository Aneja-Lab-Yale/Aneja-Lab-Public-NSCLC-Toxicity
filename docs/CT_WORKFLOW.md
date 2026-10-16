# CT inputs and retraining

This advanced workflow needs CT images in Hounsfield units, one selected pretreatment scan per patient, optional event-free follow-up scans, and five lung-lobe masks for each baseline CT. Convert DICOM to NIfTI and review images and masks before analysis. Segmentation/clinical quality assurance is a prerequisite; it is not performed by this package.

## Prepare inputs for the released model

```bash
python -m pip install -e ".[imaging]"
python -m nsclc_toxicity download-weights --imaging
```

Create `private_data/ct_manifest.csv` following [`data/ct_manifest_example.csv`](../data/ct_manifest_example.csv). The example contains placeholder paths and is a template, not a runnable CT dataset.

Required columns: `patient_id`, `scan_id`, `scan_day`, `is_baseline`, the six clinical columns in the data dictionary, `ct_path`, `mask_dir`, and `treated_lobe`. Repeat baseline clinical values on follow-up rows. Paths are relative to the manifest file or absolute. `treated_lobe` is `RUL`, `RML`, `RLL`, `LUL`, or `LLL`; blank leaves the three lobe-contrast features missing. Follow-up `mask_dir` may be blank because Q7 is taken from the baseline scan.

Each baseline mask folder contains:

```text
lung_upper_lobe_right.nii.gz
lung_middle_lobe_right.nii.gz
lung_lower_lobe_right.nii.gz
lung_upper_lobe_left.nii.gz
lung_lower_lobe_left.nii.gz
```

Masks must match the CT's shape and affine exactly (1e-3 tolerance); the extractor rejects mismatches. The NIfTI third array axis must index axial slices. Preserve the acquisition array orientation used by your validated preprocessing; this model does not automatically reorient, crop, segment, or harmonize scans.

```bash
python -m nsclc_toxicity prepare-ct --manifest private_data/ct_manifest.csv --output results/prepared.csv
python -m nsclc_toxicity predict --input results/prepared.csv --output results/ct_predictions.csv
```

Add `--device cuda` to `prepare-ct` if CUDA is available. It writes one row per scan with Q7 and all 12 fold-specific hazard inputs. These hazard columns are produced by networks; do not fill them with toxicity labels or manually estimated patient risks.

## Retrain every branch on your own data

Add `event_days`, `censor_days`, and `progression_event_by_365d` to the manifest. Use days from treatment start, not from the baseline scan. Include an optional patient-level `fold` column (1, 2, or 3) to preserve existing folds.

```bash
python -m nsclc_toxicity train-ct --manifest private_data/ct_manifest.csv --output results/ct_training --device cuda
```

This initializes each fold from the public [timm DINOv3 ViT-S/16 pretrained backbone](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), trains the adapters/head only, saves three checkpoints, prepares the CSV, and runs the final tabular/fusion experiment. Internet access is needed for initialization; `--initial-backbone /path/to/timm_state.pt` accepts a local matching backbone state. It must be the pretrained backbone, not one of this project's supervised checkpoints.

Outputs include `prepared.csv`, `dino_fold1.pt`–`dino_fold3.pt`, training loss tables, and `experiment/` with the fitted tabular model, held-out predictions, metrics, and figure. Use `--weights results/ct_training/experiment/model.json` for inference with your trained model. For new CTs, pass `--weights results/ct_training` to `prepare-ct` first. Training uses all volumes in memory and can take hours; GPU training is recommended.

## Feature definitions

DL preprocessing clips HU to [-1000, 400], scales to [0, 1], transposes XYZ to ZYX, and linearly resizes to 48 × 96 × 96 with no antialiasing. Sixteen uniformly spaced three-slice slabs pass through a learned 1×1 input adapter, bilinear resize to 224 × 224, DINOv3, equal-weight pooling, and the hazard head. No ImageNet normalization is added.

Q7 uses baseline CT values on the native exact mask grid. Whole-lung volume is the five-lobe union. Mean HU and density fractions use original HU; the two fractions are HU < -950 and -700 ≤ HU < -300. Contrasts subtract the union of non-target lobes from the treated lobe; target voxels are removed from the other-lobe mask. Contrasts are P95 HU, fraction HU ≥ -300, and signed Sobel mean. Sobel uses clipped HU, `scipy.ndimage.sobel` with default last in-plane axis/mode on every occupied axial slice, then equal-weight averaging of within-mask slice means.
