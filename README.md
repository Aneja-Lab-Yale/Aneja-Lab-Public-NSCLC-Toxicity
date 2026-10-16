# NSCLC_Toxicity

Research code for grade ≥2 pulmonary toxicity at **3, 6, 9, and 12 months after lung SBRT**. The final model combines CT imaging, seven measured quantitative CT features, and six clinical features.

**Research use only.** Scores are not calibrated patient risk percentages. Independent external validation is required before clinical use.

## Quickstart: a real CT

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). In Terminal or Anaconda Prompt, run:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-NSCLC-Toxicity.git NSCLC_Toxicity
cd NSCLC_Toxicity
conda env create -f environment.yml
conda activate nsclc_toxicity
python -m nsclc_toxicity download-demo-ct
python -m nsclc_toxicity infer-ct --manifest data/public_ct/clinical.csv --output results/public_ct
```

Already cloned? Run `git pull` and `conda env update -f environment.yml`, then activate `nsclc_toxicity` again.

This downloads the **real public [3D Slicer CTChest scan](https://www.slicer.org/wiki/SampleData)** and the free model weights. It then converts the image, generates five lung-lobe masks, measures QCT features, runs three DINOv3 CT networks, and produces final ensemble scores. No synthetic imaging features enter this workflow. It runs locally on a CPU; the first run needs internet for downloads and can take several minutes. Add `--device cuda` for a CUDA GPU.

Open `results/public_ct/predictions.csv` in Excel. **`Multimodal` is the final model**; other rows show its component models. `prepared.csv` contains the measured QCT features and actual CT network outputs. Masks are saved in `results/public_ct/lobe_masks/`.

**The CT is real; the example clinical values are fictional.** Its actual tumor, treatment, and toxicity outcomes are not supplied by the source. This example demonstrates the complete software workflow, not that person's risk or the model's clinical accuracy. The [source and checksum](docs/PUBLIC_CT.md) are recorded.

## Notebook

```bash
jupyter lab notebooks/quickstart.ipynb
```

Select the `nsclc_toxicity` Python environment and choose **Run → Run All Cells**. The notebook runs the image workflow, displays the measured features and CT network outputs, and overlays the computed lobe masks on the scan.

## Your own CT and clinical data

Copy [`data/clinical_manifest_example.csv`](data/clinical_manifest_example.csv) into `private_data/clinical.csv`. Edit the image path and these clinical inputs:

| Column | What to enter |
| --- | --- |
| `ct_path` | HU NIfTI/NRRD file, or folder containing one CT DICOM series |
| `tumor_size_mm` | Greatest baseline tumor diameter in mm |
| `t_stage` | Clinical T stage, e.g. `T1c` or `1` |
| `total_dose_gy` | Total prescribed SBRT dose in Gy |
| `treated_lobe` | `RUL`, `RML`, `RLL`, `LUL`, or `LLL` |
| `histology` | e.g. `adenocarcinoma`, `squamous`, `SCC`, or `other` |

Keep the study/scan IDs, `is_baseline`, and `scan_day` columns. `scan_day` is days from treatment start; baseline is ≤0. Paths are relative to the CSV. Leave unknown clinical values blank. Tumor location and treatment information must come from clinical records.

```bash
python -m nsclc_toxicity infer-ct --manifest private_data/clinical.csv --output results/my_case
```

No precomputed imaging features or masks are required. Use an event-free CT with complete lung coverage; inspect the masks before interpreting outputs. The development study excluded lung-cancer progression within one year. See the [CT guide](docs/CT_WORKFLOW.md) for follow-up scans, segmentation resolution, DICOM details, and supplied masks.

## Training and experiments

The [CT guide](docs/CT_WORKFLOW.md) explains retraining every branch on your own images and outcomes. For a small tabular experiment with simulated imaging inputs:

```bash
python -m nsclc_toxicity experiment --input data/sample.csv --output results/sample_experiment
```

`data/sample.csv` and `notebooks/csv_demo.ipynb` remain **fully synthetic tabular demos**. See [methods](docs/METHODS.md), the [column dictionary](data/data_dictionary.csv), and [setup help](docs/SETUP.md). Model downloads are free, versioned [GitHub Release assets](https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-NSCLC-Toxicity/releases/tag/v1.0.0) with SHA256 verification; no account is needed.

Code and synthetic data: [MIT](LICENSE). DINOv3 weights: [DINOv3 license](docs/DINOV3_LICENSE.md). Public CT terms and third-party models retain their upstream licenses; see [acknowledgments](docs/THIRD_PARTY.md).
