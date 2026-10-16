# NSCLC_Toxicity

Research code for predicting grade ≥2 pulmonary toxicity at **3, 6, 9, and 12 months after lung SBRT**. The final model combines CT imaging, seven quantitative CT features, and six clinical features.

**Research use only.** This development model requires independent external validation before clinical use. Its scores are not calibrated patient risk percentages.

## Quickstart

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), then open Terminal (macOS/Linux) or Anaconda Prompt (Windows). Copy these commands in order:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-NSCLC-Toxicity.git NSCLC_Toxicity
cd NSCLC_Toxicity
conda env create -f environment.yml
conda activate nsclc_toxicity
python -m nsclc_toxicity download-weights
python -m nsclc_toxicity predict --input data/sample.csv --output results/predictions.csv
```

The downloader puts verified model weights in `weights/`. Downloads use a [versioned public release](https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-NSCLC-Toxicity/releases/tag/v1.0.0); no account is needed. The CSV quickstart runs on a CPU.

Open `results/predictions.csv` in Excel. Each row contains a scan, a future time point, a model name, and a score from 0 to 1. **`Multimodal` is the final model**, averaged across three development models. The other rows show its component models.

`data/sample.csv` is **entirely synthetic**: invented records, outcomes, and simulated CT model outputs. It demonstrates the workflow and contains no patient records. It already includes prepared CT features; the quickstart does not process images.

## Notebook

```bash
jupyter lab notebooks/quickstart.ipynb
```

Choose **Run → Run All Cells**. The notebook downloads weights, runs the sample, and displays the final scores.

## Use your own data

Use the same columns as [`data/sample.csv`](data/sample.csv). The [data dictionary](data/data_dictionary.csv) explains every column. Use study IDs and days from treatment instead of names, record numbers, or calendar dates.

- **Prepared CSV:** put it in `private_data/my_data.csv`, then run:

  ```bash
  python -m nsclc_toxicity predict --input private_data/my_data.csv --output results/my_predictions.csv
  ```

- **CT images:** follow the [CT preparation and training guide](docs/CT_WORKFLOW.md). This calculates the imaging inputs from NIfTI CTs and lung-lobe masks.

Baseline clinical and QCT values repeat on each follow-up row. Only the latest CT model outputs change. Outcomes are unnecessary for inference. Include only event-free scans; the study excludes lung-cancer progression within one year of SBRT.

## Repeat the experiments

```bash
python -m nsclc_toxicity experiment --input data/sample.csv --output results/sample_experiment
```

Replace the input with your own prepared CSV to fit and evaluate the model on your dataset. This writes held-out predictions, AUC tables, a comparison figure, and fitted weights. The sample uses simulated CT outputs. To retrain the CT networks too, use `train-ct` in the [CT guide](docs/CT_WORKFLOW.md).

See [methods and reproducibility](docs/METHODS.md) for the exact model, fold rules, and scope of the experiments; [setup help](docs/SETUP.md) for installation alternatives.

## Files

| Folder | Contents |
| --- | --- |
| `data/` | Synthetic sample, column dictionary, CT manifest example |
| `notebooks/` | Step-by-step inference notebook |
| `nsclc_toxicity/` | Inference, training, evaluation, and CT processing |
| `docs/` | Methods, CT workflow, validation, and setup |

Code and synthetic data: [MIT](LICENSE). DINOv3 checkpoints: [DINOv3 license](docs/DINOV3_LICENSE.md). See [third-party acknowledgments](docs/THIRD_PARTY.md).
