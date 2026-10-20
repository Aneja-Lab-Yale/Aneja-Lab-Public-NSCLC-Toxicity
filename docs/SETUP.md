# Setup help

Run commands from the repository folder, with `nsclc_toxicity` activated.

**Without Conda:** for the real-CT notebook, use Python 3.10–3.12 (3.11 recommended):

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\activate` in Windows Command Prompt. Then:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,imaging,segmentation]"
python -m ipykernel install --sys-prefix --name nsclc_toxicity --display-name "NSCLC Toxicity (CT)"
python -m jupyter lab notebooks/quickstart.ipynb
```

**Notebook kernel:** select **NSCLC Toxicity (CT)**. In VS Code, select **Python Environments → nsclc_toxicity** (or your `.venv`). An open notebook keeps using its previous Python until you switch/restart the kernel. Installing packages in another terminal environment does not add them to the running kernel.

**`No module named 'torch'`:** the selected kernel is missing the CT installation. For the Conda environment, run these commands from the repository folder:

```bash
conda env update -f environment.yml
conda activate nsclc_toxicity
python -m ipykernel install --sys-prefix --name nsclc_toxicity --display-name "NSCLC Toxicity (CT)"
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Then switch to **NSCLC Toxicity (CT)** and restart the notebook kernel. The first cell prints the selected Python and reports missing dependencies with repair instructions.

**CSV-only installation:** `python -m pip install -e ".[notebook]"` is sufficient for `notebooks/csv_demo.ipynb` and the synthetic tabular sample. It does not install PyTorch or raw CT processing. The CSV-only command-line workflow also supports Python 3.9.

**Weights not found:** run `download-weights` from the repository folder. Or provide a custom file with `predict --weights /path/to/final_model.json`.

**Download blocked by your institution:** download assets from the [release page](https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-NSCLC-Toxicity/releases/tag/v1.0.0), place them in `weights/`, and rerun `download-weights` to verify them. Core inference needs only `final_model.json`; CT processing also needs the three `.pt` files.

**CSV error:** compare headers with `data/data_dictionary.csv`. Missing clinical/QCT values are blank cells. CT hazard inputs must be complete probabilities from 0 to 1. Do not use `unknown`, `%`, or units inside numeric cells.

**Raw CT dependencies:** the main `environment.yml` includes them and uses Python 3.11. For an existing Python ≥3.10 environment, install `python -m pip install -e ".[imaging,segmentation,notebook]"`. CPU works; CT retraining is much faster on a CUDA GPU. The prepared synthetic CSV demo needs no PyTorch or segmentation installation.

**CPU-only PyTorch on Linux/Windows:** before installing this package in a Python environment, run:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[imaging,segmentation,notebook]"
```

**Old notebook kernel:** if VS Code selects `nsclc_env` from the private project, switch to the new `nsclc_toxicity` environment. The new environment includes `ipykernel`; select it before running the notebook.

**Verify the installation:**

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
