# Example data

`sample.csv` contains 72 entirely fictional patients and 198 scans. Every value was generated from a fixed random seed and hand-chosen example ranges. No source records, images, fitted weights, patient-derived distributions, or real identifiers were used to create it. The `dl_fold*_hazard_*d` values are simulated network outputs, not outputs from patient CTs.

The same CSV works for inference and the experiment demo. For inference, `event_days`, `censor_days`, `progression_event_by_365d`, and `fold` are optional. Training/evaluation requires outcomes; folds are optional. The [dictionary](data_dictionary.csv) describes all columns.

Regenerate the example from the repository root:

```bash
python -m nsclc_toxicity.synthetic
```

`ct_manifest_example.csv` is a template with invented IDs and nonexistent image paths. Real CTs and patient data belong in the ignored `private_data/` folder. Generated predictions and fitted models belong in the ignored `results/` folder.
