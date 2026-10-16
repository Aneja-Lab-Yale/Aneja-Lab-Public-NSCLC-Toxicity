# Methods and reproducibility

## Final model

The endpoint is first cumulative grade ≥2 pulmonary toxicity at 90, 183, 274, and 365 days from treatment start. Development used 90 patients and three patient-level folds after excluding documented local or distant progression within 365 days. These are development estimates, not external validation.

The released inference model averages **three fold models**. Each uses:

| Branch | Selected specification |
| --- | --- |
| DL | Latest CT; DINOv3 ViT-S/16; LoRA rank 8, alpha 16, dropout 0.05; 64-unit MLP; four interval hazards; training-baseline empirical-CDF normalization |
| Q | Seven baseline QCT descriptors; four separate L2 logistic interval-hazard models, C=0.30 |
| Clinical | Six baseline clinical features; one shared L2 logistic hazard model with four interval indicators, C=0.03 |
| Imaging | One-third DL + two-thirds Q |
| Multimodal | Imaging/Clinical proportions 80/20, 60/40, 40/60, 20/80 at successive horizons |

The DL network uses the pretrained encoder with trained LoRA and input adapters. It does not use PCA embeddings as its final prediction input. Quantitative and clinical features are listed in `nsclc_toxicity/schema.py` and the data dictionary. Tumor size is in mm, total dose in Gy, T stage is ordinal 1–4, and location/histology use binary indicators. Histology matches adenocarcinoma and squamous/SCC/SqCC; other histologies have both indicators zero. Use the reconciled total prescribed treatment dose. Missing features remain blank, including unknown location/histology.

Median imputation, missingness indicators, and standardization are fitted inside each training fold. Logistic training weights give equal total weight to each patient, then upweight events by the control/event weight ratio capped at 8. Clinical weights are computed over the stacked observed intervals. Q weights are computed separately for each interval.

For each scan, accumulated risk is `1 − product(1 − hazard)` over bins ending after the scan and at/before the requested horizon. A partially elapsed bin is included in full; there is no interpolation within a bin. A prediction is emitted only for a future horizon. DL normalization uses the direct cumulative-risk distribution of that fold's training baseline scans, with the same midpoint rank and finite-sample offset as development.

The DL normalizer's unlinked, sorted breakpoints are learned transformation parameters stored with the weights, like medians and scaling parameters. No patient IDs, labels, dates, source paths, images, embeddings, or linked prediction tables are distributed with the weights. The learned parameters do not provide a differential-privacy guarantee.

The mixed scores are not calibrated clinical probabilities and need not increase across horizons because branch weights and normalization change with time. Do not interpret `0.7` as a 70% patient risk.

## Own-data experiments

`experiment` retrains Q/Clinical inside each fold and evaluates the fixed final fusion, plus standalone branches. All scans from a patient stay together. An optional `fold` column locks folds 1–3; otherwise stratified patient folds use seed 20260720. Retain the same folds used to produce your fold-specific DL inputs.

The command applies the progression exclusion, removes scans at/after toxicity and scans after observation ends, and masks unobserved horizons. Cases have an event after the scan and by the horizon. Controls have an event after the horizon, or no event with observation through the horizon. Earlier censoring is unknown, not a negative label.

Time-dependent AUC uses inverse scan-count patient weights within each fold/horizon. The summary averages the three folds. Integrated AUC uses a time-weighted trapezoid from day 90 to day 365; it is blank when any horizon lacks all three evaluable folds. `performance.png` compares all scans, baseline, and follow-up. `oof_predictions.csv` uses only a patient's held-out model, while new-patient inference averages the three models.

For the **full training procedure**, `train-ct` fits a new CT model on baseline scans from the other two folds, scores every scan with all three models, and runs the tabular experiment with the same folds. It uses 8 epochs, batch size 2, AdamW learning rate/weight decay 0.0001, natural sampling, masked hazard BCE, train-only smoothed hazard-bias initialization, and seed 20260714. No test outcomes select weights or epochs. CPU training and different devices need not be bitwise identical to development.

Using released CT weights on a genuinely new cohort evaluates a fixed pretrained CT representation. Reusing those weights on their development patients is not an independent CT validation.

## Validation and scope

The public implementation reproduced all 654 held-out prediction rows per branch and fusion model in the locked development artifacts to within 1e-15. Only aggregate numerical differences are recorded in [`development_parity.json`](development_parity.json); no original predictions are included. The three CT checkpoints load strictly with the portable architecture. Version 1.1 adds a real public CT end-to-end example, automatic lung-lobe segmentation, and explicit LAS axis alignment matching all 231 development scans. Existing LAS inputs and fitted model parameters are unchanged; earlier raw input files in other orientations now receive the required flips/permutations. See [public CT validation](PUBLIC_CT.md).

This repository reproduces the selected final model, branch comparisons, baseline/follow-up evaluation, feature extraction, and retraining on compatible user data. Historical model-search scripts and manuscript-specific cohort panels are not part of the portable workflow. Exact original study figures require the private cohort, outcomes, scan selection, and original fold assignments. The synthetic sample cannot reproduce study performance.
