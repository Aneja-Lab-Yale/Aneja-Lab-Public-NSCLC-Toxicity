import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nsclc_toxicity.model import accumulate, load_model, logistic_predict, percentile, predict
from nsclc_toxicity.schema import eligible_cohort, targets, validate
from nsclc_toxicity.synthetic import make_sample
from nsclc_toxicity.training import assign_folds, export_logistic, run_experiment, train_ensemble


def test_treatment_origin_boundaries_and_censoring():
    frame = pd.DataFrame({"scan_day": [-20, -20, -20, 90, 100, 183],
                          "event_days": [90, 183, np.nan, 140, 140, np.nan],
                          "censor_days": [400, 400, 89, 400, 400, 400]})
    cumulative, hazard = targets(frame)
    np.testing.assert_equal(cumulative[0], [1, 1, 1, 1])
    np.testing.assert_equal(hazard[0], [1, np.nan, np.nan, np.nan])
    np.testing.assert_equal(hazard[1], [0, 1, np.nan, np.nan])
    assert np.isnan(cumulative[2]).all()
    np.testing.assert_equal(hazard[3], [np.nan, 1, np.nan, np.nan])
    np.testing.assert_equal(cumulative[5], [np.nan, np.nan, 0, 0])
    np.testing.assert_allclose(accumulate(np.full((1, 4), 0.2), np.array([90]))[0], [0, 0.2, 0.36, 0.488])


def test_export_preserves_missing_features_and_indicators():
    matrix = np.array([[0, np.nan, np.nan], [2, 2, np.nan], [np.nan, 4, np.nan], [5, 5, np.nan], [1, 2, np.nan], [4, 3, np.nan]])
    target = np.array([0, 0, 1, 1, 0, 1])
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()), ("model", LogisticRegression(solver="liblinear"))])
    with pytest.warns(UserWarning, match="Skipping features"):
        pipeline.fit(matrix, target)
    exported = json.loads(json.dumps(export_logistic(pipeline, 3)))
    with pytest.warns(UserWarning, match="Skipping features"):
        expected = pipeline.predict_proba(matrix)[:, 1]
    np.testing.assert_allclose(logistic_predict(exported, matrix), expected, rtol=1e-14)


def test_normalizer_ties_and_extremes():
    np.testing.assert_allclose(percentile([0.2, 0.2, 0.5], np.array([0.0, 0.2, 0.3, 0.9])), [0.125, 0.375, 0.625, 0.875])


def test_patient_folds_and_exclusions():
    sample = make_sample()
    new = assign_folds(sample.drop(columns="fold"))
    assert new.groupby("patient_id")["fold"].nunique().eq(1).all()
    assert set(new["fold"]) == {1, 2, 3}
    patient = sample["patient_id"].iloc[0]
    sample.loc[sample["patient_id"].eq(patient), "progression_event_by_365d"] = 1
    assert patient not in set(eligible_cohort(sample)["patient_id"])
    bad = make_sample()
    bad.loc[1, "fold"] = 3
    with pytest.raises(ValueError, match="stay in one fold"):
        assign_folds(bad)


def test_held_out_values_do_not_change_their_fold_model():
    sample = make_sample()
    before = train_ensemble(sample)["folds"][0]
    sample.loc[sample["fold"].eq(1), "clinical_tumor_size_mm"] += 1000
    after = train_ensemble(sample)["folds"][0]
    assert before == after


def test_validation_rejects_nonbaseline_features_and_invalid_hazards():
    sample = make_sample()
    sample.loc[1, "lung_volume_ml"] += 1
    with pytest.raises(ValueError, match="repeat the baseline"):
        validate(sample)
    sample = make_sample()
    sample.loc[0, "dl_fold1_hazard_90d"] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        validate(sample)


def test_full_experiment_and_inference(tmp_path):
    sample = make_sample()
    report = run_experiment(sample, tmp_path)
    assert report["patients"] == 72
    model = load_model(tmp_path / "model.json")
    scores = predict(sample, model)
    assert scores["score"].between(0, 1).all()
    assert scores["scan_day"].lt(scores["horizon_days"]).all()
    assert set(scores["model"]) == {"DL", "Q", "Clinical", "Imaging", "Multimodal"}
    oof = pd.read_csv(tmp_path / "oof_predictions.csv")
    assert not oof.duplicated(["patient_id", "scan_id", "horizon_days", "model"]).any()
    assert (tmp_path / "performance.png").stat().st_size > 1000


def test_local_release_weights_if_present():
    path = Path("weights/final_model.json")
    if not path.is_file():
        pytest.skip("Release assets are downloaded separately")
    scores = predict(pd.read_csv("data/sample.csv"), load_model(path))
    assert len(scores) == 3030
    assert scores["score"].between(0, 1).all()
