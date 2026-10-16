import io
import json
from pathlib import Path

import pandas as pd
import pytest

from nsclc_toxicity import download
from nsclc_toxicity.synthetic import make_sample


def test_sample_is_exactly_the_independently_generated_example():
    sample = pd.read_csv("data/sample.csv")
    pd.testing.assert_frame_equal(sample, make_sample(), check_dtype=False, rtol=1e-9, atol=1e-8)
    assert sample["patient_id"].str.fullmatch(r"SYN\d{3}").all()
    assert sample["scan_id"].str.fullmatch(r"SYN\d{3}_V\d").all()
    assert not any("path" in col or "date" in col or "name" in col for col in sample)


def test_notebook_contains_no_saved_results():
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_primary_notebook_computes_real_images():
    notebook = json.loads(Path("notebooks/quickstart.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "download_demo(" in source
    assert "prepare_ct(" in source and "auto_masks=True" in source
    assert "data/sample.csv" not in source
    assert "make_sample" not in source


def test_download_rejects_corruption_and_preserves_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "final_model.json"
    target.write_text("old file")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt data"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download.download_weights(tmp_path)
    assert target.read_text() == "old file"
    assert not list(tmp_path.glob("*.part"))
