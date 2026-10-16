"""A real, checksum-pinned chest CT from 3D Slicer's public Sample Data.

Images are downloaded from the original publisher, not redistributed in Git.
Clinical inputs below are fictional demonstration values, not scan metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request

from .download import sha256

CT_SHA256 = "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
CT_URL = f"https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/{CT_SHA256}"
CT_SOURCE = "https://github.com/Slicer/Slicer/blob/9ae94df99124986e65094520d5631baef32314e3/Modules/Scripted/SampleData/SampleData.py"


def download_demo(destination: Path) -> Path:
    import pandas as pd
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "CT-chest.nrrd"
    if not target.is_file() or sha256(target) != CT_SHA256:
        print("Downloading the real public 3D Slicer CTChest scan (42.2 MB)...", flush=True)
        temporary = target.with_suffix(".part")
        try:
            request = urllib.request.Request(CT_URL, headers={"User-Agent": "NSCLC-Toxicity/1.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
            if sha256(temporary) != CT_SHA256:
                raise ValueError("Public CT checksum mismatch; rerun the download.")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    # Never replace a clinical manifest a user may have edited.
    manifest = destination / "clinical.csv"
    if not manifest.is_file():
        pd.DataFrame([{"patient_id": "PUBLIC_CT_DEMO", "scan_id": "PUBLIC_CT_BASELINE",
                       "scan_day": -30, "is_baseline": 1, "ct_path": "CT-chest.nrrd",
                       "tumor_size_mm": 24, "t_stage": "T1c", "total_dose_gy": 50,
                       "treated_lobe": "RLL", "histology": "adenocarcinoma"}]).to_csv(manifest, index=False)
    (destination / "source.json").write_text(json.dumps({
        "name": "3D Slicer CTChest", "source": CT_SOURCE, "download_url": CT_URL,
        "sha256": CT_SHA256, "upstream_terms": "https://github.com/Slicer/Slicer/blob/main/License.txt",
        "clinical_inputs": "Fictional example only. The CT's real tumor, treatment and outcomes are not supplied.",
        "purpose": "End-to-end software demonstration; not validation of toxicity prediction."}, indent=2) + "\n")
    print(f"Verified public CT: {target}\nClinical manifest (fictional example): {manifest}")
    return manifest
