"""CT preprocessing, exact-grid Q7 features, and the selected DINOv3 network.

The architecture and tensor names match the development checkpoints. Optional
imaging dependencies are kept out of the lightweight CSV quickstart.
"""

from __future__ import annotations

import json
import math
import random
import hashlib
import os
import re
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import sobel, zoom
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .schema import C_FEATURES, FOLDS, HORIZONS, IDENTITY, Q_FEATURES, dl_columns, eligible_cohort, require_columns, validate

LOBES = {
    "RUL": "lung_upper_lobe_right", "RML": "lung_middle_lobe_right",
    "RLL": "lung_lower_lobe_right", "LUL": "lung_upper_lobe_left", "LLL": "lung_lower_lobe_left",
}


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear):
        super().__init__()
        self.base_layer = base_layer
        self.dropout = nn.Dropout(0.05)
        self.lora_A = nn.Parameter(torch.empty(8, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, 8))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * 2.0


def select_slabs(x):
    b, c, depth, height, width = x.shape
    centers = torch.linspace(0, depth - 1, steps=min(16, depth), device=x.device).round().long()
    indices = (centers[:, None] + torch.arange(-1, 2, device=x.device)[None, :]).clamp(0, depth - 1)
    gathered = [x[:, :, indices[:, i]] for i in range(3)]
    slabs = torch.stack(gathered, dim=3)
    return slabs.permute(0, 2, 1, 3, 4, 5).reshape(b * len(centers), c * 3, height, width)


class SlabEncoder(nn.Module):
    def __init__(self, pretrained: bool = False, initial_backbone: Path | None = None):
        super().__init__()
        self.input_adapter = nn.Conv2d(3, 3, kernel_size=1)
        self.backbone = timm.create_model("vit_small_patch16_dinov3", pretrained=pretrained and initial_backbone is None,
                                          in_chans=3, num_classes=0, global_pool="token", img_size=224)
        if initial_backbone is not None:
            self.backbone.load_state_dict(torch.load(initial_backbone, map_location="cpu", weights_only=True), strict=True)
        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, nn.Linear) and any(token in name for token in ["qkv", "proj", "fc", "mlp"]):
                parent = self.backbone
                pieces = name.split(".")
                for part in pieces[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, pieces[-1], LoRALinear(module))
        for name, parameter in self.named_parameters():
            parameter.requires_grad = "lora_" in name or "input_adapter" in name

    def forward(self, x):
        slabs = self.input_adapter(select_slabs(x))
        slabs = F.interpolate(slabs, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.backbone(slabs).reshape(x.shape[0], -1, 384)
        # Match the original explicit equal-weight sum, including operation order.
        weights = features.new_full(features.shape[:2], 1.0 / features.shape[1])
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class HazardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(384, 64), nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


class DinoModel(nn.Module):
    def __init__(self, pretrained: bool = False, initial_backbone: Path | None = None):
        super().__init__()
        self.feature_extractor = SlabEncoder(pretrained, initial_backbone)
        self.head = HazardHead()

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.backbone.eval()
        if mode:
            for module in self.feature_extractor.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train()
        return self

    def forward(self, x):
        return self.head(self.feature_extractor(x))


def load_ct(path: Path) -> nib.Nifti1Image:
    """Read HU and match the development cohort's LAS array orientation.

    This flips/permutates voxel axes without interpolation or intensity changes.
    The neural network and signed axial Sobel feature depend on axis direction.
    """
    image = nib.load(str(path))
    if len(image.shape) != 3 or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("CT must have a valid 3D voxel grid and spatial affine.")
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                               nib.orientations.axcodes2ornt(("L", "A", "S")))
    return image.as_reoriented(transform)


def preprocess_ct(path: Path) -> np.ndarray:
    image = load_ct(path)
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("CT must be a finite 3D NIfTI volume in Hounsfield units.")
    array = (np.clip(array, -1000, 400) + 1000) / 1400
    array = array.transpose(2, 1, 0)
    return zoom(array, [s / n for s, n in zip((48, 96, 96), array.shape)], order=1).astype(np.float32)


def qct_features(path: Path, mask_dir: Path, treated_lobe: str) -> dict:
    image = load_ct(path)
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("QCT requires a finite 3D CT volume.")
    masks = {}
    for lobe, filename in LOBES.items():
        mask = nib.load(str(mask_dir / f"{filename}.nii.gz"))
        mask = mask.as_reoriented(nib.orientations.ornt_transform(
            nib.orientations.io_orientation(mask.affine), nib.orientations.axcodes2ornt(("L", "A", "S"))))
        if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-3, rtol=0):
            raise ValueError("All five lobe masks must match the exact CT voxel grid and affine.")
        mask_array = mask.get_fdata(dtype=np.float32)
        if not np.isfinite(mask_array).all():
            raise ValueError("Lobe masks must contain finite values.")
        masks[lobe] = mask_array > 0
        if np.count_nonzero(masks[lobe]) < 10:
            raise ValueError("Each lobe mask must contain at least 10 voxels.")
    lung = np.logical_or.reduce(list(masks.values()))
    values = array[lung]
    result = dict(zip(Q_FEATURES, [
        float(values.size * np.prod(image.header.get_zooms()[:3]) / 1000.0),
        float(np.mean(values)), float(np.mean(values < -950)),
        float(np.mean((values >= -700) & (values < -300))), np.nan, np.nan, np.nan,
    ]))
    if treated_lobe not in masks:
        return result
    target = masks[treated_lobe]
    other = np.logical_or.reduce([v for k, v in masks.items() if k != treated_lobe]) & ~target
    if np.count_nonzero(other) < 10:
        raise ValueError("Non-target lobes have fewer than 10 voxels after removing overlap.")
    result[Q_FEATURES[4]] = float(np.percentile(array[target], 95) - np.percentile(array[other], 95))
    result[Q_FEATURES[5]] = float(np.mean(array[target] >= -300) - np.mean(array[other] >= -300))
    clipped = np.clip(array, -1000, 400).astype(np.float32, copy=False)
    means = [[], []]
    for z in np.flatnonzero((target | other).any(axis=(0, 1))):
        gradient = sobel(clipped[:, :, z])
        for destination, mask in zip(means, (target, other)):
            if mask[:, :, z].any():
                destination.append(float(np.mean(gradient[mask[:, :, z]])))
    result[Q_FEATURES[6]] = float(np.mean(means[0]) - np.mean(means[1]))
    return result


def convert_ct(source: Path, destination: Path) -> Path:
    """Accept HU NIfTI/NRRD/MHA or a directory containing one DICOM CT series."""
    if source.is_file() and source.name.lower().endswith((".nii", ".nii.gz")):
        return source
    if not source.exists():
        raise ValueError(f"CT input does not exist: {source}")
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ValueError('NRRD/DICOM conversion needs: python -m pip install -e ".[segmentation]"') from exc
    if source.is_dir():
        series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source))
        if not series or len(series) != 1:
            raise ValueError("Place exactly one CT DICOM series in each input directory; separate localizers and other series first.")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(source), series[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()
        if reader.HasMetaDataKey(0, "0008|0060") and reader.GetMetaData(0, "0008|0060").strip() != "CT":
            raise ValueError("The input DICOM series must have CT modality.")
    else:
        image = sitk.ReadImage(str(source))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("CT input must be a scalar 3D volume, in Hounsfield units.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # SimpleITK applies DICOM rescale slope/intercept when reading; no windowing here.
    # Publish the converted file only when complete. A notebook and a command
    # may read the same demo folder at once; neither should see truncated gzip.
    with tempfile.TemporaryDirectory(prefix=".conversion-", dir=destination.parent) as temporary:
        pending = Path(temporary) / destination.name
        sitk.WriteImage(image, str(pending))
        os.replace(pending, destination)
    return destination


def clinical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate a short physician-facing manifest into the six model inputs."""
    frame = frame.copy()
    lobe = frame["treated_lobe"].fillna("").astype(str).str.strip().str.upper()
    if not lobe.isin(["", *LOBES]).all():
        raise ValueError("treated_lobe must be RUL, RML, RLL, LUL, LLL, or blank.")
    if not set(C_FEATURES).issubset(frame):
        require_columns(frame, ["tumor_size_mm", "t_stage", "total_dose_gy", "histology"])
        frame[C_FEATURES[0]] = pd.to_numeric(frame["tumor_size_mm"], errors="raise")
        frame[C_FEATURES[2]] = pd.to_numeric(frame["total_dose_gy"], errors="raise")
        def stage(value):
            if pd.isna(value) or not str(value).strip():
                return np.nan
            text = str(value).strip().upper()
            if re.fullmatch(r"[0-4]\.0", text):
                return float(text)
            match = re.fullmatch(r"[CPRY]*T?([0-4])[ABC]?", text)
            if not match:
                raise ValueError("t_stage must be 0–4 or a T stage such as T1a; leave unknown values blank.")
            return float(match.group(1))
        frame[C_FEATURES[1]] = frame["t_stage"].map(stage)
        frame[C_FEATURES[3]] = np.where(lobe.eq(""), np.nan, lobe.isin(["RLL", "LLL"]).astype(float))
        histology = frame["histology"].fillna("").astype(str).str.lower().str.strip()
        unknown = histology.isin(["", "unknown", "not known"])
        frame[C_FEATURES[4]] = np.where(unknown, np.nan, histology.str.contains("adeno", regex=False).astype(float))
        frame[C_FEATURES[5]] = np.where(unknown, np.nan, histology.str.contains(r"squamous|\bscc\b|sqcc", regex=True).astype(float))
    known = lobe.ne("") & frame[C_FEATURES[3]].notna()
    if frame.loc[known, C_FEATURES[3]].astype(float).ne(lobe.loc[known].isin(["RLL", "LLL"]).astype(float)).any():
        raise ValueError("The lower-lobe clinical indicator disagrees with treated_lobe.")
    frame["treated_lobe"] = lobe
    return frame


def segment_lobes(ct_path: Path, output: Path, *, device: str = "cpu", resolution: float = 6.0,
                  cache: Path | None = None) -> Path:
    """Generate five real lobe masks, reuse only a matching completed run."""
    from .download import sha256
    output.mkdir(parents=True, exist_ok=True)
    cache = cache or Path("weights/totalsegmentator")
    cache.mkdir(parents=True, exist_ok=True)
    # Keep weights/config inside the project rather than writing to a user's home.
    os.environ["TOTALSEG_HOME_DIR"] = str(cache.resolve())
    if resolution not in (6, 3, 1.5):
        raise ValueError("Segmentation resolution must be 6, 3, or 1.5 mm.")
    expected = {"ct_sha256": sha256(ct_path), "task": "total", "resolution_mm": resolution,
                "rois": list(LOBES.values()), "TotalSegmentator": "2.13.0"}
    record = output / "segmentation.json"
    if record.is_file() and json.loads(record.read_text()) == expected and all((output / f"{lobe}.nii.gz").is_file() for lobe in LOBES.values()):
        return output
    try:
        from importlib.metadata import version
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ValueError('Automatic masks need Python >=3.10 and: python -m pip install -e ".[imaging,segmentation]"') from exc
    if version("TotalSegmentator") != expected["TotalSegmentator"]:
        raise ValueError('Install the tested segmentation version: python -m pip install -e ".[segmentation]"')
    # The total task is open and needs no paid license; do not send usage telemetry.
    config = cache / "config.json"
    if config.is_file():
        settings = json.loads(config.read_text())
    else:
        settings = {"totalseg_id": "nsclc-local", "prediction_counter": 0}
    settings["send_usage_stats"] = False
    config.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Segmenting lung lobes ({resolution:g} mm, {device})...", flush=True)
    totalsegmentator(ct_path, output, task="total", fast=resolution == 3, fastest=resolution == 6, roi_subset=list(LOBES.values()),
                    device="cpu" if device == "cpu" else "gpu", nr_thr_resamp=1, nr_thr_saving=1,
                    quiet=False, statistics=False)
    # Geometry, finite values, and all five nonempty masks are checked before use.
    qct_features(ct_path, output, "RUL")
    record.write_text(json.dumps(expected, indent=2) + "\n")
    return output


def read_manifest(path: Path, *, auto_masks: bool = False, mask_output: Path | None = None,
                  device: str = "cpu", resolution: float = 6.0, cache: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, [*IDENTITY, "ct_path", "treated_lobe"])
    frame = clinical_features(frame)
    frame = validate(frame.assign(**{col: np.nan for col in Q_FEATURES}), dl=False)
    if "mask_dir" not in frame:
        frame["mask_dir"] = ""
    frame["mask_dir"] = frame["mask_dir"].fillna("")
    for col in ("ct_path", "mask_dir"):
        frame[col] = frame[col].map(lambda v: str((path.parent / str(v)).resolve()) if str(v) and not Path(str(v)).is_absolute() else str(v))
    converted = {}
    for source in frame["ct_path"].unique():
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        converted[source] = str(convert_ct(Path(source), (mask_output or Path("results/lobe_masks")).parent / "converted_ct" / f"{key}.nii.gz"))
    frame["ct_path"] = frame["ct_path"].map(converted)
    feature_rows = []
    for _, row in frame.loc[frame["is_baseline"].eq(1)].iterrows():
        mask_dir = Path(row["mask_dir"]) if row["mask_dir"] else None
        if auto_masks:
            # Hash-based folder names avoid putting patient identifiers into paths.
            key = hashlib.sha256(str(row["ct_path"]).encode()).hexdigest()[:16]
            mask_dir = segment_lobes(Path(row["ct_path"]), (mask_output or Path("results/lobe_masks")) / key,
                                     device=device, resolution=resolution, cache=cache)
        if mask_dir is None:
            raise ValueError("Supply baseline mask_dir or add --auto-masks to generate lung lobes from the CT.")
        lobe = str(row["treated_lobe"]).strip().upper() if pd.notna(row["treated_lobe"]) else ""
        if lobe and lobe not in LOBES:
            raise ValueError("treated_lobe must be RUL, RML, RLL, LUL, LLL, or blank; do not infer tumor location from the lobe masks.")
        feature_rows.append({"patient_id": row["patient_id"], **qct_features(Path(row["ct_path"]), mask_dir, lobe)})
    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise ValueError("The CT manifest needs one baseline per patient.")
    frame = frame.drop(columns=list(Q_FEATURES), errors="ignore").merge(features, on="patient_id", validate="many_to_one", how="left")
    return validate(frame, outcomes=False, dl=False)


def infer_hazards(model: DinoModel, volumes: torch.Tensor, device: str) -> np.ndarray:
    model.to(device).eval()
    parts = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(volumes), batch_size=2, shuffle=False):
            parts.append(torch.sigmoid(model(batch.to(device)).float()).cpu().numpy())
    return np.concatenate(parts)


def volume_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(np.stack([preprocess_ct(Path(path)) for path in frame["ct_path"]]))[:, None]


def prepare_ct(manifest: Path, weights: Path, output: Path, device: str = "cpu", *, auto_masks: bool = False,
               resolution: float = 6.0) -> None:
    frame = read_manifest(manifest, auto_masks=auto_masks, mask_output=output.parent / "lobe_masks",
                          device=device, resolution=resolution, cache=weights / "totalsegmentator")
    volumes = volume_tensor(frame)
    for fold in FOLDS:
        checkpoint = weights / f"dino_fold{fold}.pt"
        if not checkpoint.is_file():
            raise ValueError("CT weights are missing. Run: python -m nsclc_toxicity download-weights --imaging")
        print(f"Scoring CTs with model {fold}/3", flush=True)
        model = DinoModel()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        frame[dl_columns(fold)] = infer_hazards(model, volumes, device)
    # Retain only the public input format, never file paths or extra source metadata.
    columns = [*IDENTITY, *Q_FEATURES, *C_FEATURES, *[c for fold in FOLDS for c in dl_columns(fold)]]
    columns += [c for c in ("event_days", "censor_days", "progression_event_by_365d", "fold") if c in frame]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(output, index=False)
    print(f"Prepared {len(frame)} scans: {output}")


def infer_ct(manifest: Path, weights: Path, output: Path, device: str = "cpu", *, resolution: float = 6.0) -> pd.DataFrame:
    """Run raw CT -> lobes -> Q7/DINOv3 -> final three-fold ensemble."""
    from .download import download_weights
    from .model import load_model, predict
    download_weights(weights, imaging=True)
    output.mkdir(parents=True, exist_ok=True)
    prepare_ct(manifest, weights, output / "prepared.csv", device, auto_masks=True, resolution=resolution)
    result = predict(pd.read_csv(output / "prepared.csv"), load_model(weights / "final_model.json"))
    result.to_csv(output / "predictions.csv", index=False)
    # This is run provenance, not a clinical validation claim.
    from . import __version__
    (output / "run.json").write_text(json.dumps({
        "package_version": __version__, "input_kind": "raw CT and clinical manifest",
        "orientation": "LAS (matches development)", "segmentation": {"package": "TotalSegmentator 2.13.0", "task": "total", "resolution_mm": resolution},
        "computed": {"qct_features": 7, "ct_networks": 3, "hazards_per_network": 4},
        "prepared_input": "prepared.csv", "predictions": "predictions.csv",
        "rows": len(result), "score_kind": "uncalibrated research score"}, indent=2) + "\n")
    print(f"Final ensemble predictions: {output / 'predictions.csv'}")
    return result


def train_ct(manifest: Path, output: Path, device: str = "cpu", seed: int = 20260714, initial_backbone: Path | None = None,
             *, auto_masks: bool = False, resolution: float = 6.0) -> None:
    from .training import assign_folds, run_experiment
    frame = assign_folds(eligible_cohort(validate(read_manifest(manifest, auto_masks=auto_masks,
        mask_output=output / "lobe_masks", device=device, resolution=resolution, cache=output / "totalsegmentator"), outcomes=True, dl=False)))
    volumes = volume_tensor(frame)
    output.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = DinoModel(pretrained=True, initial_backbone=initial_backbone).to(device)
        train = frame["fold"].ne(fold) & frame["is_baseline"].eq(1)
        train_frame = frame.loc[train].copy()
        # Deep learning uses treatment-origin baseline labels. The tabular
        # branches additionally train on eligible longitudinal landmarks.
        from .schema import targets
        train_frame["scan_day"] = 0.0
        _, hazard = targets(train_frame)
        mask = torch.from_numpy(np.isfinite(hazard).astype(np.float32))
        target = torch.from_numpy(np.nan_to_num(hazard).astype(np.float32))
        positive = (target * mask).sum(dim=0)
        observed_count = mask.sum(dim=0)
        rates = torch.where(observed_count > 0, (positive + 0.5) / (observed_count + 1.0), 0.5).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            model.head.net[-1].bias.copy_(torch.logit(rates).to(device))
        loader = DataLoader(TensorDataset(volumes[train.to_numpy()], target, mask), batch_size=2, shuffle=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
        amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        history = []
        for epoch in range(8):
            model.train()
            losses = []
            for x, y, observed in loader:
                x, y, observed = x.to(device), y.to(device), observed.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=amp):
                    loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * observed).sum() / observed.sum().clamp_min(1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
            print(f"CT model {fold}/3, epoch {epoch + 1}/8: loss={history[-1]['loss']:.4f}", flush=True)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, output / f"dino_fold{fold}.pt")
        pd.DataFrame(history).to_csv(output / f"training_fold{fold}.csv", index=False)
        frame[dl_columns(fold)] = infer_hazards(model, volumes, device)
    cols = [*IDENTITY, *Q_FEATURES, *C_FEATURES, *[c for fold in FOLDS for c in dl_columns(fold)], "event_days", "censor_days", "progression_event_by_365d", "fold"]
    frame[cols].to_csv(output / "prepared.csv", index=False)
    run_experiment(frame[cols], output / "experiment")
    (output / "training_config.json").write_text(json.dumps({"seed": seed, "epochs": 8, "batch_size": 2, "learning_rate": 1e-4, "weight_decay": 1e-4, "device": device}, indent=2) + "\n")
