"""CT preprocessing, exact-grid Q7 features, and the selected DINOv3 network.

The architecture and tensor names match the development checkpoints. Optional
imaging dependencies are kept out of the lightweight CSV quickstart.
"""

from __future__ import annotations

import json
import math
import random
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


def preprocess_ct(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("CT must be a finite 3D NIfTI volume in Hounsfield units.")
    if nib.aff2axcodes(image.affine)[2] not in ("S", "I"):
        raise ValueError("The NIfTI third axis must represent axial slices.")
    array = (np.clip(array, -1000, 400) + 1000) / 1400
    array = array.transpose(2, 1, 0)
    return zoom(array, [s / n for s, n in zip((48, 96, 96), array.shape)], order=1).astype(np.float32)


def qct_features(path: Path, mask_dir: Path, treated_lobe: str) -> dict:
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("QCT requires a finite 3D CT volume.")
    masks = {}
    for lobe, filename in LOBES.items():
        mask = nib.load(str(mask_dir / f"{filename}.nii.gz"))
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


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, [*IDENTITY, *C_FEATURES, "ct_path", "mask_dir", "treated_lobe"])
    for col in ("ct_path", "mask_dir"):
        frame[col] = frame[col].map(lambda v: str((path.parent / str(v)).resolve()) if not Path(str(v)).is_absolute() else str(v))
    feature_rows = []
    for _, row in frame.loc[frame["is_baseline"].eq(1)].iterrows():
        feature_rows.append({"patient_id": row["patient_id"], **qct_features(Path(row["ct_path"]), Path(row["mask_dir"]), str(row["treated_lobe"]).strip().upper())})
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


def prepare_ct(manifest: Path, weights: Path, output: Path, device: str = "cpu") -> None:
    frame = read_manifest(manifest)
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


def train_ct(manifest: Path, output: Path, device: str = "cpu", seed: int = 20260714, initial_backbone: Path | None = None) -> None:
    from .training import assign_folds, run_experiment
    frame = assign_folds(eligible_cohort(validate(read_manifest(manifest), outcomes=True, dl=False)))
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
