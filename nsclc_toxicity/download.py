"""Download versioned public assets, verify SHA256, and place them in weights/."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights(destination: Path, *, imaging: bool = False) -> None:
    manifest = json.loads(Path(__file__).with_name("weights_manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        if asset["group"] == "imaging" and not imaging:
            continue
        name = asset["name"]
        if Path(name).name != name:
            raise ValueError("Invalid weight filename in manifest.")
        target = destination / name
        if target.is_file() and sha256(target) == asset["sha256"]:
            print(f"Verified {target}")
            continue
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "NSCLC-Toxicity/1.0"})
        temporary = target.with_name(target.name + ".part")
        try:
            print(f"Downloading {name} ({asset['bytes'] / 1e6:.1f} MB)")
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if temporary.stat().st_size != asset["bytes"] or sha256(temporary) != asset["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}; rerun the download.")
            os.replace(temporary, target)
            print(f"Verified {target}")
        except (OSError, urllib.error.URLError) as exc:
            raise ValueError(f"Could not download {name}. Check your internet connection.\n{asset['url']}") from exc
        finally:
            temporary.unlink(missing_ok=True)
