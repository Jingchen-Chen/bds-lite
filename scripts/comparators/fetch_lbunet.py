"""Fetch the official LB-UNet model file at a pinned commit (not redistributed).

Upstream: https://github.com/xuxuxuxuxuxjh/LB-UNet (MICCAI 2024, Xu & Tong).
The repository declares no license, so the file is downloaded locally into
third_party/ (git-ignored) and verified by SHA-256 instead of being vendored.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "913ee687ecc4514508ec20f01af0df5b0b5af5cc"
SHA256 = "a41b91d3b9787c55fbc9e9f2103847f1b452708b16dc583d4a32522a60dac047"
URL = f"https://raw.githubusercontent.com/xuxuxuxuxuxjh/LB-UNet/{COMMIT}/lbunet.py"
PATH = ROOT / "third_party" / "lbunet" / COMMIT / "lbunet.py"


def ensure() -> Path:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            PATH.write_bytes(response.read())
    digest = hashlib.sha256(PATH.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"LB-UNet source hash mismatch: {digest}")
    return PATH


if __name__ == "__main__":
    print(ensure())
