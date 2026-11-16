"""Fetch official WP1a archives without evaluating labels or modifying legacy data."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "data/revision/isic2018_official/archives"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = []
    for url in json.loads((ROOT / "docs/provenance/sources/isic_downloads.json").read_text()):
        path = directory / url.rsplit("/", 1)[1]
        if not path.exists():
            temporary = path.with_suffix(".partial")
            if temporary.exists():
                raise FileExistsError(f"partial archive needs investigation: {temporary}")
            if shutil.disk_usage(directory).free < 25 * 1024**3:
                raise OSError("disk below 25 GiB floor")
            print("Downloading", url, flush=True)
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("xb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            temporary.rename(path)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP CRC failed: {path}")
            members = archive.namelist()
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt.append(
            dict(
                url=url,
                path=str(path.relative_to(ROOT)),
                sha256=sha,
                bytes=path.stat().st_size,
                members=len(members),
                license_source="https://challenge.isic-archive.com/data/",
                license="CC0 for 2018 Task1 row; final figure approval remains human",
            )
        )
        (ROOT / "docs/provenance/sources/isic_acquisition.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print("Verified", path.name, path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
