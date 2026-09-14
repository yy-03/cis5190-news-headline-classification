"""Fetch the fixed upstream course checkpoint and verify its integrity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "model.pt")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model-manifest.json").read_text())
    target = args.output.resolve()
    expected = manifest["sha256"]
    if target.exists():
        if sha256(target) != expected:
            raise SystemExit(f"Existing file does not match the recorded checkpoint: {target}")
        print(f"Verified existing checkpoint: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    url = ("https://raw.githubusercontent.com/Anthoneeee/ember-text-notes/"
           + manifest["source_commit"] + "/model.pt")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "cis5190-project-downloader"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as dest:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                dest.write(block)
        if temporary.stat().st_size != manifest["size_bytes"] or sha256(temporary) != expected:
            raise RuntimeError("Checkpoint integrity verification failed")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified checkpoint: {target}")

if __name__ == "__main__":
    main()
