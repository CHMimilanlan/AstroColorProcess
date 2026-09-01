"""Re-download official model/sample assets with integrity verification."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = {
    ROOT / "models" / "deep_denoise_cnn_AI3_6.pth": (
        "https://github.com/setiastro/cosmicclarity/releases/download/Linux/deep_denoise_cnn_AI3_6.pth",
        "f860e234dc6a3ab1acacbea399c235179c86cb8b287f2ab9010bc18c8af08671",
    ),
    ROOT / "samples" / "M51_short.jpg": (
        "https://raw.githubusercontent.com/Steffenhir/AstroNoiseNet/main/examples/M51_short.jpg",
        None,
    ),
    ROOT / "samples" / "M51_long_reference.jpg": (
        "https://raw.githubusercontent.com/Steffenhir/AstroNoiseNet/main/examples/M51_long.jpg",
        None,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for destination, (url, expected_hash) in ASSETS.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual_hash = sha256(temporary)
        if expected_hash and actual_hash != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {destination.name}: {actual_hash}")
        temporary.replace(destination)
        print(f"Saved {destination} (sha256={actual_hash})")


if __name__ == "__main__":
    main()

