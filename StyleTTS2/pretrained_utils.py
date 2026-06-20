from __future__ import annotations

import os
import os.path as osp
import urllib.request
from pathlib import Path


KOKORO_BASE_URL = (
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth"
)


def _download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_pretrained_model(model_path: str, url: str | None = None) -> str:
    if not model_path:
        return model_path

    path = Path(model_path)
    if path.exists():
        return model_path

    download_url = url
    if download_url is None and path.name == "kokoro_base.pth":
        download_url = KOKORO_BASE_URL
    if download_url is None:
        raise FileNotFoundError(
            f"Missing pretrained model: {model_path}. "
            "Set pretrained_model_url in the config to auto-download it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained model to {osp.abspath(model_path)} ...")
    _download_file(download_url, path)
    return model_path
