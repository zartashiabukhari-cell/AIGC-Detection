"""
Lightweight dataset loaders. Training-free detection needs no train split --
only (1) a small CALIBRATION split to fit the two thresholds per stage
(Section 12) and (2) a TEST split that is never touched during calibration.

Works with the standard layout used by GenImage / most public AIGC benchmarks:

    root/
      real/   (or "nature")
        img001.jpg ...
      fake/   (or "ai")
        img001.jpg ...

Also includes a bias-controlled loading helper (Section 21, "Fake or JPEG?")
that matches real/fake images on resolution and re-encodes both at a fixed
JPEG quality so the detector cannot shortcut on compression artifacts.
"""
from __future__ import annotations
import io
import random
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


REAL_DIR_NAMES = ("real", "nature", "0_real")
FAKE_DIR_NAMES = ("fake", "ai", "1_fake", "generated")


def _find_subdir(root: Path, candidates) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


class RealFakeImageDataset(Dataset):
    """
    Generic real-vs-fake image folder dataset.
    label: 0 = real, 1 = AI-generated.
    """

    def __init__(self, root: str, image_size: int = 224,
                 match_jpeg_quality: Optional[int] = 95,
                 max_per_class: Optional[int] = None, seed: int = 0):
        root = Path(root)
        real_dir = _find_subdir(root, REAL_DIR_NAMES)
        fake_dir = _find_subdir(root, FAKE_DIR_NAMES)
        if real_dir is None or fake_dir is None:
            raise FileNotFoundError(
                f"Expected subfolders like real/ and fake/ (or nature/ and ai/) under {root}, "
                f"found: {[p.name for p in root.iterdir() if p.is_dir()]}"
            )

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        real_paths = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in exts)
        fake_paths = sorted(p for p in fake_dir.rglob("*") if p.suffix.lower() in exts)

        rng = random.Random(seed)
        rng.shuffle(real_paths)
        rng.shuffle(fake_paths)
        if max_per_class is not None:
            real_paths = real_paths[:max_per_class]
            fake_paths = fake_paths[:max_per_class]

        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        rng.shuffle(self.samples)
        self.image_size = image_size
        self.match_jpeg_quality = match_jpeg_quality

    def __len__(self):
        return len(self.samples)

    def _load(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.match_jpeg_quality is not None:
            # re-encode both classes at the same JPEG quality (Section 21 bias control)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.match_jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._load(path)
        tensor = TF.to_tensor(img)  # -> (3, H, W) in [0, 1]
        return tensor, label, str(path)


def split_calibration_test(dataset: RealFakeImageDataset, n_calib_per_class: int = 500,
                            seed: int = 0):
    """
    Returns (calibration_indices, test_indices) with an approximately balanced
    calibration split, disjoint from the test split, as recommended in
    Section 12 (e.g. 500 real + 500 fake for calibration only).
    """
    by_label = {0: [], 1: []}
    for i, (_, label) in enumerate(dataset.samples):
        by_label[label].append(i)

    rng = random.Random(seed)
    calib_idx, test_idx = [], []
    for label, idxs in by_label.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = min(n_calib_per_class, len(idxs) // 2)  # never let calibration eat the whole split
        calib_idx.extend(idxs[:n])
        test_idx.extend(idxs[n:])
    return calib_idx, test_idx
