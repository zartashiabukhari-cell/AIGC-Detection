"""
Pilot experiment driver (Section 34): the Phase-1 prototype the methodology
recommends running BEFORE committing to the full paper.

Usage:
    # Smoke test with synthetic data + offline dummy encoder (no downloads,
    # runs anywhere -- just verifies the pipeline logic is correct):
    python main.py --synthetic --offline

    # Real experiment on a downloaded dataset folder (e.g. a GenImage subset
    # reorganized into root/real and root/fake), with the real DINOv2 encoder:
    python main.py --data_root /path/to/genimage_subset --backbone dinov2_vits14

What it does:
  1. Loads (or synthesizes) a real/fake image dataset.
  2. Splits off a small calibration set (Section 12) -- never touched again.
  3. Runs all three stages on the calibration set to fit (tau_R, tau_F) for
     Stage 1 and for the fused Stage 1+2 score.
  4. Evaluates on the held-out test set: AUC + average cost for
     A) Stage-1-only, B) Stage-1+2, C) full 3-stage, D) proposed adaptive.
  5. Prints the Section 18 / 19 -style comparison table.
"""
from __future__ import annotations
import argparse
import torch

from encoder import load_encoder
from dataset import RealFakeImageDataset, split_calibration_test
from pipeline import AdaptiveDetector, PipelineConfig, StageThresholds
from calibration import calibrate_two_thresholds
from evaluate import run_ablation, print_ablation_table


def make_synthetic_dataset(n_per_class: int = 60, image_size: int = 224, seed: int = 0):
    """
    Purely for smoke-testing the pipeline without any real dataset:
    "real" = smooth low-frequency images, "fake" = images with extra
    injected high-frequency structure, so Stage 1 has genuine (synthetic)
    signal to detect and the whole pipeline can be sanity-checked end to end.
    """
    g = torch.Generator().manual_seed(seed)
    images, labels = [], []
    for _ in range(n_per_class):
        base = torch.rand(3, image_size, image_size, generator=g)
        smooth = torch.nn.functional.avg_pool2d(base.unsqueeze(0), 8, stride=1, padding=4)
        smooth = torch.nn.functional.interpolate(smooth, size=(image_size, image_size))[0]
        images.append(smooth.clamp(0, 1))
        labels.append(0)
    for _ in range(n_per_class):
        base = torch.rand(3, image_size, image_size, generator=g)
        high_freq_noise = 0.15 * torch.randn(3, image_size, image_size, generator=g)
        images.append((base + high_freq_noise).clamp(0, 1))
        labels.append(1)
    return images, labels


def build_calibration_scores(detector, images, labels, seed=0):
    d_f_real, d_f_fake = [], []
    comb_real, comb_fake = [], []
    for img, label in zip(images, labels):
        raw = detector.raw_scores(img.unsqueeze(0), seed=seed)
        (d_f_fake if label == 1 else d_f_real).append(raw["d_f"].item())
        (comb_fake if label == 1 else comb_real).append(raw["combined_12"].item())
    return d_f_real, d_f_fake, comb_real, comb_fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=None,
                     help="Folder with real/ and fake/ subdirs (e.g. GenImage subset).")
    ap.add_argument("--synthetic", action="store_true",
                     help="Use synthetic smoke-test data instead of a real dataset.")
    ap.add_argument("--offline", action="store_true",
                     help="Use the no-download DummyEncoder instead of real DINOv2.")
    ap.add_argument("--backbone", type=str, default="dinov2_vits14")
    ap.add_argument("--n_calib_per_class", type=int, default=25)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_crop_views", type=int, default=4)
    ap.add_argument("--n_patches", type=int, default=9)
    ap.add_argument("--criterion", type=str, default="youden",
                     choices=["youden", "eer", "target_fpr"])
    args = ap.parse_args()

    encoder = load_encoder(args.backbone, offline=args.offline, image_size=args.image_size)

    if args.synthetic or args.data_root is None:
        images, labels = make_synthetic_dataset(image_size=args.image_size)
        calib_n = min(args.n_calib_per_class, len(images) // 4)
        calib_images = images[:calib_n] + images[len(images) // 2: len(images) // 2 + calib_n]
        calib_labels = labels[:calib_n] + labels[len(images) // 2: len(images) // 2 + calib_n]
        test_images = images[calib_n: len(images) // 2] + images[len(images) // 2 + calib_n:]
        test_labels = labels[calib_n: len(images) // 2] + labels[len(images) // 2 + calib_n:]
    else:
        ds = RealFakeImageDataset(args.data_root, image_size=args.image_size)
        calib_idx, test_idx = split_calibration_test(ds, n_calib_per_class=args.n_calib_per_class)
        calib_images = [ds[i][0] for i in calib_idx]
        calib_labels = [ds[i][1] for i in calib_idx]
        test_images = [ds[i][0] for i in test_idx]
        test_labels = [ds[i][1] for i in test_idx]

    # ---- placeholder config just to run Stage 1/2/3 on the calibration set ----
    placeholder = PipelineConfig(
        stage1=StageThresholds(tau_R=0.0, tau_F=1.0, gamma=1.0),
        stage2=StageThresholds(tau_R=0.0, tau_F=1.0, gamma=1.0),
        n_crop_views=args.n_crop_views, n_patches=args.n_patches,
    )
    calib_detector = AdaptiveDetector(encoder, placeholder)
    d_f_real, d_f_fake, comb_real, comb_fake = build_calibration_scores(
        calib_detector, calib_images, calib_labels)

    tau_R1, tau_F1 = calibrate_two_thresholds(d_f_real, d_f_fake, criterion=args.criterion)
    tau_R2, tau_F2 = calibrate_two_thresholds(comb_real, comb_fake, criterion=args.criterion)
    print(f"Stage 1 thresholds:      tau_R={tau_R1:.4f}  tau_F={tau_F1:.4f}")
    print(f"Stage 1+2 thresholds:    tau_R={tau_R2:.4f}  tau_F={tau_F2:.4f}")

    config = PipelineConfig(
        stage1=StageThresholds(tau_R=tau_R1, tau_F=tau_F1, gamma=1.0),
        stage2=StageThresholds(tau_R=tau_R2, tau_F=tau_F2, gamma=1.0),
        n_crop_views=args.n_crop_views, n_patches=args.n_patches,
    )
    detector = AdaptiveDetector(encoder, config)

    results = run_ablation(detector, test_images, test_labels)
    full_cost = detector.s1.cost + detector.s2.cost + detector.s3.cost
    print()
    print_ablation_table(results, full_cost)


if __name__ == "__main__":
    main()
