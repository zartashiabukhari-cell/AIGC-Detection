# Adaptive, Training-Free AIGC Detection (Early-Exit Prototype)

Implements the Phase-1 pilot described in the methodology: a three-stage,
training-free AI-generated-image detector with a rule-based, two-threshold
early-exit policy, so that only ambiguous images pay for the expensive
forensic stages.

```
Stage 1  Global frequency-perturbation sensitivity   (cheapest, ~2 encoder passes)
Stage 2  Spatial / cropping representation consistency (+k encoder passes)
Stage 3  Multi-patch local frequency forensic analysis (+2*n_patches encoder passes)
```

No detector weights are ever trained. The only "fitting" that happens is
choosing two scalar thresholds (τ_R, τ_F) per stage on a small calibration
split — the rest of the pipeline is closed-form (FFT high-pass filtering,
cosine similarity, robust aggregation).

## Files

| File | Contents |
|---|---|
| `encoder.py` | Frozen pretrained backbone wrapper (`FrozenEncoder`, default DINOv2 ViT-S/14) + `DummyEncoder` offline fallback for smoke-testing |
| `perturbations.py` | FFT high-frequency perturbation, random-resized-crop views, patch grid split |
| `stages.py` | `Stage1Frequency`, `Stage2Spatial`, `Stage3Patch` scoring modules |
| `calibration.py` | Two-threshold calibration (`youden` / `eer` / `target_fpr` criteria) |
| `pipeline.py` | `AdaptiveDetector` — the early-exit policy + fixed-depth ablation variants |
| `dataset.py` | Generic `real/`+`fake/` folder loader, JPEG-quality bias control (Section 21), calibration/test split |
| `evaluate.py` | AUC, average computational cost, the A/B/C/D ablation table |
| `main.py` | End-to-end driver: calibrate → evaluate → print table |

## Quick start (no downloads, verifies the pipeline logic)

```bash
pip install -r requirements.txt
python main.py --synthetic --offline
```

This uses a random-projection `DummyEncoder` and synthetic images purely to
confirm the plumbing (FFT perturbation → embeddings → thresholds → early
exit → cost accounting) runs correctly end to end. It carries **no forensic
meaning** — do not report these numbers.

## Real experiment

```bash
# reorganize a downloaded dataset into: root/real/*.jpg  and  root/fake/*.jpg
python main.py --data_root /path/to/genimage_subset --backbone dinov2_vits14
```

The first run will download DINOv2 weights via `torch.hub`
(`facebookresearch/dinov2`), which requires outbound network access to
`github.com` (and, for the HF route, `huggingface.co`).

Key flags:
- `--backbone` : `dinov2_vits14` (default), `dinov2_vitb14`, `mobilenet_v3_small` (edge baseline), or `dinov2-small`/`dinov2-base` (HF route)
- `--n_calib_per_class` : calibration-only images per class (paper suggests ~500; keep it small and never reuse for testing)
- `--criterion` : `youden` (default), `eer`, or `target_fpr` for threshold selection
- `--n_crop_views`, `--n_patches` : Stage 2 / Stage 3 breadth (cost knobs)

## Recommended pilot protocol (Section 34)

1. ~1,000 real + ~1,000 generated images.
2. Backbone: DINOv2 ViT-S/14.
3. Compare A) Stage-1-only, B) Stage-1+2, C) full 3-stage, D) proposed
   adaptive — `run_ablation()` in `evaluate.py` does this in one call.
4. If D doesn't land near C's AUC at meaningfully less average cost, revisit
   thresholds/backbone before scaling up to the full paper experiments.

---

## Publicly available datasets (verified links)

**Primary — GenImage** (million-scale, 7 generators incl. Midjourney, SD,
ADM, GLIDE, Wukong, VQDM, BigGAN; matches ImageNet's 1000 classes; includes
degraded-image splits):
- Repo: https://github.com/GenImage-Dataset/GenImage (Baidu Yunpan links for the full train/test data are posted there; check the repo's issues/homepage for any mirrored Google Drive links, as availability changes over time)

**GenImage++** (NeurIPS 2025, test-only, stresses generalization to modern
generators — good "unseen generator" stress test on top of GenImage):
- https://huggingface.co/datasets/Lunahera/genimagepp (mirror: https://modelscope.cn/datasets/Anahera/Genimage-plus-plus)

**UniversalFakeDetect / ForenSynths ("Ojha et al." CNN-generated image
detection benchmark)** — for comparison against the 2023 universal-detection
literature:
- Bundled inside the large aggregated benchmark below (`Ojha.tar`, `ForenSynths.part.*`)

**Large aggregated AIGC-detection benchmark mirror** (bundles GenImage
subsets, ForenSynths/UniversalFakeDetect, DiffusionForensics, AntifakePrompt,
GANGen-Detection, Celeb-DF, and more into one HF dataset repo — convenient if
you want several benchmarks without chasing individual mirrors):
- https://huggingface.co/datasets/nebula/dfbenchmark

**DRCT-2M** (million-scale, 16 diffusion-model variants incl. SDv1.4/1.5/2,
SDXL, SDXL-Turbo, LCM, ControlNet variants, and diffusion-reconstruction hard
negatives; explicitly built to probe generalization):
- Code + dataset access instructions: https://github.com/beibuwandeluori/DRCT
- Paper: https://proceedings.mlr.press/v235/chen24ay.html

**DiffusionForensics** (used by the DIRE / AEROBLADE line of training-free
reconstruction-based detectors — useful baseline dataset for your Stage-1
frequency-perturbation comparisons):
- https://github.com/ZhendongWang6/DIRE (BaiduDrive / RecDrive links in the README, password `dire`)
- Also bundled in `nebula/dfbenchmark` above (`DiffusionForensics.tar`)

**Notes on bias control (Section 21):** whichever dataset(s) you use, match
real and fake images on resolution, JPEG quality, and file format before
scoring — `dataset.py`'s `RealFakeImageDataset(match_jpeg_quality=95, ...)`
re-encodes both classes at the same JPEG quality for exactly this reason.

Dataset licensing/availability on these mirrors can change; always check
each repository's current README/dataset card before depending on a link in
a paper submission.
