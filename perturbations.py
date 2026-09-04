"""
Perturbation / view generators for each stage.

Stage 1 (frequency):  I_f = I + delta_f, delta_f a controlled high-frequency
                       perturbation built from the image's own Fourier spectrum.
Stage 2 (spatial):     k independent crop/resize views of I.
Stage 3 (patch):       a regular grid of n x n local patches for localized
                       frequency-forensic analysis.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def frequency_perturbation(img: torch.Tensor, strength: float = 0.08,
                            high_freq_radius_frac: float = 0.6) -> torch.Tensor:
    """
    Structured high-frequency perturbation via 2D FFT (Section 10).

    img: (B, 3, H, W) in [0, 1]
    strength: perturbation magnitude, delta_f = strength * high_freq_component
    high_freq_radius_frac: fraction of the Nyquist radius above which a
        frequency is considered "high" (mask = 1 outside this radius).

    Returns I_f = clamp(I + delta_f, 0, 1)
    """
    B, C, H, W = img.shape
    fft = torch.fft.fft2(img, norm="ortho")
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

    # build a radial high-pass mask (same for every image / channel)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij",
    )
    radius = torch.sqrt(xx ** 2 + yy ** 2)
    mask = (radius > high_freq_radius_frac).float()  # 1 = keep (high freq), 0 = zero out low freq
    mask = mask.view(1, 1, H, W)

    high_freq_fft = fft_shift * mask
    high_freq_spatial = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)),
                                         norm="ortho").real

    # normalize perturbation energy per-image so `strength` is comparable across images
    norm = high_freq_spatial.flatten(1).norm(dim=1).clamp_min(1e-8).view(B, 1, 1, 1)
    delta_f = strength * high_freq_spatial / norm * (H * W) ** 0.5

    return (img + delta_f).clamp(0.0, 1.0)


@torch.no_grad()
def crop_views(img: torch.Tensor, k: int = 4, min_scale: float = 0.5,
               max_scale: float = 0.9, seed: int = None) -> list[torch.Tensor]:
    """
    Stage 2 (Section 13): generate k spatial views I_1..I_k of the image via
    random-resized-crop-style sampling, each resized back to the original
    resolution so they can be re-encoded.

    Deterministic per-call if `seed` is given (recommended for calibration /
    reproducible evaluation).
    """
    B, C, H, W = img.shape
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    views = []
    for i in range(k):
        scale = min_scale + (max_scale - min_scale) * torch.rand(1, generator=gen).item()
        ch, cw = int(H * scale), int(W * scale)
        top = torch.randint(0, max(H - ch, 1), (1,), generator=gen).item()
        left = torch.randint(0, max(W - cw, 1), (1,), generator=gen).item()
        crop = img[:, :, top:top + ch, left:left + cw]
        resized = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        views.append(resized)
    return views


@torch.no_grad()
def patch_split(img: torch.Tensor, n_patches: int = 9) -> list[torch.Tensor]:
    """
    Stage 3 (Section 15): split the image into a regular sqrt(n) x sqrt(n)
    grid of patches, each upsampled back to the encoder's expected size by
    the caller (encoders already interpolate internally).

    n_patches should be a perfect square (4, 9, 16, ...); if not, the closest
    grid <= n_patches is used.
    """
    B, C, H, W = img.shape
    g = max(2, int(n_patches ** 0.5))
    ph, pw = H // g, W // g
    patches = []
    for r in range(g):
        for c in range(g):
            patch = img[:, :, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            patches.append(patch)
    return patches
