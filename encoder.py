"""
Frozen, pretrained vision encoder used as the representation space for all
three stages. No fine-tuning / gradient updates ever happen here -- this is
what makes the detector "training-free".

Primary choice (per the methodology, Section 22-24): DINOv2 ViT-S/14.
A `DummyEncoder` random-projection fallback is provided purely so the rest
of the pipeline can be unit-tested in environments without internet access
to download pretrained weights (e.g. offline CI). Swap in the real
`FrozenEncoder` for any actual experiment.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenEncoder(nn.Module):
    """
    Wraps a pretrained, frozen backbone and returns L2-normalized
    global (CLS/pooled) embeddings.

    backbone_name options (all frozen, eval mode, no_grad):
      - "dinov2_vits14" / "dinov2_vitb14"  (via torch.hub facebookresearch/dinov2)
      - "dinov2-small" / "dinov2-base"     (via HuggingFace transformers, facebook/dinov2-*)
      - any torchvision classifier name (fallback edge baseline, e.g. "mobilenet_v3_small")
    """

    def __init__(self, backbone_name: str = "dinov2_vits14", device: str = None,
                 image_size: int = 224):
        super().__init__()
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mode = None
        self.model = self._build(backbone_name)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ImageNet normalization used by DINOv2 / most torchvision backbones
        self.register_mean_std()

    def register_mean_std(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _build(self, name: str):
        if name.startswith("dinov2_") :
            # torch.hub route: requires network access to github.com/facebookresearch/dinov2
            self._mode = "hub"
            model = torch.hub.load("facebookresearch/dinov2", name)
            return model
        if name.startswith("dinov2-"):
            # transformers route: requires network access to huggingface.co
            from transformers import AutoModel
            self._mode = "hf"
            hf_name = f"facebook/{name}"
            model = AutoModel.from_pretrained(hf_name)
            return model
        # torchvision fallback (e.g. small CNN edge baseline)
        import torchvision.models as tvm
        self._mode = "tv"
        ctor = getattr(tvm, name)
        model = ctor(weights="DEFAULT")
        model.classifier = nn.Identity() if hasattr(model, "classifier") else nn.Identity()
        return model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) float tensor in [0, 1], any spatial size.
        returns: (B, D) L2-normalized embedding.
        """
        x = x.to(self.device)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode="bilinear", align_corners=False)
        x = (x - self.mean.to(self.device)) / self.std.to(self.device)

        if self._mode == "hub":
            feats = self.model(x)  # (B, D) pooled CLS token by default for dinov2 hub models
        elif self._mode == "hf":
            out = self.model(pixel_values=x)
            feats = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
        else:
            feats = self.model(x)
            if feats.dim() > 2:
                feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return F.normalize(feats, dim=-1)

    @property
    def embed_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            return self.forward(dummy).shape[-1]

    # one forward pass through the encoder == 1 unit of computational cost
    unit_cost = 1.0


class DummyEncoder(nn.Module):
    """
    Deterministic random-projection encoder with NO network dependency.
    Used only to exercise/unit-test the pipeline end-to-end offline.
    It is intentionally sensitive to high-frequency pixel changes (a fixed
    random linear projection is enough to make Stage 1/2/3 scores non-trivial)
    but it carries no forensic meaning -- do not use it for real experiments.
    """

    def __init__(self, out_dim: int = 256, image_size: int = 224, seed: int = 0):
        super().__init__()
        self.image_size = image_size
        self.device = "cpu"
        g = torch.Generator().manual_seed(seed)
        in_dim = 3 * image_size * image_size
        # fixed (non-trainable) random projection -> acts like a frozen encoder
        self.register_buffer("proj", torch.randn(in_dim, out_dim, generator=g) / (in_dim ** 0.5))
        self.out_dim = out_dim
        self.unit_cost = 0.05  # far cheaper than a real ViT forward pass

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode="bilinear", align_corners=False)
        flat = x.flatten(1)
        feats = flat @ self.proj
        return F.normalize(feats, dim=-1)

    @property
    def embed_dim(self) -> int:
        return self.out_dim


def load_encoder(name: str = "dinov2_vits14", offline: bool = False, **kwargs):
    """
    Convenience factory.
      offline=True  -> DummyEncoder (no downloads, for smoke-testing the pipeline)
      offline=False -> real FrozenEncoder(name)
    """
    if offline:
        return DummyEncoder(**kwargs)
    return FrozenEncoder(backbone_name=name, **kwargs)
