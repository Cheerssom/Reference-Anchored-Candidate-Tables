"""Frozen CLIP ViT encoder (timm backend) with the same interface as ResNet32.

forward(x) -> (context_tokens, pooled)
  context_tokens : (B, 49, d)   patch tokens (ViT-B/32 @ 224px)
  pooled         : (B, d)       CLS token

The typed decision head is expected to earn its keep on these richer
features: cross-attention over 49 patch tokens can align fine-grained
regions against candidate prototypes, which a 4x4x64 ResNet map could not
support (design.md v2 lesson: the CIFAR/ResNet pilot capped the head at
the NCM-cos floor).

Transforms: use TF_CLIP from train.py (Resize 224, CLIP normalization).
"""

import torch
import torch.nn as nn

DEFAULT_MODEL = "vit_base_patch32_clip_224.openai"  # CLIP ViT-B/32, openai weights


class CLIPVisionEncoder(nn.Module):
    dim = 768

    def __init__(self, model_name=DEFAULT_MODEL):
        super().__init__()
        import timm
        self.net = timm.create_model(model_name, num_classes=0, pretrained=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.n_patches = getattr(self.net, "patch_embed", None)
        # forward_features returns the full token sequence (B, 1+P, d)
        self._forward_features = self.net.forward_features

    def train(self, mode=True):
        # always keep BN/eval stats frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        tokens = self.net.forward_features(x)          # B, 1+P, d
        return tokens[:, 1:], tokens[:, 0]             # patches, CLS


# CLIP normalization statistics (for train.py's TF_CLIP)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
