from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VGG19SliceSpec:
    """Spec for the VGG19 features slice used for losses."""

    max_conv: int = 5


def build_vgg19_features_until_conv(max_conv: int = 5) -> nn.Sequential:
    """Build a VGG19.features-like Sequential, truncated after conv_{max_conv}.

    The returned module matches the layer structure of torchvision's VGG19.features
    up to the same cutoff, so a `state_dict` extracted from that slice can be loaded.
    """
    cfg: List[int | str] = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]

    layers: List[nn.Module] = []
    in_channels = 3
    conv_count = 0
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        out_channels = int(v)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        conv_count += 1
        if conv_count >= max_conv:
            break
        layers.append(nn.ReLU(inplace=False))
        in_channels = out_channels

    return nn.Sequential(*layers)


def extract_conv_features(
    features: nn.Sequential,
    x: torch.Tensor,
    conv_ids: Iterable[int],
) -> Dict[int, torch.Tensor]:
    """Run `features(x)` and return outputs at selected conv IDs (1-indexed)."""
    want: Set[int] = set(int(i) for i in conv_ids)
    out: Dict[int, torch.Tensor] = {}
    conv_count = 0
    h = x
    for layer in features:
        h = layer(h)
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            if conv_count in want:
                out[conv_count] = h
    return out

