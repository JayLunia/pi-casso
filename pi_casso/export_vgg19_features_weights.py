from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export a *truncated* VGG19.features state_dict (no classifier weights) "
            "for Raspberry Pi. This matches the common NST conv_1..conv_5 loss layers."
        )
    )
    p.add_argument("--out", type=str, required=True, help="Output .pth path.")
    p.add_argument(
        "--max-conv",
        type=int,
        default=5,
        help="Export up to conv_{max_conv} (recommended: 5).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    try:
        from torchvision.models import VGG19_Weights, vgg19  # type: ignore
    except Exception as e:
        raise SystemExit(
            "torchvision is required for this export script. "
            "Run it on your dev machine (not necessarily on the Pi)."
        ) from e

    max_conv = int(args.max_conv)
    if max_conv < 1:
        raise SystemExit("--max-conv must be >= 1")

    model = vgg19(weights=VGG19_Weights.DEFAULT).eval()
    features = model.features

    # Find the slice end index corresponding to conv_{max_conv}.
    conv_count = 0
    end_idx = None
    for idx, layer in enumerate(features):
        if layer.__class__.__name__ == "Conv2d":
            conv_count += 1
            if conv_count == max_conv:
                end_idx = idx + 1
                break
    if end_idx is None:
        raise SystemExit(f"Could not find conv_{max_conv} in VGG19.features")

    sliced = features[:end_idx]
    sd = sliced.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)  slice_end={end_idx}  convs={max_conv}")


if __name__ == "__main__":
    main()

