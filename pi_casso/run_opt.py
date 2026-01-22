from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


from pi_casso.vgg19_features import build_vgg19_features_until_conv, extract_conv_features


VGG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
VGG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pi-casso: optimization-based Neural Style Transfer (no training).\n"
            "CLI is intentionally minimal: only content/style/out.\n"
            "Make sure `vgg19_conv1_to_conv5.pth` exists in the repo root (or set PI_CASSO_VGG_WEIGHTS)."
        )
    )
    p.add_argument("--content", type=str, required=True, help="Path to content image.")
    p.add_argument("--style", type=str, required=True, help="Path to style image (single style).")
    p.add_argument("--out", type=str, required=True, help="Output image path.")
    p.add_argument(
        "--inky",
        action="store_true",
        help="Display the final result on a Pimoroni Inky e-ink display (if available).",
    )
    p.add_argument(
        "--inky-type",
        type=str,
        default="",
        help="Force a specific Inky type if auto-detect fails (eg: phat, what, impressions).",
    )
    p.add_argument(
        "--inky-colour",
        type=str,
        default="black",
        choices=["black", "red", "yellow"],
        help="Inky colour for mono/tri-colour displays (eg: black, red, yellow).",
    )
    p.add_argument(
        "--inky-border",
        type=str,
        default="white",
        choices=["white", "black", "red", "yellow"],
        help="Inky border colour (white/black/red/yellow).",
    )
    p.add_argument(
        "--inky-rotate",
        type=int,
        default=0,
        help="Rotate clockwise degrees before display (0/90/180/270).",
    )
    p.add_argument(
        "--inky-saturation",
        type=float,
        default=0.5,
        help="Saturation for colour Inky displays (only if supported by the driver).",
    )
    return p


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_pyramid(arg: str, fallback: int) -> List[int]:
    if not arg.strip():
        return [int(fallback)]
    sizes = [int(s) for s in arg.split(",") if s.strip()]
    sizes = [s for s in sizes if s > 0]
    return sizes or [int(fallback)]

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _import_inky_package() -> None:
    """
    Try to make `import inky` resolve to the Pimoroni Inky Python package.

    This repo vendors Pimoroni's inky library under `./inky/inky/`. When running from the repo
    root, `import inky` can otherwise resolve to a namespace package (`./inky/`) which breaks
    imports like `inky.auto`.
    """

    try:
        importlib.import_module("inky.auto")
        return
    except Exception:
        pass

    vendor_root = _repo_root() / "inky"
    if vendor_root.exists():
        vendor_root_s = str(vendor_root)
        if vendor_root_s not in sys.path:
            sys.path.insert(0, vendor_root_s)
        sys.modules.pop("inky", None)
        sys.modules.pop("inky.auto", None)


def _init_inky_display(inky_type: str, inky_colour: str):
    _import_inky_package()

    if inky_type:
        t = inky_type.strip().lower()
        c = inky_colour.strip().lower()
        if t == "phat":
            from inky.phat import InkyPHAT

            return InkyPHAT(c)
        if t == "what":
            from inky.what import InkyWHAT

            return InkyWHAT(c)
        if t in {"impressions", "impression", "7colour", "uc8159"}:
            from inky.inky_uc8159 import Inky as InkyUC8159

            return InkyUC8159()
        raise ValueError(f"Unsupported --inky-type: {inky_type!r}")

    from inky.auto import auto

    return auto()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required (install from requirements.txt)") from e

    t = t.detach().cpu().clamp(0, 1)[0]
    arr = (t.permute(1, 2, 0).contiguous().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _show_on_inky(
    img: Image.Image,
    *,
    inky_type: str,
    inky_colour: str,
    border: str,
    rotate_clockwise: int,
    saturation: float,
) -> None:
    try:
        display = _init_inky_display(inky_type=inky_type, inky_colour=inky_colour)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to initialise Inky display. If you're running on a Raspberry Pi, make sure the "
            "Pimoroni Inky library and its system deps are installed (see https://github.com/pimoroni/inky)."
        ) from e

    if hasattr(display, "resolution"):
        width, height = display.resolution
    else:
        width, height = int(display.width), int(display.height)

    img = ImageOps.fit(img.convert("RGB"), (width, height), method=Image.BICUBIC, centering=(0.5, 0.5))
    rot = int(rotate_clockwise) % 360
    if rot:
        img = img.rotate(-rot, expand=False)

    if hasattr(display, "set_border"):
        border_name = border.strip().upper()
        border_value = getattr(display, border_name, None)
        if border_value is not None:
            display.set_border(border_value)

    set_image = getattr(display, "set_image", None)
    if set_image is None:
        raise RuntimeError("Inky display driver does not provide set_image().")

    try:
        sig = inspect.signature(set_image)
        if "saturation" in sig.parameters:
            set_image(img, saturation=float(saturation))
        else:
            set_image(img)
    except TypeError:
        set_image(img)

    show = getattr(display, "show", None)
    if show is None:
        raise RuntimeError("Inky display driver does not provide show().")
    show()


def _find_vgg_weights() -> Path:
    env = os.environ.get("PI_CASSO_VGG_WEIGHTS")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    candidates = [
        _repo_root() / "vgg19_conv1_to_conv5.pth",
        Path.cwd() / "vgg19_conv1_to_conv5.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing VGG weights file. Place `vgg19_conv1_to_conv5.pth` in the repo root "
        "or set env var `PI_CASSO_VGG_WEIGHTS`."
    )


def _vgg_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - VGG_MEAN.to(device=x.device)) / VGG_STD.to(device=x.device)


def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    _, c, h, w = feat.shape
    f = feat.view(c, h * w)
    g = f @ f.t()
    return g / float(c * h * w)


def _tv_loss(img: torch.Tensor) -> torch.Tensor:
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def _load_image(path: str | Path, imsize: int, device: torch.device) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required (install from requirements.txt)") from e

    img = Image.open(path).convert("RGB").resize((imsize, imsize), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


def _save_image(t: torch.Tensor, path: str | Path) -> None:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required (install from requirements.txt)") from e

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = t.detach().cpu().clamp(0, 1)[0]
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _build_vgg(vgg_weights: str | Path, max_conv: int, device: torch.device, channels_last: bool) -> nn.Sequential:
    feats = build_vgg19_features_until_conv(max_conv=max_conv).to(device).eval()
    sd = torch.load(vgg_weights, map_location=device)
    feats.load_state_dict(sd, strict=True)
    for p in feats.parameters():
        p.requires_grad_(False)
    if channels_last and device.type == "cpu":
        feats = feats.to(memory_format=torch.channels_last)
    return feats


def _compute_targets(
    vgg: nn.Sequential,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_conv: int,
    style_convs: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    conv_ids = sorted(set([content_conv] + style_convs))
    with torch.no_grad():
        content_feats = extract_conv_features(vgg, _vgg_normalize(content_img), conv_ids)
        style_feats = extract_conv_features(vgg, _vgg_normalize(style_img), style_convs)

    content_target = content_feats[content_conv].detach()
    style_targets: Dict[int, torch.Tensor] = {cid: _gram_matrix(style_feats[cid]).detach() for cid in style_convs}
    return content_target, style_targets


def _optimize_adam(
    vgg: nn.Sequential,
    content_target: torch.Tensor,
    style_targets: Dict[int, torch.Tensor],
    init_img: torch.Tensor,
    content_conv: int,
    style_convs: List[int],
    steps: int,
    lr: float,
    style_weight: float,
    content_weight: float,
    tv_weight: float,
    print_every: int,
    save_every: int,
    run_dir: Optional[Path],
) -> torch.Tensor:
    x = nn.Parameter(init_img.clone().detach())
    opt = torch.optim.Adam([x], lr=lr)
    conv_ids = sorted(set([content_conv] + style_convs))

    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        feats = extract_conv_features(vgg, _vgg_normalize(x), conv_ids)
        c_loss = F.mse_loss(feats[content_conv], content_target)
        s_loss = torch.zeros((), device=x.device)
        for conv_id in style_convs:
            g = _gram_matrix(feats[conv_id])
            s_loss = s_loss + F.mse_loss(g, style_targets[conv_id])
        tv = _tv_loss(x) if tv_weight > 0 else torch.zeros((), device=x.device)
        loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        if print_every > 0 and step % print_every == 0:
            dt = round(time.time() - t0, 2)
            print(
                f"step={step}/{steps} content={float(c_loss.detach().cpu()):.4f} "
                f"style={float(s_loss.detach().cpu()):.4f} tv={float(tv.detach().cpu()):.4f} t={dt}s"
            )
        if run_dir is not None and save_every > 0 and step % save_every == 0:
            _save_image(x, run_dir / f"step_{step:04d}.jpg")

    return x.detach()


def main() -> None:
    args = build_parser().parse_args()

    # Fixed defaults for Raspberry Pi Zero 2
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    channels_last = True
    max_conv = 5
    content_conv = 4
    style_convs = [1, 2, 3, 4, 5]

    # Fast-ish defaults with good quality on Pi
    # Target: 512 final resolution with 200 total steps (256->512 pyramid).
    pyramid = [256, 512]
    steps_per_stage = [150, 50]
    lr_per_stage = [0.04, 0.02]
    style_weight = 100000.0
    content_weight = 1.0
    tv_weight = 0.0
    init_mode = "content"
    print_every = 25
    save_every = 0
    run_dir = None

    vgg_weights = _find_vgg_weights()
    vgg = _build_vgg(vgg_weights, max_conv=max_conv, device=device, channels_last=channels_last)

    prev_out: Optional[torch.Tensor] = None
    total_t0 = time.time()

    for stage_idx, size in enumerate(pyramid, start=1):
        stage_steps = int(steps_per_stage[stage_idx - 1])
        stage_lr = float(lr_per_stage[stage_idx - 1])
        print(f"[stage {stage_idx}/{len(pyramid)}] imsize={size} steps={stage_steps} lr={stage_lr}")
        content_img = _load_image(args.content, size, device=device)
        style_img = _load_image(args.style, size, device=device)

        if channels_last and device.type == "cpu":
            content_img = content_img.contiguous(memory_format=torch.channels_last)
            style_img = style_img.contiguous(memory_format=torch.channels_last)

        content_target, style_targets = _compute_targets(
            vgg=vgg,
            content_img=content_img,
            style_img=style_img,
            content_conv=content_conv,
            style_convs=style_convs,
        )

        if prev_out is None:
            if init_mode == "noise":
                init_img = torch.rand_like(content_img)
            else:
                init_img = content_img.clone()
        else:
            init_img = F.interpolate(prev_out, size=(size, size), mode="bilinear", align_corners=False)

        prev_out = _optimize_adam(
            vgg=vgg,
            content_target=content_target,
            style_targets=style_targets,
            init_img=init_img,
            content_conv=content_conv,
            style_convs=style_convs,
            steps=stage_steps,
            lr=stage_lr,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            print_every=print_every,
            save_every=save_every,
            run_dir=run_dir,
        )

    assert prev_out is not None
    _save_image(prev_out, args.out)
    dt = round(time.time() - total_t0, 2)
    print(f"Saved: {args.out}")
    print(f"Total time: {dt}s")
    if args.inky:
        _show_on_inky(
            _tensor_to_pil(prev_out),
            inky_type=str(args.inky_type),
            inky_colour=str(args.inky_colour),
            border=str(args.inky_border),
            rotate_clockwise=int(args.inky_rotate),
            saturation=float(args.inky_saturation),
        )
        print("Displayed on Inky.")


if __name__ == "__main__":
    main()
