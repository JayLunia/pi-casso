#!/usr/bin/env python3
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_casso.vgg19_features import build_vgg19_features_until_conv, extract_conv_features


VGG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
VGG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

_TORCH_THREADS_STATE_KEY = "_pi_casso_torch_threads"
_VGG_WEIGHTS_STATE_KEY = "_pi_casso_vgg_weights_path"


def _configure_torch_threads(num_threads: int, interop_threads: int) -> None:
    """Configure torch threads safely under Streamlit.

    Streamlit re-runs the script; PyTorch disallows changing inter-op threads
    after work has started. So we attempt configuration once per Streamlit
    session and require restart for changes.
    """
    desired = (int(num_threads), int(interop_threads))
    previous = st.session_state.get(_TORCH_THREADS_STATE_KEY)

    if previous is None:
        try:
            torch.set_num_threads(desired[0])
        except RuntimeError as e:
            st.warning(f"Could not set torch threads ({e}).")
        try:
            torch.set_num_interop_threads(desired[1])
        except RuntimeError as e:
            st.warning(f"Could not set torch interop threads ({e}).")
        st.session_state[_TORCH_THREADS_STATE_KEY] = desired
        return

    if previous != desired:
        st.info(
            "Torch thread settings can’t be changed after the app has started. "
            "Restart Streamlit to apply new values."
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _list_style_presets() -> List[Path]:
    styles_dir = _repo_root() / "styles"
    if not styles_dir.exists():
        return []
    return sorted([p for p in styles_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])


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


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required") from e
    t = t.detach().cpu().clamp(0, 1)[0]
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


@st.cache_data(show_spinner=False)
def _load_content_bytes(file_bytes: bytes, imsize: int) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img.resize((imsize, imsize), resample=Image.BICUBIC)

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


def _parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _expand_int(values: List[int], n: int, default: int) -> List[int]:
    if not values:
        return [int(default)] * n
    if len(values) == 1:
        return [int(values[0])] * n
    if len(values) != n:
        raise ValueError(f"Expected {n} values but got {len(values)}")
    return [int(v) for v in values]


def _expand_float(values: List[float], n: int, default: float) -> List[float]:
    if not values:
        return [float(default)] * n
    if len(values) == 1:
        return [float(values[0])] * n
    if len(values) != n:
        raise ValueError(f"Expected {n} values but got {len(values)}")
    return [float(v) for v in values]


def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required") from e
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


@st.cache_resource(show_spinner=False)
def _load_vgg(vgg_weights_path: str, max_conv: int, device_str: str, channels_last: bool) -> torch.nn.Sequential:
    device = torch.device(device_str)
    vgg = build_vgg19_features_until_conv(max_conv=max_conv).to(device).eval()
    sd = torch.load(vgg_weights_path, map_location=device)
    vgg.load_state_dict(sd, strict=True)
    for p in vgg.parameters():
        p.requires_grad_(False)
    if channels_last and device.type == "cpu":
        vgg = vgg.to(memory_format=torch.channels_last)
    return vgg


def _compute_targets(
    vgg: torch.nn.Sequential,
    content: torch.Tensor,
    style: torch.Tensor,
    content_conv: int,
    style_convs: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    conv_ids = sorted(set([content_conv] + style_convs))
    with torch.no_grad():
        content_feats = extract_conv_features(vgg, _vgg_normalize(content), conv_ids)
        style_feats = extract_conv_features(vgg, _vgg_normalize(style), style_convs)
    content_target = content_feats[content_conv].detach()
    style_targets = {cid: _gram_matrix(style_feats[cid]).detach() for cid in style_convs}
    return content_target, style_targets


def _run_adam(
    vgg: torch.nn.Sequential,
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
    preview_every: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = torch.nn.Parameter(init_img.clone().detach())
    opt = torch.optim.Adam([x], lr=lr)
    conv_ids = sorted(set([content_conv] + style_convs))

    t0 = time.time()
    preview_placeholder = st.empty()
    prog = st.progress(0, text="Optimizing…")
    last: Dict[str, float] = {"content": 0.0, "style": 0.0, "tv": 0.0, "total": 0.0}

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        feats = extract_conv_features(vgg, _vgg_normalize(x), conv_ids)
        c_loss = F.mse_loss(feats[content_conv], content_target)
        s_loss = torch.zeros((), device=x.device)
        for cid in style_convs:
            s_loss = s_loss + F.mse_loss(_gram_matrix(feats[cid]), style_targets[cid])
        tv = _tv_loss(x) if tv_weight > 0 else torch.zeros((), device=x.device)
        loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        prog.progress(step / steps, text=f"Optimizing… {step}/{steps}")
        if preview_every > 0 and (step == 1 or step % preview_every == 0 or step == steps):
            preview_placeholder.image(_tensor_to_pil(x), caption=f"Preview step {step}", use_container_width=True)

        last = {
            "content": float(c_loss.detach().cpu()),
            "style": float(s_loss.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }

    last["seconds"] = round(time.time() - t0, 3)
    return x.detach(), last


def main() -> None:
    st.set_page_config(page_title="Pi-casso (Camera NST)", layout="wide")
    st.title("Pi-casso — Neural Style Transfer on Raspberry Pi (No Training)")
    st.caption("Optimization-based NST (per-run). Uses a truncated VGG19 feature slice to fit in low RAM.")

    styles = _list_style_presets()

    with st.sidebar:
        st.header("Runtime")
        device_str = st.selectbox("Device", options=["cpu"], index=0)
        preset = st.selectbox("Quality preset", options=["Fast (Pi)", "Balanced", "High", "Custom"], index=1)
        if preset == "Fast (Pi)":
            imsize_default, steps_default, lr_default = 192, 120, 0.04
            pyramid_default, steps_stage_default, lr_stage_default = "128,192", "60,60", "0.05,0.03"
        elif preset == "Balanced":
            imsize_default, steps_default, lr_default = 256, 160, 0.03
            pyramid_default, steps_stage_default, lr_stage_default = "128,256", "80,30", "0.04,0.02"
        elif preset == "High":
            imsize_default, steps_default, lr_default = 256, 250, 0.03
            pyramid_default, steps_stage_default, lr_stage_default = "", "", ""
        else:
            imsize_default, steps_default, lr_default = 256, 200, 0.03
            pyramid_default, steps_stage_default, lr_stage_default = "", "", ""

        imsize = st.slider("Image size", min_value=64, max_value=512, value=int(imsize_default), step=64)
        steps = st.slider("Steps (Adam)", min_value=10, max_value=600, value=int(steps_default), step=10)
        lr = st.number_input("Learning rate", min_value=1e-4, max_value=0.5, value=float(lr_default), step=0.01, format="%.4f")
        st.caption("Optional multi-scale (often better quality/faster on Pi). Leave blank to disable.")
        pyramid = st.text_input("Pyramid sizes (e.g. 128,256)", value=pyramid_default)
        steps_per_stage = st.text_input("Steps per stage (e.g. 80,30)", value=steps_stage_default)
        lr_per_stage = st.text_input("LR per stage (e.g. 0.04,0.02)", value=lr_stage_default)

        style_weight = st.number_input("Style weight", min_value=1.0, value=100000.0, step=1000.0, format="%.0f")
        content_weight = st.number_input("Content weight", min_value=0.0, value=1.0, step=0.1, format="%.3f")
        tv_weight = st.number_input("TV weight (optional)", min_value=0.0, value=0.0, step=0.001, format="%.4f")
        preview_every = st.select_slider("Preview every N steps", options=[0, 5, 10, 25, 50], value=25)
        init_mode = st.selectbox("Init", options=["content", "noise"], index=0)
        channels_last = st.checkbox("channels_last", value=True)
        num_threads = st.number_input("torch threads", min_value=1, max_value=8, value=4, step=1)
        interop_threads = st.number_input("interop threads", min_value=1, max_value=4, value=1, step=1)

        st.divider()
        st.header("VGG Slice Weights")
        st.caption("Generate once on a dev machine, then copy to the Pi (or upload here).")
        vgg_default = str((_repo_root() / "vgg19_conv1_to_conv5.pth"))
        vgg_path = st.text_input(
            "vgg weights path (.pth)",
            value=str(st.session_state.get(_VGG_WEIGHTS_STATE_KEY, vgg_default)),
        )
        vgg_upload = st.file_uploader("…or upload vgg weights (.pth)", type=["pth"])
        if vgg_upload is not None:
            tmp_dir = _repo_root() / ".streamlit_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / "vgg19_conv1_to_conv5.pth"
            tmp_path.write_bytes(vgg_upload.getvalue())
            st.session_state[_VGG_WEIGHTS_STATE_KEY] = str(tmp_path)
            vgg_path = str(tmp_path)

        st.caption("Create weights: `python3 -m pi_casso.export_vgg19_features_weights --max-conv 5 --out vgg19_conv1_to_conv5.pth`")
        run_button = st.button("Run", type="primary")

    _configure_torch_threads(int(num_threads), int(interop_threads))
    device = torch.device(device_str)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Content (Camera)")
        cam = st.camera_input("Take a photo")
        st.caption("If camera isn’t available, upload an image.")
        upload = st.file_uploader("Upload content", type=["jpg", "jpeg", "png"])

    with col2:
        st.subheader("Style (Preset)")
        if not styles:
            st.warning("Add some style images to `styles/` (jpg/png).")
            style_path = None
        else:
            labels = [p.name for p in styles]
            chosen = st.selectbox("Choose style", options=labels, index=0)
            style_path = styles[labels.index(chosen)]
            st.image(str(style_path), caption=str(style_path), use_container_width=True)

    with col3:
        st.subheader("Output")
        out_placeholder = st.empty()
        dl_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if not run_button:
        return
    if style_path is None:
        st.error("Select a style preset first.")
        return
    if not Path(vgg_path).exists():
        st.error(f"Missing VGG weights file: {vgg_path}")
        return

    content_bytes = cam.getvalue() if cam is not None else (upload.getvalue() if upload is not None else None)
    if not content_bytes:
        st.error("Capture a photo or upload a content image.")
        return

    with st.spinner("Loading images…"):
        original_content = Image.open(io.BytesIO(content_bytes)).convert("RGB")
        original_style = Image.open(style_path).convert("RGB")

    try:
        pyramid_sizes = _parse_int_list(pyramid) if pyramid.strip() else [int(imsize)]
        pyramid_sizes = [s for s in pyramid_sizes if s > 0] or [int(imsize)]
        stage_steps = _expand_int(_parse_int_list(steps_per_stage), len(pyramid_sizes), int(steps))
        stage_lrs = _expand_float(_parse_float_list(lr_per_stage), len(pyramid_sizes), float(lr))
    except Exception as e:
        st.error(f"Invalid pyramid/stage settings: {e}")
        return

    with st.spinner("Loading VGG slice…"):
        vgg = _load_vgg(vgg_path, max_conv=5, device_str=device_str, channels_last=channels_last)

    prev_out: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = {}
    with st.spinner("Optimizing…"):
        for stage_idx, size in enumerate(pyramid_sizes, start=1):
            st.info(f"Stage {stage_idx}/{len(pyramid_sizes)}: {size}px  steps={stage_steps[stage_idx-1]}  lr={stage_lrs[stage_idx-1]}")
            content_pil = original_content.resize((size, size), resample=Image.BICUBIC)
            style_pil = original_style.resize((size, size), resample=Image.BICUBIC)
            content = _pil_to_tensor(content_pil, device=device)
            style = _pil_to_tensor(style_pil, device=device)
            if channels_last and device.type == "cpu":
                content = content.contiguous(memory_format=torch.channels_last)
                style = style.contiguous(memory_format=torch.channels_last)

            content_target, style_targets = _compute_targets(
                vgg, content, style, content_conv=4, style_convs=[1, 2, 3, 4, 5]
            )

            if prev_out is None:
                init_img = torch.rand_like(content) if init_mode == "noise" else content.clone()
            else:
                init_img = F.interpolate(prev_out, size=(size, size), mode="bilinear", align_corners=False)
            if channels_last and device.type == "cpu":
                init_img = init_img.contiguous(memory_format=torch.channels_last)

            prev_out, metrics = _run_adam(
                vgg=vgg,
                content_target=content_target,
                style_targets=style_targets,
                init_img=init_img,
                content_conv=4,
                style_convs=[1, 2, 3, 4, 5],
                steps=int(stage_steps[stage_idx - 1]),
                lr=float(stage_lrs[stage_idx - 1]),
                style_weight=float(style_weight),
                content_weight=float(content_weight),
                tv_weight=float(tv_weight),
                preview_every=int(preview_every),
            )

    assert prev_out is not None
    out = prev_out

    out_img = _tensor_to_pil(out)
    out_placeholder.image(out_img, caption="Final output", use_container_width=True)
    metrics_placeholder.json(metrics)

    buf = io.BytesIO()
    out_img.save(buf, format="JPEG")
    dl_placeholder.download_button("Download output", data=buf.getvalue(), file_name="stylized.jpg", mime="image/jpeg")


if __name__ == "__main__":
    main()
