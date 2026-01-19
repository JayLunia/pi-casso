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
    st.caption("Content + style → output. Minimal UI for Raspberry Pi.")

    styles = _list_style_presets()

    # Fixed defaults (keep UI minimal)
    device_str = "cpu"
    device = torch.device(device_str)
    _configure_torch_threads(4, 1)
    channels_last = True
    content_conv = 4
    style_convs = [1, 2, 3, 4, 5]
    pyramid_sizes = [128, 256]
    stage_steps = [80, 30]
    stage_lrs = [0.04, 0.02]
    style_weight = 100000.0
    content_weight = 1.0
    tv_weight = 0.0
    init_mode = "content"
    preview_every = 0

    # Auto-detect weights file in repo root; allow upload only if missing
    vgg_default = str((_repo_root() / "vgg19_conv1_to_conv5.pth"))
    vgg_path = str(st.session_state.get(_VGG_WEIGHTS_STATE_KEY, vgg_default))
    if not Path(vgg_path).exists():
        st.warning("Missing `vgg19_conv1_to_conv5.pth` in repo root. Upload it to run.")
        vgg_upload = st.file_uploader("Upload `vgg19_conv1_to_conv5.pth`", type=["pth"])
        if vgg_upload is not None:
            tmp_dir = _repo_root() / ".streamlit_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / "vgg19_conv1_to_conv5.pth"
            tmp_path.write_bytes(vgg_upload.getvalue())
            st.session_state[_VGG_WEIGHTS_STATE_KEY] = str(tmp_path)
            vgg_path = str(tmp_path)

    run_number = st.number_input("Run number", min_value=1, value=1, step=1)

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
        run_button = st.button("Run", type="primary")

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

    with st.spinner("Loading VGG slice…"):
        vgg = _load_vgg(vgg_path, max_conv=5, device_str=device_str, channels_last=channels_last)

    prev_out: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = {}
    with st.spinner("Optimizing…"):
        for stage_idx, size in enumerate(pyramid_sizes, start=1):
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
    out_path = _repo_root() / "outputs" / f"run_{int(run_number):04d}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    st.caption(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
