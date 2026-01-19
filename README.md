# Pi-casso

Neural Style Transfer on Raspberry Pi **without training** — you run the optimization every time for a new content + style image (same workflow as the classic NST tutorial), but packaged to be **lighter on 512MB devices**.

## Why this works on a Pi Zero 2

Classic NST uses VGG19 activations as perceptual losses. Many implementations accidentally load the full VGG19 model including the **classifier** (hundreds of MB) even though they only use `features`.

Pi-casso avoids that by:

1. Exporting a **truncated** VGG19 `features` slice (by default up to `conv_5`)
2. Loading only that small slice on the Pi (no classifier weights)

The exported weights file is ~2 MB for `conv_1..conv_5`.

## Folder Layout

- `pi_casso/run_opt.py` — CLI NST runner (single style)
- `pi_casso/streamlit_app.py` — Streamlit UI (camera + preset styles)
- `pi_casso/export_vgg19_features_weights.py` — export the VGG slice weights (run on your dev machine)
- `styles/` — put your preset style images here (jpg/png)
- `outputs/` — output images (ignored by git)

## Requirements

- Raspberry Pi Zero 2 (512MB) or similar
- Python 3.9+
- A Pi-compatible **PyTorch** install (CPU build)

This repo intentionally does **not** pin `torch` because the correct wheel depends on your OS/arch.

## Install (Pi)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then install PyTorch using whatever build you use for your Pi OS/arch.

## Step 1 — Export VGG slice weights (run on your dev machine)

You need `torchvision` only for this export step.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision

python3 -m pi_casso.export_vgg19_features_weights \
  --max-conv 5 \
  --out vgg19_conv1_to_conv5.pth
```

Copy `vgg19_conv1_to_conv5.pth` to the Pi-casso repo root on the Pi.

## Step 2 — Add some preset styles

Put `.jpg/.png` files in:

```
styles/
```

## Run (CLI)

```bash
python3 -m pi_casso.run_opt \
  --vgg-weights vgg19_conv1_to_conv5.pth \
  --content path/to/content.jpg \
  --style styles/your_style.jpg \
  --out outputs/out.jpg \
  --imsize 256 \
  --steps 200 \
  --num-threads 4 \
  --interop-threads 1 \
  --channels-last
```

Quality/speed tips:
- Big speedups: lower `--imsize` to `128` or `192`
- Better quality for similar time: use multi-scale, e.g. `--pyramid 128,256 --steps-per-stage 80,30 --lr-per-stage 0.04,0.02`
- If results look noisy: try `--init content` (default) and lower `--lr` (e.g. `0.02`)

## Run (Streamlit UI)

Install UI deps:

```bash
pip install -r requirements-streamlit.txt
```

Start:

```bash
streamlit run pi_casso/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

The UI:
- captures content via `camera_input` (or file upload fallback)
- lets you pick a style from `styles/`
- runs NST and shows previews + final image + download button
- includes a “Quality preset” and optional multi-scale pyramid controls to reduce runtime on Pi

## Troubleshooting

- **“Missing VGG weights file”**: you must copy `vgg19_conv1_to_conv5.pth` to the repo root (or point the UI to the correct path).
- **Killed / OOM**: reduce `--imsize`, reduce `--steps`, enable swap, close other apps.
- **Very slow**: Pi Zero 2 is CPU-only; use smaller resolutions and fewer steps. Multi-scale often helps quality at lower steps.

## Creating a private GitHub repo

This project is prepared as a local git repo. To push to a private GitHub repository:

```bash
git remote add origin git@github.com:<your-user-or-org>/pi-casso.git
git push -u origin main
```
