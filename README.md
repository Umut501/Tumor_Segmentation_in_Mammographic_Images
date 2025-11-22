# Mammogram Tumor Segmentation

This repository contains a Python implementation of a tumor segmentation algorithm. The script processes images in `dataset/` and writes outputs to `results/`.

Quick start

1. Create a virtual environment and activate it (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python main.py
```

Outputs

- `results/<image>_seg.png` : binary segmentation mask (white = tumor)
- `results/<image>_breast.png` : breast image after background removal and pectoral removal
- `results/<image>_overlay.png` : original image overlayed with segmentation (red) and ground truth circle (green)

Notes

- The code reads `dataset/Info.txt` and uses the (x,y,r) entries to build a circular ground-truth. The y-coordinate in the file is inverted as the assignment specified (subtract from 1024) when image height >= 1024.
- The algorithm follows the steps you described: thresholding for breast, mask multiply, threshold & remove pectoral muscle, erosion to remove vessels, then region growing seeded from remaining pixels.
- Tuning: disk radius for erosion (`disk(12)`) and region growing tolerance (`tol=12`) may need tuning per-image for best performance.
