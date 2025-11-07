# Text from Blurred Image (MNTSR + CRAFT + OCR)

End-to-end pipeline to recover text from **blurred/low-resolution images**:

1. detect text regions with **CRAFT**,
2. enhance/deblur with **MNTSR**,
3. extract text with **OCR** (e.g., EasyOCR).

---

## 📁 Repository Structure (flat)

```
.
├── README.md
├── appc.py                     # (optional app/runner if you use it)
├── craft.py                    # CRAFT model (U-net style head + VGG16-BN backbone)
├── craft_main.py               # Text detection & region cropping
├── feature_extractor.py        # MNTSR feature extractor (SRBs + attention hooks)
├── mntsr.py                    # MNTSR model (initial conv + extractor + upsampler)
├── sml.py                      # Self-supervised memory module
├── srb.py                      # Spatial refinement block(s)
├── test_mntsr_on_image.py      # Runs MNTSR on CRAFT outputs
├── upsambler.py                # PixelShuffle upsampler + reconstruction
└── vgg16_bn.py                 # VGG16-BN backbone slices for CRAFT
```

> Output folders are created at runtime:
>
> * `craft_result/` — detected regions & preview (`detected_text.png`)
> * `mntsr_result/` — deblurred/enhanced images (`enhanced_image.png`)

---

## 🔧 Installation

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn easyocr
```

(Python 3.9+ recommended.)

---

## 📦 Weights

* **CRAFT** weights (e.g., `craft_mlt_25k.pth`).
* **MNTSR** weights (e.g., `mntsr_trained.pth`).

Place them anywhere you like, then update the paths in the scripts:

* In `craft_main.py`, set `model_path = r"PATH\TO\craft_mlt_25k.pth"`.
* In `test_mntsr_on_image.py`, ensure it can find `mntsr_trained.pth` (variable `model_path` near the top).

---

## ▶️ Quick Start

### 0) Put an input image

Place `input_image.png` **next to the scripts** (repo root).
`craft_main.py` first looks for the image in the repo root; if not found it tries a parent folder.

### 1) Detect text with CRAFT

```bash
python craft_main.py
```

This will:

* load CRAFT,
* detect text regions,
* draw green boxes on a preview image,
* save crops into **`craft_result/`**,
* save a preview **`craft_result/detected_text.png`**.

> **Tip:** If no output appears, check the printed paths in the console and fix `model_path` / image path.

### 2) Enhance (deblur) text with MNTSR

```bash
python test_mntsr_on_image.py
```

This will:

* load MNTSR,
* read images from **`craft_result/`**,
* write enhanced images to **`mntsr_result/`**,
* save **`mntsr_result/enhanced_image.png`** (used for OCR).

### 3) OCR (example with EasyOCR)

```python
import easyocr
reader = easyocr.Reader(['en'])
text = reader.readtext('mntsr_result/enhanced_image.png', detail=0)
print(text)
```

---

## 🧠 What’s Inside

* **CRAFT (`craft.py`, `vgg16_bn.py`)**

  * VGG16-BN backbone slices.
  * U-shaped decoder with `double_conv` blocks.
  * Produces score maps; `craft_main.py` thresholds, finds contours, expands boxes, and **clusters boxes via DBSCAN** to form words/lines.

* **MNTSR (`mntsr.py`)**

  * Initial conv → **FeatureExtractor** (`feature_extractor.py`, SRBs + attention hooks)
  * **SelfSupervisedMemory** (`sml.py`) for memory-augmented refinement
  * **SuperResolutionUpsampler** (`upsambler.py`) with PixelShuffle → RGB reconstruction
  * `test_mntsr_on_image.py` batches CRAFT crops → saves deblurred outputs.

---

## ⚙️ Paths & Folder Notes (important)

* The current scripts create `craft_result/` and `mntsr_result/` **in the repo root** when run from the repo root.
* If you see paths pointing to a **parent** directory in prints, you can normalize to the repo root by editing:

**In `craft_main.py`:**

```python
# replace:
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# with:
parent_folder = os.path.dirname(os.path.abspath(__file__))  # repo root
```

**In `test_mntsr_on_image.py`:**

```python
# replace:
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# with:
BASE_DIR = SCRIPT_DIR  # keep outputs in repo root
```

That keeps everything (inputs/outputs) neatly inside the repository folder.

---

## 🔍 Tips

* Tune `eps` and `min_samples` in DBSCAN (inside `craft_main.py`) if words are over-/under-grouped.
* If your inputs are very small, increase resize in `craft_main.py` (e.g., 1280×720) before inference.
* MNTSR `upscale_factor` can be adjusted in `upsambler.py` / `mntsr.py` if you need larger outputs.

---

## 🤝 Contributing

PRs/issues welcome—ideas: CLI wrapper, batch processing, model zoo, dockerfile, multi-language OCR presets.

---

## 📜 License

MIT

---
