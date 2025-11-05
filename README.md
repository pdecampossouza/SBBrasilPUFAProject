# PUFA Toolkit — Labeling, Validation & Baseline Models

Tools to build a PUFA dataset (0/P/U/F/A), validate it with provenance, and train baseline models (classical ML and CNN). Designed to run locally on Windows/macOS/Linux.

## What’s inside

```
.
├── Pranchetas fotografias - PUFA - treinamento/   # Original full-mouth photos (input)
├── images_by_tooth/PUFA/                           # Tooth crops (generated)
├── annotations/                                    # JSON + validation CSV (generated)
├── metadata/                                       # Manifests + reports + plots (generated)
├── ClassificacaoIsabel/                            # A second dataset (same layout as above)
│   ├── annotations/tooth_labels_free_draw.json
│   ├── images_by_tooth/PUFA/
│   └── metadata/manifest_teeth.csv
├── example/                                        # Small sample files (optional)
├── PlotNeuralNet/                                  # (Optional) LaTeX network diagram helper
│
├── pufa_free_draw_labeler.py        # Rectangle-based labeling (Streamlit)
├── pufa_free_draw_labeler2.py       # Variant with prev/next UX
├── pufa_grid_labeler.py             # (Older) grid-labeler proof-of-concept
├── pufa_validation_tool.py          # Accept/Reject validator with provenance (Streamlit)
│
├── pufa_sanity_report.py            # Quick checks on manifest/crops
├── filter_manifest_isabel.py        # Helper to point to Isabel’s dataset
├── fix_manifest_isabel.py           # Fixes/normalizes paths/columns for Isabel’s manifest
│
├── pufa_train_baseline.py           # HOG+HSV + PCA + LinearSVC (multiclass)
├── pufa_visual_predict.py           # Sliding windows over originals (blue=good, yellow=bad)
├── pufa_visual_predict_cnn_shap.py  # Same idea but with CNN + SHAP summary
├── pufa_train_cnn.py                # Light CNN (from scratch) + k-fold by volunteer
├── pufa_train_deep.py               # EfficientNetB0/Custom CNN (transfer or scratch)
│
├── cnn_fold{1..5}.keras             # Saved CNN fold weights (generated)
├── cnn_fold{1..5}_{acc,loss}.png    # Per-fold training curves (generated)
├── cnn_model.tex                    # LaTeX diagram (optional)
└── render_tex.py                    # Helper to render LaTeX net diagram (optional)
```

> **Tip:** keep raw patient images private. Commit only code, small example files, and synthetic outputs.

---

## Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install streamlit opencv-python pillow numpy pandas scikit-learn shap matplotlib tensorflow==2.15.0
# macOS/Linux
source .venv/bin/activate
pip install -U pip
pip install streamlit opencv-python pillow numpy pandas scikit-learn shap matplotlib tensorflow==2.15.0
```

> CPU-only TF is fine; GPUs will speed up `pufa_train_cnn.py` / `pufa_train_deep.py`.

---

## 1) Labeling (free-draw rectangles)

**Goal:** draw rectangles over teeth and assign PUFA labels.  
**Output:** crops in `images_by_tooth/PUFA/` and a manifest at `metadata/manifest_teeth.csv`.

Run:

```bash
python -m streamlit run pufa_free_draw_labeler.py
```

Key points:

- Works over `Pranchetas fotografias - PUFA - treinamento/` by default.
- **Rater ID** is stored with each box for provenance.
- Clicking **Export crops & update manifest** writes:
  - crops → `images_by_tooth/PUFA/`
  - rows → `metadata/manifest_teeth.csv` with columns  
    `img_path, crop_path, x1..y2, pufa_label, rater[, width, height, tooth_idx/id]`

### Isabel’s dataset (same tool)

If you want the tool to **save into** the *Isabel* tree instead of your default:

- Use `filter_manifest_isabel.py` and/or `fix_manifest_isabel.py` to normalize paths,
- Or simply run the validator (next section) pointing to  
  `ClassificacaoIsabel/metadata/manifest_teeth.csv` (recommended for validation).

---

## 2) Validation (Accept/Reject with provenance)

**Goal:** second expert checks each crop and Accepts/Rejects (optionally corrects label).  
**Output:** `annotations/validation_votes.csv` and `metadata/manifest_teeth_validated.csv`.

Run:

```bash
python -m streamlit run pufa_validation_tool.py
```

At the top, choose which manifest to use:

- Your local: `metadata/manifest_teeth.csv`
- Isabel’s: `ClassificacaoIsabel/metadata/manifest_teeth.csv`

Then:

- Fill **Validator ID** (required).
- Use filters (class, original rater, filename substring).
- For each tile: **Decision** (accept/reject), optional **Corrected label**, **Comment**.
- **Save progress** stores/updates `annotations/validation_votes.csv` keyed by `(id, validator)`.
- **Export validated CSV** builds `metadata/manifest_teeth_validated.csv` using:
  - `final_label = corrected_label if present else original pufa_label`
  - provenance columns: `original_rater`, `validator`, `validated_at`, `id`.

---

## 3) Sanity checks

Run fast integrity checks before training:

```bash
python pufa_sanity_report.py
```

Typical checks: missing files, bad coords, per-class counts, per-volunteer counts.

---

## 4) Baseline classical ML (HOG+HSV → PCA → LinearSVC)

```bash
python pufa_train_baseline.py
```

- Reads `metadata/manifest_teeth.csv` (or point to Isabel’s).
- Extracts HOG + HSV histogram per crop; performs stratified CV (k adapted to smallest class).
- Outputs classification report and `metadata/pufa_confusion.png`.

---

## 5) CNN training (from scratch)

```bash
# Multiclass 0/P/U/F/A using validated manifest if you want:
python pufa_train_cnn.py --manifest metadata/manifest_teeth_validated.csv --epochs 15 --batch_size 32
```

- Volunteer-stratified k-fold to avoid leakage (`Voluntario*` pulled from `img_path`).
- Saves per-fold weights `cnn_fold{k}.keras` and curves `cnn_fold{k}_{acc,loss}.png`.

### Model used (default)

```python
Conv2D(32, 3, activation="relu") → MaxPool(2)
Conv2D(64, 3, activation="relu") → MaxPool(2)
Conv2D(256, 3, activation="relu") → MaxPool(2)
Flatten → Dropout(0.2) → Dense(256,"relu") → Dense(num_classes,"softmax")
```

> You can switch to transfer learning using `pufa_train_deep.py` (EfficientNetB0 or a custom 224×224 CNN). For small datasets, **freezing the base and training only the head** is recommended, then unfreezing a few top blocks for fine-tuning.

---

## 6) Visual “blue/yellow” predictions over full images

```bash
python pufa_visual_predict.py
```

- Trains a **quick binary** detector (healthy vs lesion) from your manifest.
- Slides a 128×128 window (stride 64) over each original image and draws:
  - **Blue** semi-transparent tiles for “good/healthy” (pred=0),
  - **Yellow** contours for “bad/lesion” (pred=1).
- Outputs to `metadata/visual_predictions/*.png`.

SHAP version (with PCA+LinearSVC):  
`python pufa_visual_predict_cnn_shap.py` → saves `pufa_shap_summary.png` and explained variance.

---

## File-by-file cheat-sheet

| File / Folder | What it does |
|---|---|
| **pufa_free_draw_labeler.py** | Main labeling UI (draw rectangles, pick PUFA, export crops + manifest). Stores rater. |
| **pufa_free_draw_labeler2.py** | Same engine with prev/next navigation and minor UX tweaks. |
| **pufa_grid_labeler.py** | Earlier “strip grid” prototype; kept for reference. |
| **pufa_validation_tool.py** | Accept/Reject validator with **provenance**. Exports `validation_votes.csv` + `manifest_teeth_validated.csv`. |
| **pufa_sanity_report.py** | Counts, path checks, coordinate bounds, per-volunteer distribution. |
| **pufa_train_baseline.py** | HOG+HSV features → PCA → LinearSVC (multiclass). Auto-adapts folds by class size. |
| **pufa_train_cnn.py** | Small CNN from scratch; **Group k-fold by volunteer**; saves fold weights + curves. |
| **pufa_train_deep.py** | EfficientNetB0 (or custom 224×224) transfer learning; also supports binary (`--binary`). |
| **pufa_visual_predict.py** | Sliding window visualization over original photos (blue=good, yellow=bad). |
| **pufa_visual_predict_cnn_shap.py** | Adds PCA explained variance + SHAP summary for interpretability. |
| **filter_manifest_isabel.py** | Points scripts to use `ClassificacaoIsabel/metadata/manifest_teeth.csv`. |
| **fix_manifest_isabel.py** | Normalizes paths/columns inside Isabel’s manifest (Windows/macOS safe). |
| **cnn_model.tex / render_tex.py / PlotNeuralNet/** | Optional LaTeX pipeline to draw architecture diagrams. |
| **annotations/** | `tooth_labels_free_draw.json` (labeler) and `validation_votes.csv` (validator). |
| **metadata/** | Manifests and outputs (confusion matrices, SHAP plots, PCA variance, visual predictions). |
| **ClassificacaoIsabel/** | A second dataset tree; same structure as the main one. |
| **images_by_tooth/PUFA/** | Crops generated by the labeler (input to training). |
| **Pranchetas fotografias - PUFA - treinamento/** | Original photos (input to labeler & visual predictor). |

**Manifest columns (typical):**

- `img_path` (original image), `crop_path` (tooth crop), `x1 y1 x2 y2` (bbox),
- `pufa_label` in `{0,P,U,F,A}`,
- `rater` (original rater), `width`/`height` (optional),
- `id` (stable hash, created by the validator),
- `original_rater`, `validator`, `validated_at` (in validated manifest),
- `final_label` (in validated manifest; corrected when provided).

---

## ⚙️ Reproducible runs

1. **Label some teeth** with `pufa_free_draw_labeler.py` → export crops.
2. **Sanity check** with `pufa_sanity_report.py`.
3. **(Optional) Validate** with `pufa_validation_tool.py` → export validated CSV.
4. **Train**:
   - Classical ML: `python pufa_train_baseline.py`
   - CNN: `python pufa_train_cnn.py --manifest metadata/manifest_teeth_validated.csv`
5. **Visualize**: `python pufa_visual_predict.py`.

---

## Troubleshooting

- **No items in validator grid:** Check the manifest path shown at top. It must exist and `crop_path` files must be present. Use the Isabel version by pointing to `ClassificacaoIsabel/metadata/manifest_teeth.csv`.
- **Windows backslashes:** scripts normalize, but if a manifest was edited by hand, run `fix_manifest_isabel.py`.
- **EfficientNet weight shape mismatch:** happens if you accidentally feed 1-channel images into a 3-channel backbone. This repo’s loaders force 3 channels; if you customize, keep `input_shape=(H,W,3)`.

---

## 📊 Suggested `.gitignore`

```
# data
Pranchetas fotografias - PUFA - treinamento/
images_by_tooth/
ClassificacaoIsabel/images_by_tooth/
metadata/*.csv
annotations/*.json
annotations/validation_votes*.csv

# models & runs
*.keras
*_acc.png
*_loss.png
metadata/*.png

# env
.venv/
__pycache__/
.DS_Store
```

---

## 📘 Citation

If you use this repository, please cite:

```bibtex
@article{souza2025oralhealth,
  author    = {Souza, Paulo Vitor de Campos and others},
  title     = {From PDF to Dental View Classification: A Human-in-the-Loop Dataset and Pipeline for Oral Health Imaging},
  journal   = {IEEE Journal of Translational Engineering in Health and Medicine},
  year      = {2025},
  note      = {Under Review}
}
```

---

## 🙏 Acknowledgment

This work was supported by national funds through the **Fundação para a Ciência e a Tecnologia (FCT)**, under project **UIDB/04152 – Centro de Investigação em Gestão de Informação (MagIC)**, NOVA Information Management School (NOVA IMS), Universidade Nova de Lisboa, Portugal.  

We also acknowledge the **Brazilian Ministry of Health** for providing access to calibration materials and public documentation from the *SB Brasil 2023* National Oral Health Survey, which made this research possible.

---

## ⚖️ License

This repository is released under the **MIT License**.  
You are free to use, modify, and distribute this code with proper citation of the original work.

---

## 📬 Contact

**Paulo Vitor de Campos Souza**  
NOVA Information Management School (NOVA IMS)  
Email: [psouza@novaims.unl.pt](mailto:psouza@novaims.unl.pt)

---

> ✳️ *“Bridging public health and computer vision for interpretable oral-health AI.”*

