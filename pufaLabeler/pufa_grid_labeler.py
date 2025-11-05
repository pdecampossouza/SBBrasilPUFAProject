# -*- coding: utf-8 -*-
# Streamlit grid labeler for PUFA (image-first layout, paths anchored to file)

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Paths (anchored to this file) ----------------
BASE_DIR = Path(__file__).resolve().parent

IMG_DIR = BASE_DIR / "Pranchetas fotografias - PUFA - treinamento"
ANN_PATH = BASE_DIR / "annotations" / "tooth_labels.json"
CROP_DIR = BASE_DIR / "images_by_tooth" / "PUFA"
TEETH_MANIFEST = BASE_DIR / "metadata" / "manifest_teeth.csv"

PUFA_CLASSES = [
    "0",
    "P",
    "U",
    "F",
    "A",
]  # 0=None, Pulp involvement, Ulceration, Fistula, Abscess


# ---------------- Helpers ----------------
def load_images() -> List[Path]:
    if not IMG_DIR.exists():
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    imgs: List[Path] = []
    for pat in exts:
        imgs.extend(IMG_DIR.glob(pat))
    return sorted(imgs)


def load_annotations() -> dict:
    if ANN_PATH.exists():
        return json.loads(ANN_PATH.read_text(encoding="utf-8"))
    return {}


def save_annotations(ann: dict) -> None:
    ANN_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANN_PATH.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")


def draw_preview(
    im: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    active: int,
    show_idx: bool = True,
) -> Image.Image:
    """Desenha o grid sobre a imagem, destacando a faixa ativa."""
    arr = np.array(im).copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (60, 120, 220) if i == active else (50, 180, 50)
        thickness = 3 if i == active else 2
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, thickness)
        if show_idx:
            cv2.putText(
                arr,
                f"#{i}",
                (x1 + 5, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return Image.fromarray(arr)


def propose_grid(
    w: int, h: int, n_cols: int, top_ratio: float, bot_ratio: float, margin_x: int
) -> List[Tuple[int, int, int, int]]:
    """
    Gera N faixas verticais cobrindo a altura [top_ratio, bot_ratio] da imagem.
    margin_x: recua um pouco cada faixa para não 'vazar' para fora da boca.
    """
    y1 = int(h * top_ratio)
    y2 = int(h * bot_ratio)
    boxes = []
    for i in range(n_cols):
        x_left = int(i * w / n_cols) + margin_x
        x_right = int((i + 1) * w / n_cols) - margin_x
        x_left = max(0, min(x_left, w - 1))
        x_right = max(0, min(x_right, w - 1))
        boxes.append((x_left, y1, x_right, y2))
    return boxes


def crop_and_save(img_path: Path, records: List[dict]) -> List[dict]:
    """Recorta e salva crops das faixas marcadas como 'is_tooth'."""
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    out_rows = []
    for r in records:
        if not r["is_tooth"]:
            continue
        x1, y1, x2, y2 = r["bbox"]
        crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        out_name = f"{img_path.stem}_tooth{r['tooth_idx']}_{x1}_{y1}_{x2}_{y2}.jpg"
        out_path = CROP_DIR / out_name
        cv2.imwrite(str(out_path), crop)
        out_rows.append(
            {
                "img_path": str(img_path),
                "tooth_idx": r["tooth_idx"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "pufa_label": r["label"],
                "crop_path": str(out_path),
                "width": w,
                "height": h,
            }
        )
    return out_rows


def upsert_manifest_teeth(rows: List[dict]) -> None:
    TEETH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if TEETH_MANIFEST.exists():
        base = pd.read_csv(TEETH_MANIFEST)
        df = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["crop_path"], inplace=True)
    df.to_csv(TEETH_MANIFEST, index=False)


# ---------------- App ----------------
def main():
    st.set_page_config(page_title="PUFA Grid Labeler (MVP)", layout="wide")

    # How to use
    st.info(
        "**How to use**\n\n"
        "1) Pick an image at the top (Prev/Next or dropdown).\n"
        "2) Make sure the *green stripes* cover the teeth area. If needed, open **Grid settings** (bottom) and tweak "
        "*Number of stripes*, *Top/Bottom coverage*, and *Horizontal margin*.\n"
        "3) For each stripe, tick **is tooth?** when it actually contains a tooth and select the **PUFA label**.\n"
        "4) Click **Save annotations & export crops**. Crops go to `images_by_tooth/PUFA/` and a manifest is saved at "
        "`metadata/manifest_teeth.csv`.\n",
        icon="ℹ️",
    )

    # Optional quick debug of paths
    with st.expander("Paths (debug)", expanded=False):
        st.write("BASE_DIR:", str(BASE_DIR))
        st.write("IMG_DIR:", str(IMG_DIR))
        st.write("ANN_PATH:", str(ANN_PATH))
        st.write("CROP_DIR:", str(CROP_DIR))
        st.write("TEETH_MANIFEST:", str(TEETH_MANIFEST))

    images = load_images()
    if not images:
        st.error(f"No images found in:\n`{IMG_DIR}`")
        st.stop()

    # Navigation
    cols_nav = st.columns([1, 3, 1])
    with cols_nav[0]:
        prev_btn = st.button("← Prev", use_container_width=True)
    with cols_nav[1]:
        img_name = st.selectbox("Image", [p.name for p in images], index=0)
    with cols_nav[2]:
        next_btn = st.button("Next →", use_container_width=True)

    key_list = [p.name for p in images]
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = key_list.index(img_name)

    # sync dropdown -> state
    if key_list[st.session_state.img_idx] != img_name:
        st.session_state.img_idx = key_list.index(img_name)

    if prev_btn:
        st.session_state.img_idx = (st.session_state.img_idx - 1) % len(key_list)
    if next_btn:
        st.session_state.img_idx = (st.session_state.img_idx + 1) % len(key_list)

    img_name = key_list[st.session_state.img_idx]
    img_path = next(p for p in images if p.name == img_name)

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    st.caption(f"Resolution: {w}×{h}px")

    # Session defaults for grid + preview
    ss = st.session_state
    ss.setdefault("n_cols", 10)
    ss.setdefault("top_ratio", 0.45)
    ss.setdefault("bot_ratio", 0.95)
    ss.setdefault("margin_x", max(1, int(w * 0.01)))
    ss.setdefault("active_idx", 0)
    ss.setdefault("show_idx", True)

    # Compute boxes
    boxes = propose_grid(w, h, ss.n_cols, ss.top_ratio, ss.bot_ratio, ss.margin_x)

    # -------- IMAGE FIRST --------
    st.subheader(f"Preview — {img_name}")
    st.image(
        draw_preview(im, boxes, int(ss.active_idx), ss.show_idx),
        use_container_width=True,
    )

    # Controls below: small preview/index on the right, labels on the left
    col_labels, col_small = st.columns([2, 1], gap="large")

    with col_small:
        st.markdown("**Preview stripe index**")
        ss.active_idx = st.number_input(
            "index",
            0,
            ss.n_cols - 1,
            int(ss.active_idx),
            1,
            label_visibility="collapsed",
        )
        ss.show_idx = st.toggle("Show stripe numbers", value=ss.show_idx)

    with col_labels:
        st.subheader("Labels")
        ann = load_annotations()
        key = img_path.name

        # init state respecting existing annotations
        if key in ann:
            stored = ann[key]
            is_tooth = [False] * ss.n_cols
            labels = ["0"] * ss.n_cols
            for r in stored:
                idx = r.get("tooth_idx", 0)
                if 0 <= idx < ss.n_cols:
                    is_tooth[idx] = bool(r.get("is_tooth", True))
                    labels[idx] = r.get("label", "0")
        else:
            is_tooth = [False] * ss.n_cols
            labels = ["0"] * ss.n_cols

        rows_out = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            st.write(f"Stripe #{i}")
            c1, c2 = st.columns([1, 1])
            with c1:
                is_tooth[i] = st.checkbox(
                    "is tooth?", value=is_tooth[i], key=f"tooth_{i}"
                )
            with c2:
                labels[i] = st.selectbox(
                    "PUFA",
                    PUFA_CLASSES,
                    index=PUFA_CLASSES.index(labels[i]),
                    key=f"lab_{i}",
                )
            rows_out.append(
                {
                    "tooth_idx": i,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "is_tooth": bool(is_tooth[i]),
                    "label": labels[i],
                }
            )
            st.divider()

        if not any(is_tooth):
            st.warning(
                "No stripe is marked as tooth. Tick **is tooth?** for at least one stripe.",
                icon="⚠️",
            )

        if st.button("Save annotations & export crops", type="primary"):
            ann[key] = rows_out
            save_annotations(ann)
            exported = crop_and_save(img_path, rows_out)
            if exported:
                upsert_manifest_teeth(exported)
            st.success(f"Saved! Crops exported: {len(exported)}")

    # Grid settings moved to bottom expander
    with st.expander("Grid settings", expanded=False):
        ss.n_cols = st.slider(
            "Number of vertical stripes (teeth slots)", 6, 16, ss.n_cols, 1
        )
        ss.top_ratio = st.slider(
            "Top coverage (ratio)", 0.0, 0.9, float(ss.top_ratio), 0.01
        )
        ss.bot_ratio = st.slider(
            "Bottom coverage (ratio)", 0.1, 1.0, float(ss.bot_ratio), 0.01
        )
        ss.margin_x = st.slider(
            "Horizontal margin inside each stripe (px)",
            0,
            int(w * 0.05),
            int(ss.margin_x),
            1,
        )

    st.markdown("---")
    st.caption(
        "MVP: mark stripes that actually contain teeth with **is tooth?** and assign the PUFA code. "
        "Click **Save** to write JSON, export crops and update `metadata/manifest_teeth.csv`."
    )


if __name__ == "__main__":
    main()
