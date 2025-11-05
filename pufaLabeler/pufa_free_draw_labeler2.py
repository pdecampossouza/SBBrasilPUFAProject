# -*- coding: utf-8 -*-
# PUFA Free-draw Labeler (stable, selectbox-only, scale-correct)
# Run:
#   python -m streamlit run ProjetoSBBrasilPUFA/pufa_free_draw_labeler.py

from pathlib import Path
import json
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st.stop()

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "Pranchetas fotografias - PUFA - treinamento"
ANN_JSON = BASE / "annotations" / "tooth_boxes.json"  # free-draw boxes
CROP_DIR = BASE / "images_by_tooth" / "PUFA"  # crops
TEETH_MANIFEST = BASE / "metadata" / "manifest_teeth.csv"  # all crops
CROP_DIR.mkdir(parents=True, exist_ok=True)

PUFA_LABELS = [
    ("0", "None (no clinical consequence)"),
    ("P", "Pulpal involvement"),
    ("U", "Ulceration"),
    ("F", "Fistula"),
    ("A", "Abscess"),
]
LABEL2DESC = dict(PUFA_LABELS)


# ---------- IO helpers ----------
def list_images() -> List[Path]:
    if not IMG_DIR.exists():
        return []
    return sorted(
        [*IMG_DIR.glob("*.jpg"), *IMG_DIR.glob("*.jpeg"), *IMG_DIR.glob("*.png")]
    )


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_manifest(rows: list) -> None:
    TEETH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if TEETH_MANIFEST.exists():
        df = pd.read_csv(TEETH_MANIFEST)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.drop_duplicates(subset=["crop_path"], inplace=True)
    df.to_csv(TEETH_MANIFEST, index=False)


def global_count() -> int:
    if TEETH_MANIFEST.exists():
        try:
            return int(pd.read_csv(TEETH_MANIFEST)["crop_path"].nunique())
        except Exception:
            return 0
    return 0


# ---------- Geometry ----------
def rect_from_points(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[int, int, int, int]:
    x1, y1 = p1
    x2, y2 = p2
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return int(xa), int(ya), int(xb), int(yb)


def scale_rect(rect, sx, sy):
    x1, y1, x2, y2 = rect
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


# ---------- App ----------
def main():
    st.set_page_config(page_title="PUFA Free-draw Labeler", layout="wide")

    # Header with totals and rater
    total = global_count()
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"### PUFA Free-draw Labeler  ·  **Saved teeth:** {total}")
        st.caption(
            "Draw a rectangle around a tooth, assign the PUFA code, then press **Save**."
        )
    with col2:
        rater = st.text_input(
            "Rater ID (required)", value="", help="Your name or code for provenance."
        )

    # Safety
    images = list_images()
    if not images:
        st.error(f"No images found in `{IMG_DIR}`.")
        st.stop()

    # Image selector
    img_name = st.selectbox("Image", [p.name for p in images], index=0)
    img_path = next(p for p in images if p.name == img_name)

    # Load image and compute canvas size (keep within width)
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    max_w = 980  # canvas display width
    canvas_w = min(W, max_w)
    canvas_h = int(H * (canvas_w / W))

    # Show canvas
    st.subheader(f"{img_name} · {W}×{H}px")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="#1AE069",
        background_image=pil.resize((canvas_w, canvas_h)),
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key="canvas_rect",
    )

    # Annotation store
    store = load_json(ANN_JSON)
    if img_name not in store:
        store[img_name] = []  # list of {x1,y1,x2,y2,label,rater}

    # Label chooser with descriptions
    st.markdown("**Assign PUFA label to the current selection**")
    code = st.selectbox(
        "PUFA code",
        [k for k, _ in PUFA_LABELS],
        format_func=lambda k: f"{k} — {LABEL2DESC[k]}",
        index=0,
    )

    # Buttons row
    cA, cB, cC = st.columns(3)
    with cA:
        add_btn = st.button(
            "Add selection",
            type="primary",
            use_container_width=True,
            help="Use the mouse to draw a rectangle; then click to add it.",
        )
    with cB:
        save_btn = st.button(
            "Save all & export crops",
            use_container_width=True,
            help="Write annotations JSON and export tooth crops.",
        )
    with cC:
        clear_btn = st.button(
            "Clear current image annotations", use_container_width=True
        )

    # Add selection
    if add_btn:
        if not rater.strip():
            st.warning("Please fill **Rater ID** before adding.", icon="⚠️")
        elif canvas.json_data and canvas.json_data.get("objects"):
            obj = canvas.json_data["objects"][-1]  # último retângulo
            # Coordenadas no canvas (considere escala)
            x = float(obj.get("left", 0))
            y = float(obj.get("top", 0))
            w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
            h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

            rect_canvas = (x, y, x + w, y + h)

            # Converte para a imagem original
            sx, sy = W / canvas_w, H / canvas_h
            x1 = int(rect_canvas[0] * sx)
            y1 = int(rect_canvas[1] * sy)
            x2 = int(rect_canvas[2] * sx)
            y2 = int(rect_canvas[3] * sy)

            store[img_name].append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": code,
                    "rater": rater.strip(),
                }
            )
            save_json(ANN_JSON, store)
            st.success(f"Added: {code} at {(x1,y1,x2,y2)}")
        else:
            st.info("Draw a rectangle on the image first and release the mouse.")

    # Clear current image annotations
    if clear_btn:
        store[img_name] = []
        save_json(ANN_JSON, store)
        st.warning("Cleared annotations for this image.")

    # Show current annotations table
    st.markdown("#### Current annotations for this image")
    rows = store[img_name]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
    else:
        st.caption("No boxes yet.")

    # Save & export crops
    if save_btn:
        if not rows:
            st.info("No annotations to save.")
        else:
            # Write JSON (already kept up to date) and export crops
            img_bgr = cv2.imread(str(img_path))
            H0, W0 = img_bgr.shape[:2]
            out_rows = []
            exported = 0
            for i, r in enumerate(rows):
                x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
                x1 = max(0, min(W0 - 1, x1))
                x2 = max(0, min(W0 - 1, x2))
                y1 = max(0, min(H0 - 1, y1))
                y2 = max(0, min(H0 - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img_bgr[y1:y2, x1:x2]
                out_name = f"{img_path.stem}_tooth{i}_{x1}_{y1}_{x2}_{y2}.jpg"
                out_path = CROP_DIR / out_name
                cv2.imwrite(str(out_path), crop)
                exported += 1
                out_rows.append(
                    {
                        "img_path": str(img_path),
                        "crop_path": str(out_path),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "pufa_label": r["label"],
                        "rater": r["rater"],
                        "width": W0,
                        "height": H0,
                    }
                )
            if out_rows:
                upsert_manifest(out_rows)
            st.success(f"Saved JSON and exported {exported} crops.")
            st.info(
                "Tip: move to another image via the selectbox. Your work is saved incrementally per image."
            )

    # Help
    st.markdown("---")
    st.info(
        "### How to use\n"
        "1. Choose an image.\n"
        "2. Draw a rectangle around a tooth.\n"
        "3. Pick the PUFA code and click **Add selection**.\n"
        "4. Repeat for all teeth in the image.\n"
        "5. Press **Save all & export crops**.\n"
        "**Remember to fill your *Rater ID*** so we can track provenance.",
        icon="ℹ️",
    )


if __name__ == "__main__":
    main()
