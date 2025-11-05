# -*- coding: utf-8 -*-
# Streamlit free-draw labeler for PUFA (retângulos com o mouse)
# Rodar:
#   python -m streamlit run pufa_free_draw_labeler.py

from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ----- caminhos (relativos a este arquivo) -----
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "Pranchetas fotografias - PUFA - treinamento"
ANN_DIR = BASE_DIR / "annotations"
ANN_PATH = ANN_DIR / "tooth_labels_free_draw.json"
CROP_DIR = BASE_DIR / "images_by_tooth" / "PUFA"
TEETH_MANIFEST = BASE_DIR / "metadata" / "manifest_teeth.csv"

PUFA_CLASSES = [
    "0",
    "P",
    "U",
    "F",
    "A",
]  # 0=None, P=pulpar, U=ulceração, F=fístula, A=abscesso


# -------------- helpers --------------
def list_images() -> List[Path]:
    if not IMG_DIR.exists():
        return []
    return sorted(
        [*IMG_DIR.glob("*.jpg"), *IMG_DIR.glob("*.jpeg"), *IMG_DIR.glob("*.png")]
    )


def load_ann() -> dict:
    if ANN_PATH.exists():
        return json.loads(ANN_PATH.read_text(encoding="utf-8"))
    return {}


def save_ann(ann: dict) -> None:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    ANN_PATH.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def crop_and_save(img_path: Path, rects: List[dict]) -> List[dict]:
    """
    rect = {"x1":..,"y1":..,"x2":..,"y2":..,"label":..,"tooth_id":..,"rater":..}
    """
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    cv = cv2.imread(str(img_path))
    if cv is None:
        return []
    h, w = cv.shape[:2]

    out = []
    for r in rects:
        x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = cv[y1:y2, x1:x2]
        name = f"{img_path.stem}_tooth{r['tooth_id']}_{x1}_{y1}_{x2}_{y2}.jpg"
        out_path = CROP_DIR / name
        cv2.imwrite(str(out_path), crop)
        out.append(
            {
                "img_path": str(img_path),
                "crop_path": str(out_path),
                "tooth_idx": int(r["tooth_id"]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "pufa_label": r["label"],
                "rater": r.get("rater", ""),
            }
        )
    return out


def upsert_manifest(rows: List[dict]) -> None:
    TEETH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if TEETH_MANIFEST.exists():
        base = pd.read_csv(TEETH_MANIFEST)
        df = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["crop_path"], inplace=True)
    df.to_csv(TEETH_MANIFEST, index=False)


def scale_rect(rect: dict, scale_x: float, scale_y: float) -> dict:
    """Converts rectangle from canvas scale to original image coordinates."""
    out = rect.copy()
    out["x1"] = rect["x1"] / scale_x
    out["x2"] = rect["x2"] / scale_x
    out["y1"] = rect["y1"] / scale_y
    out["y2"] = rect["y2"] / scale_y
    return out


# -------------- app --------------
def main():
    st.set_page_config(page_title="PUFA Free-Draw Labeler", layout="wide")

    st.title("PUFA Free-Draw Labeler (mouse rectangles)")

    # Rater (avaliador)
    with st.sidebar:
        st.header("Reviewer")
        rater = st.text_input("Your name / ID", value=st.session_state.get("rater", ""))
        st.session_state.rater = rater

        st.markdown("---")
        st.write("**Tips**")
        st.caption(
            "Draw a rectangle over a tooth. When you release the mouse, select the PUFA label.\n"
            "Use Undo to remove the last rectangle; Clear to clear them all."
        )

    images = list_images()
    if not images:
        st.error(f"No images in: `{IMG_DIR}`")
        st.stop()

    # --- BLOCO NOVO (cole no lugar do anterior) ---
    names = [p.name for p in images]

    # estado inicial (uma única fonte de verdade)
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        prev = st.button("← Prev", key="btn_prev", use_container_width=True)
    with col3:
        next_ = st.button("Next →", key="btn_next", use_container_width=True)

    # aplica cliques
    if prev:
        st.session_state.img_idx = (st.session_state.img_idx - 1) % len(names)
    if next_:
        st.session_state.img_idx = (st.session_state.img_idx + 1) % len(names)

    # selectbox vinculado ao estado
    with col2:
        img_selected = st.selectbox(
            "Imagem", options=names, index=st.session_state.img_idx, key="img_select"
        )

    # se mudou pelo selectbox, sincroniza o índice
    if img_selected != names[st.session_state.img_idx]:
        st.session_state.img_idx = names.index(img_selected)

    img_path = images[st.session_state.img_idx]
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # largura exibida no canvas (mantém proporção)
    display_w = min(800, W)  # pode ajustar
    scale_x = display_w / W
    display_h = int(H * scale_x)
    scale_y = display_h / H

    # carrega/salva anotações
    ann = load_ann()
    key = img_path.name
    if key not in ann:
        ann[key] = []

    # UI principal
    st.subheader(f"{img_path.name} · {W}×{H}px")
    # Canvas para desenhar retângulos
    canvas = st_canvas(
        fill_color="rgba(0, 255, 0, 0.15)",
        stroke_width=3,
        stroke_color="#22c55e",
        background_image=img.resize((display_w, display_h)),
        background_color="#00000000",
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key=f"canvas_{key}",
        display_toolbar=True,
        update_streamlit=True,
    )

    # Mostrar retângulos já existentes (como overlay simples)
    if ann[key]:
        st.caption(f"Existing annotations: {len(ann[key])}")

    # Detecta retângulos novos do canvas
    new_rect = None
    if canvas.json_data is not None and "objects" in canvas.json_data:
        objs = canvas.json_data["objects"]
        if objs:
            # pega o ÚLTIMO retângulo desenhado da sessão (tipo 'rect')
            for o in reversed(objs):
                if o.get("type") == "rect":
                    x, y = o["left"], o["top"]
                    w, h = o["width"], o["height"]
                    # desconsidera retângulos minúsculos
                    if w < 5 or h < 5:
                        continue
                    new_rect = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
                    break

    # Lógica de rotulagem do último retângulo
    if new_rect:
        # converte para coords da imagem original
        rect_full = scale_rect(new_rect, scale_x, scale_y)
        st.info("New rectangle detected. Select the label and confirm.")
        with st.form("label_form", clear_on_submit=True):
            colL, colR = st.columns([2, 1])
            with colL:
                label = st.selectbox("PUFA label", PUFA_CLASSES, index=0)
            with colR:
                tooth_id = st.number_input(
                    "Tooth ID (sequencial)", min_value=0, value=len(ann[key]), step=1
                )
            submitted = st.form_submit_button("Add annotation")
        if submitted:
            entry = {
                "x1": float(rect_full["x1"]),
                "y1": float(rect_full["y1"]),
                "x2": float(rect_full["x2"]),
                "y2": float(rect_full["y2"]),
                "label": label,
                "tooth_id": int(tooth_id),
                "rater": rater or "",
            }
            ann[key].append(entry)
            save_ann(ann)
            st.success("Annotation added.")

    # Tabela rápida com as anotações da imagem atual
    if ann[key]:
        st.write(pd.DataFrame(ann[key]))

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Undo last"):
            if ann[key]:
                ann[key].pop()
                save_ann(ann)
                st.warning("Last annotation removed.")
    with colB:
        if st.button("Clear all (this image)"):
            ann[key] = []
            save_ann(ann)
            st.warning("All annotations cleared for this image.")
    with colC:
        if st.button("Export crops & update manifest", type="primary"):
            rows = crop_and_save(img_path, ann[key])
            if rows:
                upsert_manifest(rows)
            st.success(f"Exported {len(rows)} crops.")

    st.markdown("---")
    st.caption(
        "Workflow: draw a rectangle → choose the label → **Add annotation**. "
        "Repeat for all teeth with lesion or concern. "
        "Use **Export crops** at the end to generate crops and update `metadata/manifest_teeth.csv`."
    )


if __name__ == "__main__":
    main()
