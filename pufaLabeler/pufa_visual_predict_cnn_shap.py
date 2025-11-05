import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# 🔹 Função principal de predição com SHAP-like viz
# ==========================================


def slide_and_predict_heatmap(img_bgr, model, win=128, stride=64):
    """Aplica o modelo CNN em janelas deslizantes e retorna probabilidades."""
    H, W = img_bgr.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - win, stride):
        for x in range(0, W - win, stride):
            crop = img_bgr[y : y + win, x : x + win]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (128, 128)) / 255.0
            crop_resized = np.expand_dims(crop_resized, axis=0)

            preds = model.predict(crop_resized, verbose=0)[0]
            prob_bad = preds[1] if len(preds) > 1 else preds[0]

            heatmap[y : y + win, x : x + win] += prob_bad
            counts[y : y + win, x : x + win] += 1

    heatmap /= np.maximum(counts, 1e-8)
    return heatmap


def visualize_with_heatmap(img_path, heatmap, out_path):
    """Combina imagem original com o mapa de calor interpretável."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return

    heatmap_norm = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(
        (heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.6, 0)

    # Legenda simples
    pad = 10
    cv2.rectangle(overlay, (pad, pad), (240, 80), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        "Good (Low Risk)",
        (pad + 10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        "Bad (High Risk)",
        (pad + 10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.imwrite(str(out_path), overlay)
    print(f"[INFO] Saved interpretability heatmap: {out_path}")


# ==========================================
# 🔹 Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="PUFA CNN Visual Prediction + Heatmap")
    parser.add_argument(
        "--model", required=True, help="Path to .keras model file to evaluate"
    )
    parser.add_argument(
        "--img_dir", required=True, help="Directory with test images (.jpeg or .png)"
    )
    parser.add_argument(
        "--out_dir", default="metadata/visual_predictions_shap", help="Output directory"
    )
    parser.add_argument("--win", type=int, default=128, help="Sliding window size")
    parser.add_argument(
        "--stride", type=int, default=64, help="Stride size for sliding window"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"[INFO] Loaded model: {model_path.name}")

    img_dir = Path(args.img_dir)
    imgs = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
    )
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in {img_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        print(f"[INFO] Processing {img_path.name}")
        img = cv2.imread(str(img_path))
        heatmap = slide_and_predict_heatmap(
            img, model, win=args.win, stride=args.stride
        )
        out_path = out_dir / f"{Path(img_path).stem}_cnn_heatmap.png"
        visualize_with_heatmap(img_path, heatmap, out_path)

    print("[INFO] All interpretability maps generated successfully.")


if __name__ == "__main__":
    main()
