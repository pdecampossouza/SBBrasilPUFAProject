import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

ROOT = Path(".")
MANIFEST = ROOT / "metadata" / "manifest_teeth_isabel.csv"
IMG_DIR = ROOT / "Pranchetas fotografias - PUFA - treinamento"
OUT_DIR = ROOT / "metadata" / "visual_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cores em BGR (OpenCV)
GOOD_COLOR_BGR = (255, 0, 0)  # Azul (Bom)
BAD_COLOR_BGR = (0, 255, 255)  # Amarelo (Ruim) - alto contraste


def extract_features(img_bgr):
    roi = cv2.resize(img_bgr, (128, 128))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
    ).flatten()
    hist /= hist.sum() + 1e-8
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_feat = hog.compute(roi).reshape(-1)
    return np.concatenate([hog_feat, hist])


def train_model():
    print("[INFO] Loading manifest:", MANIFEST)
    df = pd.read_csv(MANIFEST)
    df = df[df["crop_path"].apply(lambda p: Path(p).exists())]
    df["target"] = (df["pufa_label"] != "0").astype(int)
    X, y = [], []
    for _, row in df.iterrows():
        img = cv2.imread(str(row["crop_path"]))
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(row["target"])
    X, y = np.vstack(X), np.array(y)
    print(f"[INFO] Training set: {X.shape}, Labels: {np.bincount(y)}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=8, random_state=42)
    Xp = pca.fit_transform(Xs)
    clf = LinearSVC(class_weight="balanced", C=0.5, random_state=42)
    clf.fit(Xp, y)

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, "o-")
    plt.title("PCA Explained Variance (%)")
    plt.xlabel("Component")
    plt.ylabel("Cumulative Variance (%)")
    plt.grid(True)
    plt.savefig(ROOT / "metadata" / "pufa_pca_explained_variance.png", dpi=160)
    plt.close()

    print("[INFO] Training SHAP explainability...")
    explainer = shap.LinearExplainer(
        clf, Xp, feature_names=[f"PC{i+1}" for i in range(8)]
    )
    shap_values = explainer(Xp)
    shap.summary_plot(
        shap_values, Xp, feature_names=[f"PC{i+1}" for i in range(8)], show=False
    )
    plt.tight_layout()
    plt.savefig(ROOT / "metadata" / "pufa_shap_summary.png", dpi=160)
    plt.close()

    return scaler, pca, clf


def slide_and_predict(img_bgr, scaler, pca, clf, stride=64, win=128):
    H, W = img_bgr.shape[:2]
    results = []
    for y in range(0, H - win, stride):
        for x in range(0, W - win, stride):
            crop = img_bgr[y : y + win, x : x + win]
            feat = extract_features(crop)
            Xs = scaler.transform([feat])
            Xp = pca.transform(Xs)
            pred = clf.predict(Xp)[0]
            results.append((x, y, pred))
    return results


def visualize_predictions(
    img_path, preds, win=128, alpha_good=0.35, alpha_bad=0.0, thickness=2
):
    """
    Desenha janelas; boas (pred==0) em azul com preenchimento semitransparente;
    ruins (pred==1) em amarelo (por padrão só contorno). Ajuste alpha_bad>0 para
    preencher as ruins também.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return

    # Overlays separados para aplicar alphas diferentes
    overlay_good = img.copy()
    overlay_bad = img.copy()

    for x, y, pred in preds:
        p1, p2 = (x, y), (x + win, y + win)
        if pred == 0:
            # preenchimento azul
            cv2.rectangle(overlay_good, p1, p2, GOOD_COLOR_BGR, -1)
            # contorno azul
            cv2.rectangle(img, p1, p2, GOOD_COLOR_BGR, thickness, lineType=cv2.LINE_AA)
        else:
            if alpha_bad > 0:
                cv2.rectangle(overlay_bad, p1, p2, BAD_COLOR_BGR, -1)
            # contorno amarelo
            cv2.rectangle(img, p1, p2, BAD_COLOR_BGR, thickness, lineType=cv2.LINE_AA)

    # Aplica blend para os preenchimentos
    if alpha_good > 0:
        img = cv2.addWeighted(overlay_good, alpha_good, img, 1 - alpha_good, 0)
    if alpha_bad > 0:
        img = cv2.addWeighted(overlay_bad, alpha_bad, img, 1 - alpha_bad, 0)

    # Legenda
    pad = 10
    legend_w, legend_h = 230, 64
    cv2.rectangle(img, (pad, pad), (pad + legend_w, pad + legend_h), (0, 0, 0), -1)
    cv2.putText(
        img,
        "Good",
        (pad + 12, pad + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GOOD_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Bad",
        (pad + 12, pad + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BAD_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )

    out_path = OUT_DIR / f"{Path(img_path).stem}_prediction.png"
    cv2.imwrite(str(out_path), img)
    print(f"[INFO] Saved visualization: {out_path}")


def main():
    scaler, pca, clf = train_model()
    imgs = sorted(IMG_DIR.glob("*.jpeg"))
    print(f"[INFO] Found {len(imgs)} original images.")
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue
        preds = slide_and_predict(img, scaler, pca, clf, stride=64, win=128)
        visualize_predictions(img_path, preds, win=128, alpha_good=0.35, alpha_bad=0.0)
    print("[INFO] All predictions complete.")


if __name__ == "__main__":
    main()
