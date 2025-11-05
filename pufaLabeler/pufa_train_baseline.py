# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, re, os
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ROOT = Path(".")
CONF_FIG_RAW = ROOT / "metadata" / "pufa_confusion.png"
CONF_FIG_NORM = ROOT / "metadata" / "pufa_confusion_norm.png"


# ---------- features ----------
def extract_features(img_bgr, use_hog=True):
    # descarta crops muito pequenos
    h, w = img_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    # normaliza contraste (L-channel)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    roi = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
    ).flatten()
    hist = hist / (hist.sum() + 1e-8)

    if not use_hog:
        return hist

    try:
        hog = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        hog_feat = hog.compute(roi).reshape(-1)  # ~1764 dims
        return np.concatenate([hog_feat, hist])
    except Exception as e:
        print("[WARN] HOG failed, using HSV only:", repr(e))
        return hist


# ---------- viz ----------
def plot_confusion(cm, labels, out_png, title="PUFA — Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if cm.dtype != int else int(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------- utils ----------
def get_volunteer_id(p):
    # tenta "Voluntario12" do nome do arquivo; fallback = basename sem extensão
    s = str(p)
    m = re.search(r"Voluntario\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()
    # fallback: usa nome do arquivo como pseudo-grupo
    return Path(s).stem.lower()


def load_manifest(path_csv):
    df = pd.read_csv(path_csv)
    need = ["crop_path", "img_path", "pufa_label"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Manifest missing column: {c}")
    # mantém apenas arquivos existentes
    df = df[df["crop_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if df.empty:
        raise SystemExit(
            "[ERROR] No existing crops on disk after filtering by file existence."
        )
    # normaliza rótulos
    df["pufa_label"] = df["pufa_label"].astype(str).str.strip()
    # volunteer/group
    df["volunteer"] = df["img_path"].apply(get_volunteer_id)
    return df


def main():
    # -------- args --------
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "metadata" / "manifest_teeth.csv"),
        help="Path to manifest CSV (default: metadata/manifest_teeth.csv)",
    )
    ap.add_argument(
        "--binary",
        action="store_true",
        help="Binary setup: 0 (no lesion) vs 1 (lesion: P/U/F/A)",
    )
    ap.add_argument(
        "--c_grid",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="Grid of C values for LinearSVC (comma-separated)",
    )
    ap.add_argument(
        "--pca_grid",
        type=str,
        default="16,24,32",
        help="Grid of PCA components (comma-separated)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Decision threshold for positive class (binary, after calibration)",
    )
    args = ap.parse_args()

    manifest = Path(args.manifest)
    print(f"[INFO] Using manifest: {manifest.resolve()}")
    if not manifest.exists():
        raise SystemExit(f"[ERROR] Manifest not found: {manifest}")

    df = load_manifest(manifest)
    print(f"[INFO] Loaded {len(df)} rows with existing crops.")
    print("[INFO] Label counts (raw):\n", df["pufa_label"].value_counts())

    # ----- target definition -----
    if args.binary:
        df["target"] = (df["pufa_label"] != "0").astype(int)
        y_name = "target"
        labels = [0, 1]
        print("[INFO] Binary setup enabled (0=no lesion, 1=lesion)")
        print(df[y_name].value_counts())
    else:
        # multiclasse com filtro suave (>= 2 amostras por classe)
        allowed = {"0", "P", "U", "F", "A"}
        df = df[df["pufa_label"].isin(allowed)].reset_index(drop=True)
        counts = df["pufa_label"].value_counts()
        keep = counts[counts >= 2].index.tolist()
        drop = [c for c in counts.index if c not in keep]
        if drop:
            print(f"[INFO] Dropping rare classes (<2): {drop}")
            df = df[df["pufa_label"].isin(keep)].reset_index(drop=True)
        if df["pufa_label"].nunique() < 2:
            print("[WARN] <2 classes left — forcing binary mode")
            df["target"] = (df["pufa_label"] != "0").astype(int)
            y_name = "target"
            labels = [0, 1]
        else:
            y_name = "pufa_label"
            labels = sorted(df[y_name].unique().tolist())

    # ----- features -----
    X_list, y_list, grp_list = [], [], []
    skipped = 0
    for _, row in df.iterrows():
        img = cv2.imread(str(row["crop_path"]))
        if img is None:
            skipped += 1
            continue
        feat = extract_features(img, use_hog=True)
        if feat is None:
            skipped += 1
            continue
        X_list.append(feat)
        y_list.append(row[y_name])
        grp_list.append(row["volunteer"])
    if skipped:
        print(f"[INFO] Skipped {skipped} rows (read/feature issues).")
    if not X_list:
        raise SystemExit("[ERROR] No usable features.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(grp_list)

    print("Labels used:", sorted(np.unique(y)))
    print("Counts used:", {c: int((y == c).sum()) for c in np.unique(y)})

    # ----- CV splitter por voluntário -----
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    if (
        HAS_SGK and y_name != "pufa_label"
    ):  # SGK precisa de y binário/estratificado estável
        print(f"[INFO] Using StratifiedGroupKFold (n_splits={n_splits})")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        split_iter = splitter.split(X, y, groups)
    else:
        # fallback: GroupKFold evita vazamento; estratificação pode não ser perfeita
        print(f"[INFO] Using GroupKFold (n_splits={n_splits})")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, groups=groups)

    # ----- grids -----
    c_grid = [float(v) for v in args.c_grid.split(",") if v.strip()]
    pca_grid = [int(v) for v in args.pca_grid.split(",") if v.strip()]
    print(f"[INFO] Grid C: {c_grid}")
    print(f"[INFO] Grid PCA: {pca_grid}")

    best_macro_f1, best_cfg = -1.0, None
    best_preds, best_true, best_probs = None, None, []

    from sklearn.metrics import f1_score

    for C in c_grid:
        for ncomp_req in pca_grid:
            # recria o iterador a cada combinação
            if HAS_SGK and isinstance(splitter, StratifiedGroupKFold):
                split_iter = splitter.split(X, y, groups)
            else:
                split_iter = splitter.split(X, groups=groups)

            y_true_all, y_pred_all, prob_all = [], [], []

            for tr, te in split_iter:
                # 1) padroniza
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr_s = scaler.fit_transform(X[tr])
                Xte_s = scaler.transform(X[te])

                # 2) PCA fora da calibração, com n_components SEGURO por fold
                safe_ncomp = min(
                    ncomp_req,
                    Xtr_s.shape[1] - 1,  # n_features - 1
                    max(2, len(tr) - 1),  # n_amostras_treino - 1 (≥2)
                )
                pca = PCA(n_components=safe_ncomp, random_state=42)
                Xtr_p = pca.fit_transform(Xtr_s)
                Xte_p = pca.transform(Xte_s)

                # 3) classificador
                base = LinearSVC(class_weight="balanced", C=C, random_state=42)

                if y_name == "target":
                    # binário com calibração e threshold custom
                    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                    clf.fit(Xtr_p, y[tr])
                    prob = clf.predict_proba(Xte_p)[:, 1]
                    y_pred = (prob >= args.threshold).astype(int)
                    prob_all.append(prob.tolist())
                else:
                    # multiclasse direto
                    base.fit(Xtr_p, y[tr])
                    y_pred = base.predict(Xte_p)

                y_true_all.extend(y[te])
                y_pred_all.extend(y_pred)

            macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
            print(
                f"[GRID] C={C:.2f}, PCA={ncomp_req} (safe={safe_ncomp}) -> macro-F1={macro_f1:.3f}"
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_cfg = (C, ncomp_req)
                best_true = y_true_all[:]
                best_preds = y_pred_all[:]
                best_probs = prob_all[:]  # p/ ROC-AUC no binário

                from sklearn.metrics import f1_score

                macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
                print(
                    f"[GRID] C={C:.2f}, PCA={ncomp_req} (safe={safe_ncomp}) -> macro-F1={macro_f1:.3f}"
                )

                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_cfg = (C, ncomp_req)
                    best_true = y_true_all[:]
                    best_preds = y_pred_all[:]
                    best_probs = prob_all[:]  # list of lists (folds)

                # reconstruir o iterador para próximo grid
                split_iter = (
                    splitter.split(X, y, groups)
                    if HAS_SGK and isinstance(splitter, StratifiedGroupKFold)
                    else splitter.split(X, groups)
                )

        print(
            f"\n[BEST] macro-F1={best_macro_f1:.3f} @ C={best_cfg[0]:.2f}, PCA={best_cfg[1]}"
        )

    # ----- relatório e figuras -----
    labels_used = sorted(np.unique(best_true))
    print("\n=== PUFA Baseline (CV, best grid) ===")
    print(classification_report(best_true, best_preds, digits=3, labels=labels_used))

    cm_raw = confusion_matrix(best_true, best_preds, labels=labels_used)
    plot_confusion(
        cm_raw,
        [str(l) for l in labels_used],
        CONF_FIG_RAW,
        title="PUFA — Confusion (raw)",
    )
    print("Saved:", CONF_FIG_RAW)

    cm_norm = confusion_matrix(
        best_true, best_preds, labels=labels_used, normalize="true"
    )
    plot_confusion(
        cm_norm,
        [str(l) for l in labels_used],
        CONF_FIG_NORM,
        title="PUFA — Confusion (normalized)",
    )
    print("Saved:", CONF_FIG_NORM)

    if y_name == "target" and best_probs:
        # concat probs (folds) na mesma ordem de best_true/best_preds
        flat_probs = [p for fold in best_probs for p in fold]
        try:
            auc = roc_auc_score(best_true, flat_probs)
            print(f"ROC-AUC (binary): {auc:.3f}")
        except Exception as e:
            print("[WARN] ROC-AUC failed:", repr(e))


if __name__ == "__main__":
    main()
