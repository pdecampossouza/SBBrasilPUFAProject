# -*- coding: utf-8 -*-
"""
PUFA Deep Learning Training Script (Fixed Version)
Modelo: EfficientNetB0 (Transfer Learning)
Compatível com manifesto e estrutura do pufa_train_baseline.py
Autor: Paulo Vitor de Campos Souza
"""

from pathlib import Path
import argparse, re, os, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(".")
CONF_FIG_RAW = ROOT / "metadata" / "pufa_confusion_dl.png"
CONF_FIG_NORM = ROOT / "metadata" / "pufa_confusion_dl_norm.png"


# ---------- Funções utilitárias ----------
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


def get_volunteer_id(p):
    s = str(p)
    m = re.search(r"Voluntario\d+", s, flags=re.IGNORECASE)
    return m.group(0).lower() if m else Path(s).stem.lower()


def load_manifest(path_csv):
    df = pd.read_csv(path_csv)
    need = ["crop_path", "img_path", "pufa_label"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Manifest missing column: {c}")
    df = df[df["crop_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if df.empty:
        raise SystemExit("[ERROR] No crops found.")
    df["pufa_label"] = df["pufa_label"].astype(str).str.strip()
    df["volunteer"] = df["img_path"].apply(get_volunteer_id)
    return df


# ---------- Leitura das imagens ----------
def load_image(path, size=224):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # força RGB
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    # garante 3 canais mesmo se vier em escala de cinza
    if img.shape[-1] != 3:
        img = cv2.merge([img, img, img])

    img = img.astype("float32") / 255.0
    return img


# ---------- Modelo DL ----------
def build_model(num_classes, lr=1e-4):
    """
    Cria o modelo EfficientNetB0 sem pesos pré-treinados (ImageNet),
    evitando conflitos de forma entre pesos e camadas.
    Ideal para datasets customizados pequenos e setups offline.

    Args:
        num_classes (int): número de classes de saída
        lr (float): taxa de aprendizado (default 1e-4)

    Returns:
        model (tf.keras.Model): modelo compilado pronto para treino
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    # Garante que não há resíduos de sessões anteriores
    tf.keras.backend.clear_session()

    print("[INFO] Building EfficientNetB0 model (no pre-trained weights)...")

    # Cria a base da EfficientNet do zero
    base = EfficientNetB0(
        weights=None,  # <-- sem pesos ImageNet (treino do zero)
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Permite treinar todas as camadas (já que não há pesos congelados)
    base.trainable = True

    # Cabeçalho do modelo
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Compilação final
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())
    return model


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"[ERROR] Manifest not found: {manifest}")

    df = load_manifest(manifest)
    print(f"[INFO] Loaded {len(df)} samples.")

    # --- Configuração dos rótulos ---
    if args.binary:
        df["target"] = (df["pufa_label"] != "0").astype(int)
        y = df["target"].values
        labels = [0, 1]
        print("[INFO] Binary setup: 0=healthy, 1=lesion")
    else:
        allowed = {"0", "P", "U", "F", "A"}
        df = df[df["pufa_label"].isin(allowed)]
        df["target"], uniques = pd.factorize(df["pufa_label"])
        y = df["target"].values
        labels = [str(u) for u in uniques]
        print("[INFO] Multiclass setup:", labels)

    groups = df["volunteer"].values

    # --- Carregar imagens ---
    print("[INFO] Loading images...")
    X = []
    for p in df["crop_path"]:
        img = load_image(p)
        if img is not None:
            X.append(img)
        else:
            print("[WARN] Could not read:", p)
    X = np.array(X, dtype=np.float32)
    y = y[: len(X)]
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes)

    # garante coerência dimensional (N, 224, 224, 3)
    if X.ndim != 4 or X.shape[-1] != 3:
        print(f"[WARN] Fixing X shape: {X.shape} -> (N, 224, 224, 3)")
        X = np.stack(
            [np.repeat(x[..., np.newaxis], 3, axis=-1) if x.ndim == 2 else x for x in X]
        )

    print(f"[INFO] Data shape: {X.shape}, Labels: {num_classes} classes")

    # --- Validação cruzada por voluntário ---
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        split_iter = splitter.split(X, y, groups)
        print(f"[INFO] Using StratifiedGroupKFold ({n_splits} folds)")
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, groups=groups)
        print(f"[INFO] Using GroupKFold ({n_splits} folds)")

    y_true_all, y_pred_all = [], []
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    for fold, (tr, te) in enumerate(split_iter, 1):
        print(f"\n[INFO] Fold {fold}/{n_splits}")
        model = build_model(num_classes)
        hist = model.fit(
            X[tr],
            y_cat[tr],
            validation_data=(X[te], y_cat[te]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=[es],
        )
        preds = np.argmax(model.predict(X[te]), axis=1)
        y_true_all.extend(y[te])
        y_pred_all.extend(preds)

    # --- Avaliação final ---
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    print(f"\n[RESULT] Macro-F1={macro_f1:.3f}")
    print(
        classification_report(
            y_true_all, y_pred_all, digits=3, labels=np.unique(y_true_all)
        )
    )

    cm_raw = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_true_all))
    plot_confusion(cm_raw, labels, CONF_FIG_RAW, "PUFA DL — Confusion (raw)")
    print("Saved:", CONF_FIG_RAW)

    cm_norm = confusion_matrix(
        y_true_all, y_pred_all, labels=np.unique(y_true_all), normalize="true"
    )
    plot_confusion(cm_norm, labels, CONF_FIG_NORM, "PUFA DL — Confusion (normalized)")
    print("Saved:", CONF_FIG_NORM)

    # --- ROC-AUC se binário ---
    if args.binary:
        y_pred_prob = model.predict(X)[:, 1]
        auc = roc_auc_score(y_true_all, y_pred_prob)
        print(f"ROC-AUC (binary): {auc:.3f}")


if __name__ == "__main__":
    main()
