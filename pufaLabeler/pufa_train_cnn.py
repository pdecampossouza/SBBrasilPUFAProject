# -*- coding: utf-8 -*-
"""
PUFA CNN Training Script — versão PRO para paper IEEE
Inclui:
✅ Cross-validation por voluntário
✅ Gráficos automáticos de loss/accuracy
✅ Salvamento de modelos por fold
✅ Resumo estatístico ao final
Autor: Paulo + ChatGPT (2025)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==============================
# Função: identificar voluntário
# ==============================
def get_volunteer_id(p):
    s = str(p)
    m = re.search(r"Voluntario\\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()
    return Path(s).stem.lower()


# ==============================
# Função: construir modelo CNN
# ==============================
def build_model(num_classes, lr=1e-4, input_shape=(128, 128, 3)):
    tf.keras.backend.clear_session()
    print("[INFO] Building CNN model...")

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ========================================
# Função: carregar dataset (imagens + CSV)
# ========================================
def load_dataset(manifest_path, image_size=(128, 128), binary=True):
    print(f"[INFO] Loaded samples from: {manifest_path}")
    df = pd.read_csv(manifest_path)

    need = ["crop_path", "img_path", "pufa_label"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Manifest missing column: {c}")

    df = df[df["crop_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if df.empty:
        raise SystemExit("[ERROR] No existing crops on disk after filtering.")

    print(f"[INFO] Found {len(df)} image crops.")

    if binary:
        df["target"] = (df["pufa_label"] != "0").astype(int)
        y = df["target"].values
        num_classes = 2
        print("[INFO] Binary setup: 0=healthy, 1=lesion")
    else:
        allowed = {"0", "P", "U", "F", "A"}
        df = df[df["pufa_label"].isin(allowed)]
        df["target"], uniques = pd.factorize(df["pufa_label"])
        y = df["target"].values
        num_classes = len(uniques)
        print(f"[INFO] Multiclass setup: {uniques}")

    X = []
    for p in df["crop_path"]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        X.append(img.astype("float32") / 255.0)

    X = np.array(X, dtype=np.float32)
    y = y[: len(X)]
    groups = df["img_path"].apply(get_volunteer_id).values
    y_cat = to_categorical(y, num_classes)

    print(f"[INFO] Data shape: {X.shape}, Labels: {num_classes} classes")
    return X, y_cat, groups, num_classes


# ======================
# Função principal (main)
# ======================
def main():
    parser = argparse.ArgumentParser(description="PUFA CNN Trainer PRO")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV file")
    parser.add_argument(
        "--binary", action="store_true", help="Use binary classification"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    X, y, groups, num_classes = load_dataset(args.manifest, binary=args.binary)

    print("[INFO] Using StratifiedGroupKFold (5 folds)")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, np.argmax(y, axis=1), groups), start=1
    ):
        print(f"\n[INFO] Fold {fold}/5")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(num_classes, lr=args.lr)

        checkpoint_path = f"cnn_fold{fold}.keras"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_results.append((val_acc, val_loss))
        print(
            f"[RESULT] Fold {fold} — val_accuracy={val_acc:.4f}, val_loss={val_loss:.4f}"
        )
        print(f"[INFO] Saved model: {checkpoint_path}")

        # ==== Plot e salvar gráficos ====
        plt.figure(figsize=(6, 4))
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold} — Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cnn_fold{fold}_acc.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold} — Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cnn_fold{fold}_loss.png")
        plt.close()

    # ======= Resumo Final =======
    accs = [a for a, _ in fold_results]
    losses = [l for _, l in fold_results]
    print("\n=========== FINAL RESULTS ===========")
    print(f"Mean Val Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean Val Loss:     {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print("=====================================")
    print("[INFO] Training completed successfully!")


# ================
# Execução padrão
# ================
if __name__ == "__main__":
    main()
