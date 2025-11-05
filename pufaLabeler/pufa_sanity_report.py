# -*- coding: utf-8 -*-
# Relatório rápido do PUFA a partir de metadata/manifest_teeth.csv
from pathlib import Path
import pandas as pd

ROOT = Path(".")
MANIFEST = ROOT / "metadata" / "manifest_teeth.csv"


def main():
    if not MANIFEST.exists():
        print(f"[ERRO] Manifest não encontrado: {MANIFEST}")
        return
    df = pd.read_csv(MANIFEST)
    print("\n=== Cabeçalho do manifest_teeth ===")
    print(df.head(10).to_string(index=False))

    print("\n=== Contagem por classe (pufa_label) ===")
    print(df["pufa_label"].value_counts(dropna=False).to_string())

    print("\n=== Amostras por imagem original ===")
    print(df["img_path"].value_counts().to_string())

    # Opcional: amostras por faixa
    if "tooth_idx" in df.columns:
        print("\n=== Contagem por índice de faixa (tooth_idx) ===")
        print(df["tooth_idx"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
