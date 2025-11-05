# fix_manifest_isabel.py
from pathlib import Path
import pandas as pd, os

# pastas na SUA máquina
BASE = Path("ClassificacaoIsabel").resolve()
CROPS_DIR = BASE / "images_by_tooth" / "PUFA"
SRC = BASE / "metadata" / "manifest_teeth.csv"  # manifesto original da Isabel
OUT = Path("metadata") / "manifest_teeth_isabel.csv"  # manifesto corrigido

df = pd.read_csv(SRC)

# mantém apenas colunas relevantes que existirem no arquivo
cols = [
    c
    for c in [
        "img_path",
        "crop_path",
        "tooth_idx",
        "x1",
        "y1",
        "x2",
        "y2",
        "pufa_label",
        "rater",
    ]
    if c in df.columns
]
df = df[cols].copy()


def to_local(p):
    name = Path(str(p)).name  # só o nome do arquivo
    cand = CROPS_DIR / name  # tenta achar na sua pasta de crops
    return str(cand) if cand.exists() else None


df["crop_path"] = df["crop_path"].astype(str).apply(to_local)

# mantém só linhas cujos arquivos existem
df = df[df["crop_path"].notna()].drop_duplicates(subset=["crop_path"])

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"✔ Salvo {len(df)} linhas em {OUT}")
print("Exemplo:")
print(df.head(3).to_string(index=False))

# checagem extra
ok = sum(os.path.exists(p) for p in df["crop_path"])
print("Arquivos existentes no disco:", ok)
