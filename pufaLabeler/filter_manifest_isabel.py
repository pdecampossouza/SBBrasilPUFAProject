# prepare_manifest_isabel.py  (versão SEM filtro por rater)
from pathlib import Path
import pandas as pd

BASE = Path("ClassificacaoIsabel").resolve()
CROPS_DIR = BASE / "images_by_tooth" / "PUFA"
MANIFEST_IN = BASE / "metadata" / "manifest_teeth.csv"
OUT = Path("metadata") / "manifest_teeth_isabel.csv"

df = pd.read_csv(MANIFEST_IN)

keep_cols = [
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
df = df[keep_cols].copy()


def remap_crop(p):
    name = Path(str(p)).name
    cand = CROPS_DIR / name
    return str(cand) if cand.exists() else None


df["crop_path_local"] = df["crop_path"].apply(remap_crop)
df_local = df[df["crop_path_local"].notna()].copy()

# usa o caminho local encontrado
df_local["crop_path"] = df_local["crop_path_local"]
df_local.drop(columns=["crop_path_local"], inplace=True, errors="ignore")
df_local.drop_duplicates(subset=["crop_path"], inplace=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df_local.to_csv(OUT, index=False)
print(f"Salvo {len(df_local)} linhas em {OUT}")
