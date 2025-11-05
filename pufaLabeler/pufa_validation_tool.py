# -*- coding: utf-8 -*-
# PUFA Validation Tool — Accept/Reject with Provenance (multi-root aware)
# Run:
#   python -m streamlit run pufa_validation_tool.py

from __future__ import annotations

from pathlib import Path
import hashlib
from datetime import datetime
from typing import List, Dict

import pandas as pd
from PIL import Image
import streamlit as st

# =========================
# Paths / constants (multi-root)
# =========================
APP_DIR = Path(__file__).resolve().parent

# Duas árvores paralelas com a mesma estrutura:
#   Local/
#     ├─ metadata/manifest_teeth.csv
#     └─ images_by_tooth/PUFA/...
#   ClassificacaoIsabel/
#     ├─ metadata/manifest_teeth.csv
#     └─ images_by_tooth/PUFA/...
DATA_ROOTS: dict[str, Path] = {
    "Local": APP_DIR,
    "ClassificacaoIsabel": APP_DIR / "ClassificacaoIsabel",
}

PUFA_LABELS = ["0", "P", "U", "F", "A"]


# =========================
# Helpers de caminho/ID
# =========================
def _resolve_under_roots(s: str, roots: List[Path], csv_dir: Path) -> str:
    """
    Resolve um caminho 's' tentando várias âncoras:
      1) caminho absoluto existente
      2) relativo ao diretório do CSV
      3) relativo a cada raiz candidata (Local/Isabel)
    Retorna string (pode não existir; será filtrado adiante).
    """
    s = (s or "").strip().strip('"').strip("'").replace("\\", "/")
    if not s:
        return ""
    p = Path(s).expanduser()

    # absoluto
    if p.is_absolute() and p.exists():
        return str(p)

    # relativo ao CSV
    p_csv = (csv_dir / s).resolve()
    if p_csv.exists():
        return str(p_csv)

    # relativo a cada raiz
    for r in roots:
        p_try = (r / s).resolve()
        if p_try.exists():
            return str(p_try)

    # sem sucesso
    return str(p)


def file_id(row: pd.Series) -> str:
    """ID estável por crop, usando caminho + coords (se houver)."""
    s = f"{row.get('crop_path','')}|{row.get('x1','')}|{row.get('y1','')}|{row.get('x2','')}|{row.get('y2','')}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# =========================
# IO do manifest e votos
# =========================
def load_manifest(selected_root: Path) -> tuple[pd.DataFrame, Path]:
    """
    Carrega e normaliza o manifest da RAIZ escolhida.
    - Normaliza 'crop_path' e 'img_path' contra: abs, pasta do CSV, raiz escolhida e demais raízes (fallback).
    - Garante 'id' (str), 'pufa_label' (str) e 'original_rater'.
    - Filtra para crops que realmente existem após normalização.
    """
    manifest = selected_root / "metadata" / "manifest_teeth.csv"
    if not manifest.exists():
        st.error(f"Manifest not found: {manifest}")
        st.stop()

    df = pd.read_csv(manifest)

    roots_order = [selected_root] + [
        r for r in DATA_ROOTS.values() if r != selected_root
    ]
    for col in ["crop_path", "img_path"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .map(lambda s: _resolve_under_roots(s, roots_order, manifest.parent))
            )

    # ID estável e tipos
    if "id" not in df.columns:
        df["id"] = df.apply(file_id, axis=1)
    df["id"] = df["id"].astype(str)
    df["pufa_label"] = df.get("pufa_label", "").astype(str)

    # renomeia rater -> original_rater se necessário
    if "original_rater" not in df.columns:
        if "rater" in df.columns:
            df = df.rename(columns={"rater": "original_rater"})
        else:
            df["original_rater"] = ""

    # mantém apenas crops existentes
    df = df[df["crop_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
    return df, manifest


def load_votes(votes_csv: Path) -> pd.DataFrame:
    """
    Carrega o CSV de votos, garantindo colunas e tipos (id como str).
    """
    cols = ["id", "validator", "decision", "corrected_label", "validated_at", "comment"]
    if votes_csv.exists():
        v = pd.read_csv(votes_csv)
        for c in cols:
            if c not in v.columns:
                v[c] = ""
        v["id"] = v["id"].astype(str)
        v["validator"] = v["validator"].astype(str)
        v["decision"] = v["decision"].map(lambda x: str(x).lower())
        v["corrected_label"] = v["corrected_label"].astype(str)
        return v[cols]
    return pd.DataFrame(columns=cols)


def save_votes(votes_csv: Path, votes: pd.DataFrame) -> None:
    votes_csv.parent.mkdir(parents=True, exist_ok=True)
    votes.to_csv(votes_csv, index=False)


# =========================
# Upsert & export
# =========================
def upsert_vote(
    votes: pd.DataFrame,
    row_id: str,
    validator: str,
    decision: str,
    corrected_label: str,
    comment: str,
) -> pd.DataFrame:
    """Insere/atualiza o voto de (id, validator)."""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    mask = (votes["id"] == row_id) & (votes["validator"] == validator)
    if mask.any():
        idx = votes.index[mask][0]
        votes.at[idx, "decision"] = decision
        votes.at[idx, "corrected_label"] = corrected_label
        votes.at[idx, "validated_at"] = now
        votes.at[idx, "comment"] = comment
    else:
        votes = pd.concat(
            [
                votes,
                pd.DataFrame(
                    [
                        {
                            "id": row_id,
                            "validator": validator,
                            "decision": decision,
                            "corrected_label": corrected_label,
                            "validated_at": now,
                            "comment": comment,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return votes


def decisions_for_id(votes: pd.DataFrame, row_id: str) -> pd.DataFrame:
    return votes[votes["id"] == row_id].sort_values("validated_at", ascending=False)


def export_validated(df: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o manifest validado: cada decisão ACCEPT vira uma linha.
    final_label usa corrected_label (quando presente e válido) senão pufa_label.
    Mantém original_rater e validator (proveniência).
    """
    acc = votes[votes["decision"] == "accept"].copy()
    if acc.empty:
        return pd.DataFrame(
            columns=[
                "img_path",
                "crop_path",
                "x1",
                "y1",
                "x2",
                "y2",
                "final_label",
                "original_label",
                "original_rater",
                "validator",
                "validated_at",
                "id",
            ]
        )
    mv = acc.merge(df, on="id", how="left", suffixes=("", "_orig"))
    mv["final_label"] = mv.apply(
        lambda r: (
            r["corrected_label"]
            if r.get("corrected_label") in PUFA_LABELS and r["corrected_label"] != ""
            else r["pufa_label"]
        ),
        axis=1,
    )
    mv = mv.rename(columns={"pufa_label": "original_label"})
    keep = [
        "img_path",
        "crop_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "final_label",
        "original_label",
        "original_rater",
        "validator",
        "validated_at",
        "id",
    ]
    keep = [c for c in keep if c in mv.columns]
    mv = mv[keep].dropna(subset=["crop_path"])
    return mv


# =========================
# UI
# =========================
def main():
    st.set_page_config(
        page_title="PUFA Validation Tool (with provenance)", layout="wide"
    )
    st.title("PUFA Validation — Accept/Reject with Provenance")

    # Escolha da raiz (Local / ClassificacaoIsabel)
    root_key = st.sidebar.selectbox("Dataset root", list(DATA_ROOTS.keys()), index=1)
    ROOT = DATA_ROOTS[root_key]

    # Caminhos dependentes da raiz
    VOTES_CSV = ROOT / "annotations" / "validation_votes.csv"
    OUT_VALIDATED = ROOT / "metadata" / "manifest_teeth_validated.csv"

    # Carrega manifest + votos
    df, manifest_path = load_manifest(ROOT)
    votes = load_votes(VOTES_CSV)

    st.caption(f"Using manifest: `{manifest_path}`  ·  Crops found: **{len(df)}**")

    # Sumário por validador (opcional)
    if not votes.empty:
        st.caption("**Validator summary:**")
        agg = (
            votes.groupby("validator")["decision"].value_counts().unstack(fill_value=0)
        )
        st.dataframe(agg, use_container_width=True)

    # Identidade do validador
    validator = st.text_input(
        "Validator ID (required)",
        value="",
        help="Your name/code (saved with each decision).",
    )
    if not validator.strip():
        st.warning("Fill the Validator ID to enable saving decisions.", icon="⚠️")

    # Filtros (sidebar)
    st.sidebar.header("Filters")
    sel_label = st.sidebar.multiselect(
        "PUFA class (original)", PUFA_LABELS, default=PUFA_LABELS
    )
    raters = sorted([str(x) for x in df["original_rater"].dropna().unique().tolist()])
    if not raters:
        raters = [""]
    sel_rater = st.sidebar.multiselect("Original rater", raters, default=raters)
    show_only_pending = st.sidebar.checkbox(
        "Show only pending for me (no decision by current validator)", value=False
    )
    q = st.sidebar.text_input("Search in filename (optional)", value="")
    per_page = st.sidebar.slider("Items per page", 12, 96, 24, 6)
    n_cols = st.sidebar.slider("Grid columns", 3, 8, 4, 1)

    # Merge com os votos do usuário (para mostrar estado atual)
    my_votes = (
        votes[votes["validator"] == validator]
        if validator.strip()
        else pd.DataFrame(columns=votes.columns)
    )
    dfv = df.merge(my_votes[["id", "decision", "corrected_label"]], on="id", how="left")

    # Aplica filtros
    mask = dfv["pufa_label"].isin(sel_label)
    if sel_rater:
        mask &= dfv["original_rater"].astype(str).isin(sel_rater)
    if q:
        ql = q.lower()
        mask &= dfv["crop_path"].astype(str).str.lower().str.contains(ql)
    if show_only_pending and validator.strip():
        mask &= dfv["decision"].isna()  # sem decisão do validador atual

    data = dfv[mask].copy().reset_index(drop=True)
    total_items = len(data)
    st.caption(f"Filtered items: **{total_items}**")

    if total_items == 0:
        st.info("No items with current filters.")
        st.stop()

    # Paginação
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, (total_items - 1) // per_page + 1),
        value=1,
        step=1,
    )
    lo = (page - 1) * per_page
    hi = min(lo + per_page, total_items)
    page_df = data.iloc[lo:hi].copy()

    # Grid
    cols = st.columns(n_cols, gap="large")
    pending_updates: List[Dict] = []

    for i, (_, row) in enumerate(page_df.iterrows()):
        with cols[i % n_cols]:
            # imagem
            try:
                im = Image.open(row["crop_path"]).convert("RGB")
                st.image(im, use_container_width=True)
            except Exception:
                st.error("Could not open image.")
                continue

            st.markdown(
                f"<small><code>{Path(str(row['crop_path'])).name}</code></small>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Original label: **{row['pufa_label']}**  ·  Original rater: {row.get('original_rater','')}"
            )

            # histórico de decisões
            hist = decisions_for_id(votes, row["id"])
            if not hist.empty:
                hist_text = "  •  ".join(
                    [
                        f"{r['validator']}: {r['decision']}{'→'+r['corrected_label'] if r['corrected_label'] else ''}"
                        for _, r in hist.iterrows()
                    ]
                )
                st.caption(f"History: {hist_text}")

            # controles do validador atual
            current_dec = row.get("decision", "")
            decision = st.radio(
                "Decision",
                ["", "accept", "reject"],
                index=["", "accept", "reject"].index(
                    current_dec if current_dec in ["", "accept", "reject"] else ""
                ),
                horizontal=True,
                key=f"dec_{row['id']}",
            )
            corrected_default = row.get("corrected_label", "")
            corrected = st.selectbox(
                "Correct label (optional)",
                [""] + PUFA_LABELS,
                index=(
                    ([""] + PUFA_LABELS).index(corrected_default)
                    if corrected_default in ([""] + PUFA_LABELS)
                    else 0
                ),
                key=f"corr_{row['id']}",
            )
            comment = st.text_input(
                "Comment (optional)", value="", key=f"cmt_{row['id']}"
            )

            pending_updates.append(
                {
                    "id": row["id"],
                    "decision": decision,
                    "corrected_label": corrected,
                    "comment": comment,
                }
            )

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        disabled = not validator.strip()
        if st.button(
            "Save progress (per validator)",
            type="primary",
            use_container_width=True,
            disabled=disabled,
        ):
            v = load_votes(VOTES_CSV)
            changed = 0
            for upd in pending_updates:
                if (
                    upd["decision"] == ""
                    and upd["corrected_label"] == ""
                    and upd["comment"] == ""
                ):
                    continue
                v = upsert_vote(
                    votes=v,
                    row_id=upd["id"],
                    validator=validator.strip(),
                    decision=upd["decision"],
                    corrected_label=upd["corrected_label"],
                    comment=upd["comment"],
                )
                changed += 1
            save_votes(VOTES_CSV, v)
            st.success(f"Saved {changed} update(s) to {VOTES_CSV.name}")

    with c2:
        if st.button("Export validated CSV", use_container_width=True):
            v = load_votes(VOTES_CSV)
            out = export_validated(df, v)
            OUT_VALIDATED.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(OUT_VALIDATED, index=False)
            st.success(f"Exported {len(out)} accepted rows to {OUT_VALIDATED.name}")
            st.caption(
                "Each accepted row keeps `original_rater` and `validator` for provenance; "
                "`final_label` applies corrections when present."
            )

    st.markdown("---")
    with st.expander("Schema details"):
        st.markdown(
            f"""
- **Input**: `{(ROOT / 'metadata' / 'manifest_teeth.csv').relative_to(APP_DIR)}` — deve conter `img_path`, `crop_path`, `x1..y2`, `pufa_label` e `original_rater`.
- **Votes store**: `{(ROOT / 'annotations' / 'validation_votes.csv').relative_to(APP_DIR)}` com chaves (`id`,`validator`):
  - `id`, `validator`, `decision` (`accept`/`reject`), `corrected_label` (opcional), `validated_at` (UTC), `comment`.
- **Export**: `{(ROOT / 'metadata' / 'manifest_teeth_validated.csv').relative_to(APP_DIR)}` contém apenas **accepted** com:
  - `img_path`, `crop_path`, coords, `final_label`, `original_label`, `original_rater`, `validator`, `validated_at`, `id`.
- Os caminhos de imagem são normalizados contra a **raiz selecionada** e também testados nas demais raízes.
"""
        )


if __name__ == "__main__":
    main()
