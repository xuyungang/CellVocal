"""Ligand-receptor prior construction and expression-aware filtering."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from model.train.config import CellVOCALConfig
from utils.tools.cellchat_tool import cellchatdb_path

_PAIR_COLUMNS = [
    "interaction_name",
    "pathway_name",
    "ligand",
    "receptor",
    "ligand.symbol",
    "receptor.symbol",
    "co_A_receptor",
    "co_I_receptor",
]


def _clean_lr_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _norm_gene(value: Any) -> str:
    return _clean_lr_value(value).upper()


def _split_gene_tokens(value: Any) -> list[str]:
    text = _clean_lr_value(value)
    if not text:
        return []
    text = re.sub(r"[(){}\[\]]", " ", text)
    parts = re.split(r"\s*(?:,|;|\+|/|\||&|\band\b)\s*", text, flags=re.IGNORECASE)
    tokens = [_clean_lr_value(p) for p in parts if _clean_lr_value(p)]
    if len(tokens) == 1 and "_" in tokens[0]:
        tokens = [_clean_lr_value(p) for p in tokens[0].split("_") if _clean_lr_value(p)]
    return tokens


def _resolve_gene_tokens(
    tokens: Sequence[str],
    gene_by_norm: dict[str, str],
    *,
    require_all: bool = True,
) -> list[str]:
    """Resolve gene symbols, requiring a complete complex unless explicitly relaxed."""
    genes: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        gene = gene_by_norm.get(_norm_gene(token))
        if gene is None:
            if require_all:
                return []
            continue
        if gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def _detected_gene_names(
    adata: ad.AnnData,
    *,
    layer: str = "spliced",
    min_expr: float = 0.0,
) -> set[str]:
    x = adata.layers[layer] if layer in adata.layers else adata.X
    if hasattr(x, "toarray"):
        detected = np.asarray((x > min_expr).sum(axis=0)).ravel() > 0
    else:
        detected = np.asarray(x > min_expr).any(axis=0)
    return {str(g) for g in np.asarray(adata.var_names)[detected]}


def _all_expressed(genes: Sequence[str], expressed: set[str]) -> bool:
    return bool(genes) and all(g in expressed for g in genes)


def _pair_genes_from_mapping(
    pair: Dict[str, Any],
    gene_by_norm: dict[str, str],
) -> tuple[list[str], list[str]]:
    ligand_tokens = pair.get("ligand_genes")
    receptor_tokens = pair.get("receptor_genes")
    if ligand_tokens is None:
        ligand_tokens = _split_gene_tokens(pair.get("ligand"))
    if receptor_tokens is None:
        receptor_tokens = _split_gene_tokens(pair.get("receptor"))
    if isinstance(ligand_tokens, str):
        ligand_tokens = _split_gene_tokens(ligand_tokens)
    if isinstance(receptor_tokens, str):
        receptor_tokens = _split_gene_tokens(receptor_tokens)
    return (
        _resolve_gene_tokens(list(ligand_tokens), gene_by_norm),
        _resolve_gene_tokens(list(receptor_tokens), gene_by_norm),
    )


def _cellchat_gene_tokens(row: Dict[str, Any], side: str) -> list[str]:
    """Use primary names for single genes and symbol expansions for complexes."""
    primary = _split_gene_tokens(row.get(side))
    expanded = _split_gene_tokens(row.get(f"{side}.symbol"))
    if len(primary) > 1 or len(expanded) > 1:
        return expanded or primary
    return primary


def _cellchat_pairs(config: CellVOCALConfig, gene_by_norm: dict[str, str]) -> list[dict[str, Any]]:
    path = cellchatdb_path(
        config.translator.cellchatdb_dir,
        config.translator.cellchat_species,
    )
    columns = set(pd.read_csv(path, nrows=0).columns)
    df = pd.read_csv(path, usecols=[c for c in _PAIR_COLUMNS if c in columns])
    pairs: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        row_d = row.to_dict()
        ligand_tokens = _cellchat_gene_tokens(row_d, "ligand")
        receptor_tokens = _cellchat_gene_tokens(row_d, "receptor")
        co_receptor_tokens = (
            _split_gene_tokens(row_d.get("co_A_receptor"))
            + _split_gene_tokens(row_d.get("co_I_receptor"))
        )
        ligand_genes = _resolve_gene_tokens(ligand_tokens, gene_by_norm)
        receptor_genes = _resolve_gene_tokens(receptor_tokens, gene_by_norm)
        if not ligand_genes or not receptor_genes:
            continue
        # Co-receptor annotations can name modifier groups rather than genes.
        # Matched modifiers augment an already complete ligand/receptor pair.
        co_receptor_genes = _resolve_gene_tokens(
            co_receptor_tokens, gene_by_norm, require_all=False
        )
        for gene in co_receptor_genes:
            if gene not in receptor_genes:
                receptor_genes.append(gene)

        interaction = _clean_lr_value(row_d.get("interaction_name"))
        pathway = _clean_lr_value(row_d.get("pathway_name"))
        pairs.append(
            {
                "interaction": interaction,
                "pathway": pathway,
                "ligand": "+".join(ligand_genes) if ligand_genes else _clean_lr_value(row_d.get("ligand")),
                "receptor": "+".join(receptor_genes)
                if receptor_genes
                else _clean_lr_value(row_d.get("receptor")),
                "ligand_genes": ligand_genes,
                "receptor_genes": receptor_genes,
                "cellchat_ligand": _clean_lr_value(row_d.get("ligand")),
                "cellchat_receptor": _clean_lr_value(row_d.get("receptor")),
            }
        )
    return pairs


def prepare_lr_pairs_for_cellvocal(
    adata: ad.AnnData,
    config: CellVOCALConfig,
    lr_pairs: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    layer: str = "spliced",
    min_expr: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Build/filter LR modules for CellVOCAL.

    Supports case-insensitive matching and multi-gene ligand/receptor complexes.
    Complexes are retained only when every required subunit is present and expressed.
    """
    gene_by_norm = {_norm_gene(g): str(g) for g in adata.var_names}
    expressed = _detected_gene_names(adata, layer=layer, min_expr=min_expr)

    source = "provided"
    input_pairs: Sequence[Dict[str, Any]]
    if lr_pairs is None:
        input_pairs = _cellchat_pairs(config, gene_by_norm)
        source = f"CellChatDB:{config.translator.cellchat_species}"
    else:
        input_pairs = lr_pairs

    filtered: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    for pair in input_pairs:
        pair_d = dict(pair)
        ligand_genes, receptor_genes = _pair_genes_from_mapping(pair_d, gene_by_norm)
        if not _all_expressed(ligand_genes, expressed):
            continue
        if not _all_expressed(receptor_genes, expressed):
            continue

        interaction = _clean_lr_value(pair_d.get("interaction")) or (
            "+".join(ligand_genes) + "_" + "+".join(receptor_genes)
        )
        pathway = _clean_lr_value(pair_d.get("pathway"))
        key = (tuple(ligand_genes), tuple(receptor_genes), interaction)
        if key in seen:
            continue
        seen.add(key)
        pair_d.update(
            {
                "interaction": interaction,
                "pathway": pathway,
                "ligand": "+".join(ligand_genes),
                "receptor": "+".join(receptor_genes),
                "ligand_genes": ligand_genes,
                "receptor_genes": receptor_genes,
            }
        )
        filtered.append(pair_d)

    if not filtered:
        raise ValueError(
            "No ligand-receptor pairs matched adata.var_names with expression. "
            f"source={source!r}, species={config.translator.cellchat_species!r}, "
            f"layer={layer!r}"
        )
    return filtered
