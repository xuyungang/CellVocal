"""Save / load CellVOCAL training checkpoints."""
from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional, Union

import torch

from model.cellvocal import CellVOCAL
from model.train.config import (
    CellVOCALConfig,
    CellVOCALEvalConfig,
    CellVOCALGraphConfig,
    CellVOCALModelConfig,
    CellVOCALQCConfig,
    CellVOCALTrainConfig,
    CellVOCALTranslatorConfig,
)

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 13
DEFAULT_CHECKPOINT_NAME = "cellvocal.pt"
DEFAULT_TRANSLATOR_CHECKPOINT_NAME = "translator.pt"

PathLike = Union[str, Path]


def resolve_checkpoint_path(
    checkpoint_dir: PathLike,
    filename: str = DEFAULT_CHECKPOINT_NAME,
) -> Path:
    return Path(checkpoint_dir) / filename


def cellvocal_checkpoint_path(config: CellVOCALConfig) -> Path:
    return resolve_checkpoint_path(config.train.checkpoint_dir)


def _config_to_dict(cfg: CellVOCALConfig) -> dict[str, Any]:
    """Serialize reproducible settings without machine-local runtime paths."""
    d = asdict(cfg)
    d["train"].pop("checkpoint_dir", None)
    d["eval"].pop("output_dir", None)
    d["translator"].pop("cellchatdb_dir", None)
    return d


def _config_section(cls, values: dict[str, Any], *path_fields: str):
    allowed = {item.name for item in fields(cls)}
    kwargs = {key: value for key, value in values.items() if key in allowed}
    for key in path_fields:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    return cls(**kwargs)


def _config_from_dict(d: dict[str, Any]) -> CellVOCALConfig:
    return CellVOCALConfig(
        qc=_config_section(CellVOCALQCConfig, d["qc"]),
        model=_config_section(CellVOCALModelConfig, d["model"]),
        graph=_config_section(CellVOCALGraphConfig, d["graph"]),
        translator=_config_section(
            CellVOCALTranslatorConfig,
            d["translator"],
            "cellchatdb_dir",
        ),
        train=_config_section(CellVOCALTrainConfig, d["train"], "checkpoint_dir"),
        eval=_config_section(CellVOCALEvalConfig, d["eval"], "output_dir"),
    )


def save_cellvocal_checkpoint(
    model: CellVOCAL,
    path: PathLike,
    *,
    config: CellVOCALConfig,
    n_genes: int,
    n_modules: int,
    epoch: Optional[int] = None,
    event_semantics: Optional[Any] = None,
) -> Path:
    """Persist CellVOCAL weights and metadata; translator weights live in their own checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k: v for k, v in model.state_dict().items() if not k.startswith("ccc_translator.")
    }
    payload = {
        "method": "CellVOCAL",
        "version": CHECKPOINT_VERSION,
        "model_state_dict": state_dict,
        "n_genes": n_genes,
        "n_modules": n_modules,
        "config": _config_to_dict(config),
        "epoch": epoch,
        "translator_semantic_dim": model.translator_semantic_dim,
        "event_semantic_dim": model.ccc_head.event_semantic_dim,
        "event_semantics": event_semantics,
    }
    torch.save(payload, path)
    logger.info("Saved CellVOCAL checkpoint → %s", path)
    return path


def load_cellvocal_checkpoint(
    path: PathLike,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> tuple[CellVOCAL, CellVOCALConfig, dict[str, Any]]:
    """
    Restore a :class:`CellVOCAL` model from disk.

    Returns
    -------
    model, config, meta
        ``meta`` contains ``n_genes``, ``n_modules``, ``epoch``, ``version``,
        and cached event-semantic metadata when present.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {version!r} (expected {CHECKPOINT_VERSION})")

    config = _config_from_dict(payload["config"])
    n_genes = int(payload["n_genes"])
    n_modules = int(payload["n_modules"])
    model = CellVOCAL.from_config(n_genes, n_modules, config)

    sd = dict(payload["model_state_dict"])
    # Recreate the optional fitted annotation layer before strict loading.
    annotation_embedding = sd.get("encoder._ann_embed.weight")
    if annotation_embedding is not None:
        model.encoder.set_annotation_vocab(*map(int, annotation_embedding.shape))
    event_semantic_dim = payload.get("event_semantic_dim")
    if event_semantic_dim is not None:
        model.configure_event_semantics(int(event_semantic_dim))

    model.load_state_dict(sd, strict=True)

    meta = {
        "method": payload.get("method", "CellVOCAL"),
        "version": version,
        "n_genes": n_genes,
        "n_modules": n_modules,
        "epoch": payload.get("epoch"),
        "translator_semantic_dim": payload.get("translator_semantic_dim"),
        "event_semantic_dim": event_semantic_dim,
        "event_semantics": payload.get("event_semantics"),
        "path": path,
    }
    logger.info("Loaded CellVOCAL checkpoint ← %s (epoch=%s)", path, meta["epoch"])
    return model, config, meta


TRANSLATOR_CHECKPOINT_VERSION = 9
_TRANSLATOR_WEIGHT_PREFIXES = (
    "component_encoder.",
    "soft_translator.",
    "event_norm.",
    "module_head.",
    "scalar_head.",
    "component_reconstruction_head.",
)


def _translator_weight_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if k.startswith(_TRANSLATOR_WEIGHT_PREFIXES)
    }


def save_translator_checkpoint(
    translator: torch.nn.Module,
    path: PathLike,
    *,
    config: CellVOCALConfig,
    semantic_dim: Optional[int] = None,
    epoch: Optional[int] = None,
) -> Path:
    """Save translator weights trained for the current pipeline run."""
    if semantic_dim is None:
        raise ValueError("semantic_dim is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tcfg_snapshot = asdict(config.translator)
    tcfg_snapshot.pop("cellchatdb_dir", None)
    payload = {
        "method": "CellVOCAL",
        "version": TRANSLATOR_CHECKPOINT_VERSION,
        "translator_state_dict": _translator_weight_state_dict(translator.state_dict()),
        "translator_semantic_dim": int(semantic_dim),
        "epoch": epoch,
        "config": tcfg_snapshot,
    }
    torch.save(payload, path)
    logger.info("Saved translator checkpoint → %s", path)
    return path


def load_translator_checkpoint(
    path: PathLike,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    """Load a translator checkpoint payload (does not apply weights to a model)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Translator checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("version")
    if version != TRANSLATOR_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported translator checkpoint version {version!r} "
            f"(expected {TRANSLATOR_CHECKPOINT_VERSION})"
        )
    return payload


def apply_translator_checkpoint(
    translator: torch.nn.Module,
    payload: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load trainable translator weights from ``payload``."""
    sd = payload.get("translator_state_dict")
    if not sd:
        raise ValueError("Translator checkpoint is missing translator_state_dict")

    weights = _translator_weight_state_dict(sd)
    missing, unexpected = translator.load_state_dict(weights, strict=False)
    if strict:
        missing_trainable = [k for k in missing if k.startswith(_TRANSLATOR_WEIGHT_PREFIXES)]
        if missing_trainable or unexpected:
            raise RuntimeError(
                "Failed to load translator checkpoint: "
                f"missing={missing_trainable}, unexpected={unexpected}"
            )

    logger.info(
        "Loaded translator weights from checkpoint (epoch=%s, semantic_dim=%s)",
        payload.get("epoch"),
        payload.get("translator_semantic_dim"),
    )
    return {
        "translator_semantic_dim": payload.get("translator_semantic_dim"),
        "epoch": payload.get("epoch"),
    }
