"""Resolve the bundled CellChatDB CSV for a species."""
from __future__ import annotations

from pathlib import Path

_SPECIES_FILES = {
    "mouse": "CellChatDB.mouse.csv",
    "human": "CellChatDB.human.csv",
}


def cellchatdb_path(db_dir: Path | str, species: str = "mouse") -> Path:
    species = species.lower()
    if species not in _SPECIES_FILES:
        raise ValueError(f"Unknown species {species!r}; choose from {sorted(_SPECIES_FILES)}")
    path = Path(db_dir) / _SPECIES_FILES[species]
    if not path.is_file():
        raise FileNotFoundError(f"CellChatDB file not found: {path}")
    return path


__all__ = ["cellchatdb_path"]
