"""H5AD reading and explicit null-field cleanup utilities."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py


def strip_null_encodings(path: str | Path) -> list[str]:
    """Drop datasets stored as AnnData ``encoding-type=null`` (``None`` values).

    This explicit operation modifies the file in place and removes None-valued fields.
    Apply it to a copy when needed; :func:`read_h5ad` never calls it automatically.
    """
    path = Path(path)
    removed: list[str] = []
    with h5py.File(path, "a") as handle:
        to_delete: list[str] = []

        def _visit(name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            enc = obj.attrs.get("encoding-type")
            if enc in (b"null", "null"):
                to_delete.append(name)

        handle.visititems(_visit)
        for name in sorted(to_delete, key=len, reverse=True):
            del handle[name]
            removed.append(name)
    return removed


def read_h5ad(path: str | Path, **kwargs) -> ad.AnnData:
    """Read H5AD, propagating decoding errors without rewriting the input file."""
    return ad.read_h5ad(Path(path), **kwargs)
