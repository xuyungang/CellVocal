"""Ground-truth CCC alignment helpers for filtered AnnData objects."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anndata as ad
import numpy as np


def _current_obs_lookup(adata: ad.AnnData) -> dict[int, int] | None:
    qc = adata.uns.get("cellvocal_qc")
    if isinstance(qc, Mapping) and "kept_obs_indices" in qc:
        kept = np.asarray(qc["kept_obs_indices"], dtype=np.int64)
        if kept.shape[0] == adata.n_obs:
            return {int(old): int(new) for new, old in enumerate(kept)}

    lookup: dict[int, int] = {}
    for new_idx, obs_name in enumerate(adata.obs_names):
        text = str(obs_name)
        try:
            old_idx = int(text)
        except ValueError:
            return None
        lookup[old_idx] = int(new_idx)
    return lookup if len(lookup) == adata.n_obs else None


def aligned_ground_truth_events(
    adata: ad.AnnData,
    gt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return ground-truth events indexed in the current AnnData coordinate system."""
    events = gt["events"]
    if "qc_obs_remap" in gt:
        return events

    sender = np.asarray(events["sender_id"], dtype=np.int64)
    receiver = np.asarray(events["receiver_id"], dtype=np.int64)
    if sender.size == 0:
        return events
    if int(sender.max()) < adata.n_obs and int(receiver.max()) < adata.n_obs:
        return events

    lookup = _current_obs_lookup(adata)
    if lookup is None:
        return events

    mapped_sender = np.fromiter(
        (lookup.get(int(i), -1) for i in sender),
        dtype=np.int64,
        count=sender.size,
    )
    mapped_receiver = np.fromiter(
        (lookup.get(int(i), -1) for i in receiver),
        dtype=np.int64,
        count=receiver.size,
    )
    keep = (mapped_sender >= 0) & (mapped_receiver >= 0)

    out: dict[str, Any] = {}
    n_events = sender.shape[0]
    for name, value in events.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n_events,):
            arr = arr[keep]
        out[name] = arr
    out["sender_id"] = mapped_sender[keep]
    out["receiver_id"] = mapped_receiver[keep]
    return out
