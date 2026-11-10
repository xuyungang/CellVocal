"""Evaluate the CCC recovery quantities used by benchmark figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from .ccc_ground_truth import aligned_ground_truth_events

EventKey = tuple[int, int, int]
PairKey = tuple[int, ...]


def _as_int64(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _as_float64(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _event_keys(
    sender: np.ndarray,
    receiver: np.ndarray,
    module: np.ndarray,
) -> list[EventKey]:
    return [
        (int(s), int(r), int(m))
        for s, r, m in zip(sender, receiver, module, strict=True)
    ]


def _max_scores(
    keys: Iterable[PairKey],
    scores: np.ndarray,
) -> dict[PairKey, float]:
    output: dict[PairKey, float] = {}
    for key, score in zip(keys, _as_float64(scores), strict=True):
        output[key] = max(output.get(key, 0.0), float(score))
    return output


def _sum_scores(
    keys: Iterable[PairKey],
    scores: np.ndarray,
) -> dict[PairKey, float]:
    output: dict[PairKey, float] = {}
    for key, score in zip(keys, _as_float64(scores), strict=True):
        output[key] = output.get(key, 0.0) + float(score)
    return output


def _neighbor_pairs(
    adata: ad.AnnData,
    *,
    spatial_graph_key: str | None,
    spatial_key: str,
    n_neighbors: int,
    max_distance: float | None,
) -> set[tuple[int, int]]:
    if spatial_graph_key is not None:
        graph = adata.obsp[spatial_graph_key].tocsr()
        return {
            (int(sender), receiver)
            for receiver in range(graph.shape[0])
            for sender in graph.indices[graph.indptr[receiver] : graph.indptr[receiver + 1]]
            if int(sender) != receiver
        }

    coords = _as_float64(adata.obsm[spatial_key])
    k = min(max(1, int(n_neighbors)), coords.shape[0])
    distances, indices = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    return {
        (int(sender), receiver)
        for receiver, (row_distances, row_indices) in enumerate(
            zip(distances, indices, strict=True)
        )
        for distance, sender in zip(row_distances, row_indices, strict=True)
        if int(sender) != receiver
        and (max_distance is None or float(distance) <= max_distance)
    }


def _in_neighborhood(
    sender: np.ndarray,
    receiver: np.ndarray,
    pairs: set[tuple[int, int]],
) -> np.ndarray:
    return np.fromiter(
        ((int(s), int(r)) in pairs for s, r in zip(sender, receiver, strict=True)),
        dtype=bool,
        count=len(sender),
    )


@dataclass(frozen=True)
class LevelCurves:
    roc_x: np.ndarray
    roc_y: np.ndarray


def _level_curves(y_true: np.ndarray, y_score: np.ndarray) -> LevelCurves:
    y_true = _as_int64(y_true)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise ValueError("ROC requires both positive and negative examples.")
    x, y, _ = roc_curve(y_true, _as_float64(y_score))
    return LevelCurves(x.astype(np.float64), y.astype(np.float64))


def _aggregate_levels(
    *,
    gt_sender: np.ndarray,
    gt_receiver: np.ndarray,
    gt_module: np.ndarray,
    gt_label: np.ndarray,
    pred_sender: np.ndarray,
    pred_receiver: np.ndarray,
    pred_module: np.ndarray,
    pred_score: np.ndarray,
    pred_receiver_module_score: np.ndarray,
    neighbor_pairs: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_keep = _in_neighborhood(gt_sender, gt_receiver, neighbor_pairs)
    pred_keep = _in_neighborhood(pred_sender, pred_receiver, neighbor_pairs)

    module_positives = {
        (int(receiver), int(module))
        for receiver, module, label, keep in zip(
            gt_receiver,
            gt_module,
            gt_label,
            gt_keep,
            strict=True,
        )
        if keep and label
    }
    module_keys = sorted(
        {
            (int(receiver), int(module))
            for receiver, module, keep in zip(
                gt_receiver,
                gt_module,
                gt_keep,
                strict=True,
            )
            if keep
        }
        | {
            (int(receiver), int(module))
            for receiver, module, keep in zip(
                pred_receiver,
                pred_module,
                pred_keep,
                strict=True,
            )
            if keep
        }
    )
    module_scores = _max_scores(
        (
            (int(receiver), int(module))
            for receiver, module, keep in zip(
                pred_receiver,
                pred_module,
                pred_keep,
                strict=True,
            )
            if keep
        ),
        pred_receiver_module_score[pred_keep],
    )

    pair_positives = {
        (int(sender), int(receiver))
        for sender, receiver, label, keep in zip(
            gt_sender,
            gt_receiver,
            gt_label,
            gt_keep,
            strict=True,
        )
        if keep and label
    }
    pair_keys = sorted(
        {
            (int(sender), int(receiver))
            for sender, receiver, keep in zip(
                gt_sender,
                gt_receiver,
                gt_keep,
                strict=True,
            )
            if keep
        }
        | {
            (int(sender), int(receiver))
            for sender, receiver, keep in zip(
                pred_sender,
                pred_receiver,
                pred_keep,
                strict=True,
            )
            if keep
        }
    )
    pair_scores = _sum_scores(
        (
            (int(sender), int(receiver))
            for sender, receiver, keep in zip(
                pred_sender,
                pred_receiver,
                pred_keep,
                strict=True,
            )
            if keep
        ),
        pred_score[pred_keep],
    )

    return (
        _as_int64([key in module_positives for key in module_keys]),
        _as_float64([module_scores.get(key, 0.0) for key in module_keys]),
        _as_int64([key in pair_positives for key in pair_keys]),
        _as_float64([pair_scores.get(key, 0.0) for key in pair_keys]),
    )


def _topk_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_positives: int,
) -> pd.DataFrame:
    y_true = _as_int64(y_true)
    y_score = _as_float64(y_score)
    if y_true.size == 0:
        raise ValueError("CCC recovery requires at least one predicted event.")
    if n_positives <= 0:
        raise ValueError("CCC recovery requires at least one positive event.")
    order = np.argsort(-y_score, kind="stable")
    prevalence = float(y_true.mean())
    k_values = sorted(
        {
            max(1, int(np.ceil(fraction * y_true.size)))
            for fraction in CCCRecoveryEvaluator.TOP_FRACTIONS
        }
    )
    rows = []
    for k in k_values:
        hits = int(y_true[order[:k]].sum())
        precision = hits / k
        rows.append(
            {
                "top_percent": round(100.0 * k / y_true.size, 2),
                "recall": hits / n_positives,
                "precision_lift": (
                    precision / prevalence if prevalence > 0.0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CCCRecoveryResult:
    event_curves: LevelCurves
    module_curves: LevelCurves
    sender_receiver_curves: LevelCurves
    topk: pd.DataFrame

    def roc_table(self) -> pd.DataFrame:
        rows = []
        for level, curves in (
            ("event", self.event_curves),
            ("module", self.module_curves),
            ("sender_receiver", self.sender_receiver_curves),
        ):
            rows.extend(
                {
                    "level": level,
                    "point_index": index,
                    "fpr": float(fpr),
                    "tpr": float(tpr),
                }
                for index, (fpr, tpr) in enumerate(
                    zip(curves.roc_x, curves.roc_y, strict=True)
                )
            )
        return pd.DataFrame(rows)

    def topk_table(self) -> pd.DataFrame:
        return self.topk.copy()


def _module_indices(
    pred: Mapping[str, Any],
    module_index: Mapping[str, int],
) -> np.ndarray:
    compact_to_gt = _as_int64(
        [module_index[str(module)] for module in pred["lr_pair_module_ids"]]
    )
    return compact_to_gt[_as_int64(pred["module"])]


def ccc_event_auprc_from_anndata(
    adata: ad.AnnData,
    *,
    pred_prefix: str = "cv",
    gt_key: str = "ground_truth",
) -> float:
    """Compute event-level CCC AUPRC using aligned simulation events."""
    ground_truth = adata.uns[gt_key]
    prediction = adata.uns[f"{pred_prefix}_ccc"]
    module_index = {
        str(module): index
        for index, module in enumerate(ground_truth["lr_pairs"]["module_id"])
    }
    events = aligned_ground_truth_events(adata, ground_truth)
    prediction_map = _max_scores(
        _event_keys(
            _as_int64(prediction["sender"]),
            _as_int64(prediction["receiver"]),
            _module_indices(prediction, module_index),
        ),
        _as_float64(prediction["event_influence"]),
    )
    truth_keys = _event_keys(
        _as_int64(events["sender_id"]),
        _as_int64(events["receiver_id"]),
        _as_int64([module_index[str(module)] for module in events["module_id"]]),
    )
    truth_labels = _as_int64(np.asarray(events["label"], dtype=object) == "positive")
    if not truth_labels.any():
        raise ValueError("Event AUPRC requires at least one positive event.")
    scores = _as_float64([prediction_map.get(key, 0.0) for key in truth_keys])
    return float(average_precision_score(truth_labels, scores))


class CCCRecoveryEvaluator:
    """Build ROC and top-fraction recovery data used by the benchmark report."""

    TOP_FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        *,
        pred_prefix: str = "cv",
        gt_key: str = "ground_truth",
        spatial_key: str = "spatial",
        spatial_graph_key: str | None = "spatial_connectivities",
        n_neighbors: int = 12,
        max_distance: float | None = None,
    ) -> CCCRecoveryResult:
        ground_truth = adata.uns[gt_key]
        prediction = adata.uns[f"{pred_prefix}_ccc"]
        module_index = {
            str(module): index
            for index, module in enumerate(ground_truth["lr_pairs"]["module_id"])
        }
        events = aligned_ground_truth_events(adata, ground_truth)
        return cls.evaluate(
            gt_sender=_as_int64(events["sender_id"]),
            gt_receiver=_as_int64(events["receiver_id"]),
            gt_module=_as_int64(
                [module_index[str(module)] for module in events["module_id"]]
            ),
            gt_label=np.asarray(events["label"], dtype=object) == "positive",
            pred_sender=_as_int64(prediction["sender"]),
            pred_receiver=_as_int64(prediction["receiver"]),
            pred_module=_module_indices(prediction, module_index),
            pred_score=_as_float64(prediction["event_influence"]),
            pred_receiver_module_score=_as_float64(prediction["receptor_expr"]),
            neighbor_pairs=_neighbor_pairs(
                adata,
                spatial_graph_key=spatial_graph_key,
                spatial_key=spatial_key,
                n_neighbors=n_neighbors,
                max_distance=max_distance,
            ),
        )

    @classmethod
    def evaluate(
        cls,
        *,
        gt_sender: np.ndarray,
        gt_receiver: np.ndarray,
        gt_module: np.ndarray,
        gt_label: np.ndarray,
        pred_sender: np.ndarray,
        pred_receiver: np.ndarray,
        pred_module: np.ndarray,
        pred_score: np.ndarray,
        pred_receiver_module_score: np.ndarray,
        neighbor_pairs: set[tuple[int, int]],
    ) -> CCCRecoveryResult:
        gt_label = np.asarray(gt_label, dtype=bool)
        prediction_map = _max_scores(
            _event_keys(pred_sender, pred_receiver, pred_module),
            pred_score,
        )
        truth_keys = _event_keys(gt_sender, gt_receiver, gt_module)
        event_scores = _as_float64(
            [prediction_map.get(key, 0.0) for key in truth_keys]
        )
        event_labels = gt_label.astype(np.int64)
        positive_keys = {
            key for key, positive in zip(truth_keys, gt_label, strict=True) if positive
        }
        prediction_labels = _as_int64(
            [
                key in positive_keys
                for key in _event_keys(pred_sender, pred_receiver, pred_module)
            ]
        )
        (
            module_labels,
            module_scores,
            pair_labels,
            pair_scores,
        ) = _aggregate_levels(
            gt_sender=gt_sender,
            gt_receiver=gt_receiver,
            gt_module=gt_module,
            gt_label=gt_label,
            pred_sender=pred_sender,
            pred_receiver=pred_receiver,
            pred_module=pred_module,
            pred_score=pred_score,
            pred_receiver_module_score=pred_receiver_module_score,
            neighbor_pairs=neighbor_pairs,
        )
        return CCCRecoveryResult(
            event_curves=_level_curves(event_labels, event_scores),
            module_curves=_level_curves(module_labels, module_scores),
            sender_receiver_curves=_level_curves(pair_labels, pair_scores),
            topk=_topk_table(
                prediction_labels,
                pred_score,
                n_positives=len(positive_keys),
            ),
        )

    def save(self, result: CCCRecoveryResult, path: str | Path) -> Path:
        target = Path(path)
        output_dir, prefix = (
            (target.parent, target.stem)
            if target.suffix.lower() == ".csv"
            else (target, "ccc_recovery")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result.roc_table().to_csv(output_dir / f"{prefix}_roc.csv", index=False)
        result.topk_table().to_csv(output_dir / f"{prefix}_topk.csv", index=False)
        return output_dir
