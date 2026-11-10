"""Metric entry points with lazy optional-dependency imports."""
from __future__ import annotations

from typing import Any

__all__ = [
    "CCCRecoveryEvaluator",
    "CCCRecoveryResult",
    "SPATIAL_KNN_KEY",
    "VelocityEvaluator",
    "balanced_sample_indices",
    "compute_partition_metrics",
    "metric_representation",
    "embed_spatial_velocity",
    "prepare_metrics_adata",
    "spatial_graph",
]


def __getattr__(name: str) -> Any:
    if name in {"CCCRecoveryEvaluator", "CCCRecoveryResult"}:
        from .ccc_recovery import CCCRecoveryEvaluator, CCCRecoveryResult

        return {
            "CCCRecoveryEvaluator": CCCRecoveryEvaluator,
            "CCCRecoveryResult": CCCRecoveryResult,
        }[name]
    if name in {
        "balanced_sample_indices",
        "compute_partition_metrics",
        "metric_representation",
    }:
        from .translator_partition import (
            balanced_sample_indices,
            compute_partition_metrics,
            metric_representation,
        )

        return {
            "balanced_sample_indices": balanced_sample_indices,
            "compute_partition_metrics": compute_partition_metrics,
            "metric_representation": metric_representation,
        }[name]
    if name in {
        "SPATIAL_KNN_KEY",
        "VelocityEvaluator",
        "embed_spatial_velocity",
        "prepare_metrics_adata",
        "spatial_graph",
    }:
        from .velocity import (
            SPATIAL_KNN_KEY,
            VelocityEvaluator,
            embed_spatial_velocity,
            prepare_metrics_adata,
            spatial_graph,
        )

        return {
            "SPATIAL_KNN_KEY": SPATIAL_KNN_KEY,
            "VelocityEvaluator": VelocityEvaluator,
            "embed_spatial_velocity": embed_spatial_velocity,
            "prepare_metrics_adata": prepare_metrics_adata,
            "spatial_graph": spatial_graph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
