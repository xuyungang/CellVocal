"""UMAP plots and partition metrics for Translator event representations."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.train.config import CellVOCALConfig
from utils.metric.ccc_ground_truth import aligned_ground_truth_events
from utils.metric.translator_partition import (
    balanced_sample_indices,
    compute_partition_metrics,
    metric_representation,
)
from utils.tools.clustering import ClusteringUtils, active_base_seed


class TranslatorUmapTransfer:
    """Build cell-level and event-level UMAPs from Translator features."""

    _NATURE_RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }

    _NATURE_PALETTE = np.array(
        [
            "#0057B8",
            "#E64A19",
            "#00843D",
            "#C62828",
            "#6A1B9A",
            "#00838F",
            "#AD1457",
            "#5D4037",
            "#455A64",
            "#F57C00",
            "#2E7D32",
            "#283593",
            "#7B1FA2",
            "#0277BD",
            "#D84315",
            "#00695C",
            "#9E9D24",
            "#4E342E",
            "#37474F",
            "#C2185B",
        ]
    )
    _LAYOUT_CACHE_VERSION = 3

    DEFAULT_EVENT_LABELS = (
        "module_label",
        "gt_concept",
    )

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        prefix: str = "cv",
        n_neighbors: int = 30,
        n_pcs: int = 30,
        n_clusters: Optional[int] = None,
        dpi: int = 300,
        seed: int | None = None,
        plot_max_events: int = 0,
        metric_max_events: int = 30_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.n_neighbors = int(n_neighbors)
        self.n_pcs = int(n_pcs)
        self.n_clusters = n_clusters
        self.dpi = dpi
        self.seed = active_base_seed() if seed is None else int(seed)
        self.plot_max_events = int(plot_max_events)
        self.metric_max_events = int(metric_max_events)

    @classmethod
    def from_config(
        cls,
        config: CellVOCALConfig,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
        n_neighbors: int | None = None,
        n_pcs: int | None = None,
        n_clusters: Optional[int] = None,
        dpi: int | None = None,
        seed: int | None = None,
    ) -> TranslatorUmapTransfer:
        return cls(
            output_dir or config.eval.output_dir / "translator",
            prefix=tag or config.eval.velocity_key,
            n_neighbors=(
                config.eval.umap_n_neighbors if n_neighbors is None else n_neighbors
            ),
            n_pcs=config.eval.umap_n_pcs if n_pcs is None else n_pcs,
            n_clusters=n_clusters,
            dpi=config.eval.translator_plot_dpi if dpi is None else dpi,
            seed=seed if seed is not None else config.eval.seed,
            plot_max_events=config.eval.translator_plot_max_events,
            metric_max_events=config.eval.translator_metric_max_events,
        )

    @staticmethod
    def _ccc(adata: ad.AnnData, prefix: str) -> dict:
        key = f"{prefix}_ccc"
        ccc = adata.uns.get(key)
        if ccc is None:
            raise KeyError(f"Missing adata.uns[{key!r}]")
        return ccc

    @staticmethod
    def pool_events_to_cells(
        n_obs: int,
        receiver: np.ndarray,
        event_influence: np.ndarray,
        event_feat: np.ndarray,
    ) -> np.ndarray:
        acc = np.zeros((n_obs, event_feat.shape[1]), dtype=np.float64)
        wt = np.zeros(n_obs, dtype=np.float64)

        np.add.at(acc, receiver, event_influence[:, None] * event_feat)
        np.add.at(wt, receiver, event_influence)

        cell_feat = np.zeros((n_obs, event_feat.shape[1]), dtype=np.float32)
        ok = wt > 0
        cell_feat[ok] = (acc[ok] / wt[ok, None]).astype(np.float32)

        if not ok.all():
            cell_feat[~ok] = event_feat.mean(axis=0).astype(np.float32)

        return cell_feat

    @staticmethod
    def _as_feature_matrix(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"features must be a 2D matrix, got shape {x.shape}")
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _cell_features(self, adata: ad.AnnData, event_feat: np.ndarray) -> np.ndarray:
        ccc = self._ccc(adata, self.prefix)
        receiver = np.asarray(ccc["receiver"], dtype=np.int64)
        event_influence = np.asarray(
            ccc.get("event_influence", np.ones(receiver.size, dtype=np.float32)),
            dtype=np.float64,
        )
        return self.pool_events_to_cells(
            adata.n_obs,
            receiver,
            event_influence,
            self._as_feature_matrix(event_feat),
        )

    def _event_embeddings(self, adata: ad.AnnData) -> np.ndarray:
        ccc = self._ccc(adata, self.prefix)
        if "event_semantics" not in ccc:
            raise KeyError(
                f"Missing uns['{self.prefix}_ccc']['event_semantics']; "
                "run CellVOCAL with translator enabled."
            )
        return self._as_feature_matrix(np.asarray(ccc["event_semantics"], dtype=np.float32))

    def _umap_embed(
        self,
        features: np.ndarray,
        obs_names,
        *,
        n_neighbors: int,
        n_pcs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self._as_feature_matrix(features)
        try:
            xy, labels = ClusteringUtils.cluster_and_umap(
                features,
                obs_names,
                n_clusters=self.n_clusters,
                n_neighbors=n_neighbors,
                n_comps=n_pcs,
            )
        except ValueError as exc:
            print(f"[warn] UMAP clustering failed ({exc}); falling back to PCA layout")
            xy = metric_representation(
                features, n_pcs=self.n_pcs, seed=self.seed
            )[:, :2]
            if xy.shape[1] < 2:
                xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
            labels = np.zeros(features.shape[0], dtype=int)
        return xy, labels.astype(str)

    def _layout_cache_key(
        self,
        features: np.ndarray,
    ) -> str:
        """Fingerprint embeddings and layout parameters for safe cache reuse."""
        x = np.ascontiguousarray(features, dtype=np.float32)
        digest = hashlib.blake2b(digest_size=20)
        digest.update(str(self._LAYOUT_CACHE_VERSION).encode())
        digest.update(str(x.shape).encode())
        digest.update(memoryview(x).cast("B"))
        digest.update(
            repr(
                (
                    self.n_neighbors,
                    self.n_pcs,
                    self.n_clusters,
                    self.seed,
                )
            ).encode()
        )
        return digest.hexdigest()

    def _event_layouts(
        self,
        features: np.ndarray,
        event_names: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a cached event-sample layout or compute and cache it."""
        cache_path = self.output_dir / "event_layout_cache.npz"
        cache_key = self._layout_cache_key(features)
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cached_key = str(cached["cache_key"].item())
                    if cached_key == cache_key:
                        xy = np.asarray(cached["xy"], dtype=np.float32)
                        louvain = np.asarray(cached["louvain"], dtype=str)
                        n = features.shape[0]
                        if (
                            xy.shape == (n, 2)
                            and louvain.shape == (n,)
                        ):
                            print(f"[translator] reuse event layout cache: {cache_path}")
                            return xy, louvain
            except (OSError, KeyError, ValueError) as exc:
                print(f"[warn] ignoring invalid event layout cache ({exc})")

        started = time.perf_counter()
        xy, louvain = self._umap_embed(
            features,
            event_names,
            n_neighbors=self.n_neighbors,
            n_pcs=self.n_pcs,
        )
        np.savez(
            cache_path,
            cache_key=np.asarray(cache_key),
            xy=np.asarray(xy, dtype=np.float32),
            louvain=np.asarray(louvain, dtype=str),
        )
        elapsed = time.perf_counter() - started
        print(
            f"[translator] computed event layout in {elapsed:.1f}s; "
            f"cached at {cache_path}"
        )
        return xy, louvain

    def _save_umap(
        self,
        xy: np.ndarray,
        labels: np.ndarray,
        path: Path,
        title: str,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        codes, uniques = pd.factorize(labels.astype(str))
        colors = self._NATURE_PALETTE[codes % len(self._NATURE_PALETTE)]
        title = title.replace(" UMAP", "")
        point_size, point_alpha = self._scatter_style(xy.shape[0])
        order = np.random.default_rng(self.seed).permutation(xy.shape[0])

        with plt.rc_context(self._NATURE_RC):
            fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor="white")

            ax.scatter(
                xy[order, 0],
                xy[order, 1],
                c=colors[order],
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                antialiaseds=False,
                rasterized=True,
            )

            ax.set_title(f"{title} | Louvain, n={len(uniques)}", loc="left", pad=4)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.margins(0.02)

            for spine in ax.spines.values():
                spine.set_visible(False)

            fig.savefig(path, dpi=self.dpi, facecolor="white", transparent=False)
            plt.close(fig)

        print(f"[plot] {path}")
        return path

    def plot(self, adata: ad.AnnData) -> Path:
        """Write a cell-level UMAP from pooled Translator event embeddings."""
        xy, labels = self._umap_embed(
            self._cell_features(adata, self._event_embeddings(adata)),
            adata.obs_names,
            n_neighbors=self.n_neighbors,
            n_pcs=self.n_pcs,
        )
        return self._save_umap(
            xy,
            labels,
            self.output_dir / "translator.png",
            "Translator UMAP",
        )

    @staticmethod
    def _compact_module_metadata(ccc: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        module = np.asarray(ccc["module"], dtype=np.int64)
        lr_pairs = np.asarray(ccc.get("lr_pairs", []), dtype=str)
        module_ids = np.asarray(ccc.get("lr_pair_module_ids", []), dtype=str)

        label = np.asarray([f"module_{m}" for m in module], dtype=object)
        module_id = label.copy()
        valid_label = (module >= 0) & (module < lr_pairs.size)
        valid_id = (module >= 0) & (module < module_ids.size)
        if lr_pairs.size:
            label[valid_label] = lr_pairs[module[valid_label]]
        if module_ids.size:
            module_id[valid_id] = module_ids[module[valid_id]]
        return module_id.astype(str), label.astype(str)

    @staticmethod
    def _ground_truth_maps(
        adata: ad.AnnData,
        *,
        gt_key: str,
    ) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, dict[str, Any]]]:
        gt = adata.uns.get(gt_key) or adata.uns.get("cellvocal_ground_truth")
        if gt is None:
            return {}, {}

        events = aligned_ground_truth_events(adata, gt)
        modules_raw = gt.get("modules", {})
        modules = {
            str(k): dict(v)
            for k, v in modules_raw.items()
            if isinstance(v, Mapping)
        } if isinstance(modules_raw, Mapping) else {}

        labels = np.asarray(events.get("label", []), dtype=object)
        sender = np.asarray(events.get("sender_id", []), dtype=np.int64)
        receiver = np.asarray(events.get("receiver_id", []), dtype=np.int64)
        module = np.asarray(events.get("module_id", []), dtype=object)
        negative = np.asarray(
            events.get("negative_type", np.full(sender.size, "", dtype=object)),
            dtype=object,
        )
        simulated_event_influence = np.asarray(
            events.get("simulated_event_influence", np.full(sender.size, np.nan)),
            dtype=np.float64,
        )

        out: dict[tuple[int, int, str], dict[str, Any]] = {}
        for i, (s, r, m) in enumerate(zip(sender, receiver, module, strict=True)):
            module_id = str(m)
            label = str(labels[i]) if i < labels.size else ""
            neg = str(negative[i]) if i < negative.size else ""
            mod = modules.get(module_id, {})
            if label == "positive":
                event_type = "positive"
            elif neg:
                event_type = neg
            elif label:
                event_type = label
            else:
                event_type = "ground_truth_event"
            out[(int(s), int(r), module_id)] = {
                "gt_label": label,
                "gt_event_type": event_type,
                "simulated_event_influence": float(simulated_event_influence[i]),
                "gt_concept": str(mod.get("concept_name", "")),
                "gt_range_type": str(mod.get("range_type", "")),
                "gt_module_name": str(mod.get("module_name", "")),
            }
        return out, modules

    def event_metadata(
        self,
        adata: ad.AnnData,
        *,
        gt_key: str = "ground_truth",
    ) -> pd.DataFrame:
        """Return one metadata row per predicted candidate event."""
        ccc = self._ccc(adata, self.prefix)
        sender = np.asarray(ccc["sender"], dtype=np.int64)
        receiver = np.asarray(ccc["receiver"], dtype=np.int64)
        module = np.asarray(ccc["module"], dtype=np.int64)
        module_id, module_label = self._compact_module_metadata(ccc)
        gt_map, modules = self._ground_truth_maps(adata, gt_key=gt_key)

        n = sender.size
        rows: dict[str, Any] = {
            "event_id": np.arange(n, dtype=np.int64),
            "sender": sender,
            "receiver": receiver,
            "module": module,
            "module_id": module_id,
            "module_label": module_label,
        }
        for name in (
            "event_influence",
            "candidate_score",
            "lr_compatibility",
            "module_activation_prob",
            "sender_share",
            "ligand_expr",
            "receptor_expr",
        ):
            if name in ccc:
                rows[name] = np.asarray(ccc[name])

        gt_label = np.full(n, "unmatched_candidate", dtype=object)
        gt_event_type = np.full(n, "unmatched_candidate", dtype=object)
        gt_concept = np.empty(n, dtype=object)
        gt_range_type = np.empty(n, dtype=object)
        gt_module_name = np.empty(n, dtype=object)
        simulated_event_influence = np.full(n, np.nan, dtype=np.float64)

        for i, (s, r, m) in enumerate(zip(sender, receiver, module_id, strict=True)):
            info = gt_map.get((int(s), int(r), str(m)))
            mod = modules.get(str(m), {})
            gt_concept[i] = str(mod.get("concept_name", ""))
            gt_range_type[i] = str(mod.get("range_type", ""))
            gt_module_name[i] = str(mod.get("module_name", ""))
            if info is None:
                continue
            gt_label[i] = info["gt_label"] or "ground_truth_event"
            gt_event_type[i] = info["gt_event_type"]
            simulated_event_influence[i] = info["simulated_event_influence"]
            if info["gt_concept"]:
                gt_concept[i] = info["gt_concept"]
            if info["gt_range_type"]:
                gt_range_type[i] = info["gt_range_type"]
            if info["gt_module_name"]:
                gt_module_name[i] = info["gt_module_name"]

        rows["gt_label"] = gt_label.astype(str)
        rows["gt_event_type"] = gt_event_type.astype(str)
        rows["gt_concept"] = pd.Series(gt_concept).replace("", "unknown").to_numpy(dtype=str)
        rows["gt_range_type"] = pd.Series(gt_range_type).replace("", "unknown").to_numpy(dtype=str)
        rows["gt_module_name"] = pd.Series(gt_module_name).replace("", "unknown").to_numpy(dtype=str)
        rows["simulated_event_influence"] = simulated_event_influence
        rows["is_gt_positive"] = rows["gt_label"] == "positive"
        return pd.DataFrame(rows)

    @staticmethod
    def _scatter_style(
        n_points: int,
        point_size: float | None = None,
        point_alpha: float | None = None,
    ) -> tuple[float, float]:
        """Scale marker area and opacity to reduce overplotting."""
        n = max(1, int(n_points))
        if point_size is None or float(point_size) <= 0:
            point_size = float(np.clip(18.0 / np.sqrt(n), 0.06, 1.2))
        if point_alpha is None or float(point_alpha) <= 0:
            point_alpha = 1.0
        return float(point_size), float(np.clip(point_alpha, 0.05, 1.0))

    def _save_categorical_event_umap(
        self,
        xy: np.ndarray,
        labels: np.ndarray,
        path: Path,
        title: str,
        *,
        point_size: float | None = None,
        point_alpha: float | None = None,
        max_legend: int = 20,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = labels.astype(str)
        codes, uniques = pd.factorize(labels)
        colors = self._NATURE_PALETTE[codes % len(self._NATURE_PALETTE)]
        point_size, point_alpha = self._scatter_style(
            len(labels),
            point_size,
            point_alpha,
        )
        order = np.random.default_rng(self.seed).permutation(len(labels))

        with plt.rc_context(self._NATURE_RC):
            fig, ax = plt.subplots(figsize=(4.4, 3.8), facecolor="white")
            ax.scatter(
                xy[order, 0],
                xy[order, 1],
                c=colors[order],
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                antialiaseds=False,
                rasterized=True,
            )
            ax.set_title(
                f"{title} | n={len(labels)}, k={len(uniques)}",
                loc="left",
                pad=4,
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.margins(0.02)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if 1 < len(uniques) <= max_legend:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        color=self._NATURE_PALETTE[i % len(self._NATURE_PALETTE)],
                        markersize=4,
                        label=str(label)[:40],
                    )
                    for i, label in enumerate(uniques)
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0,
                    handletextpad=0.3,
                    fontsize=6,
                )
            fig.savefig(path, dpi=self.dpi, facecolor="white", transparent=False)
            plt.close(fig)

        print(f"[plot] {path}")
        return path

    def _save_continuous_event_umap(
        self,
        xy: np.ndarray,
        values: np.ndarray,
        path: Path,
        title: str,
        *,
        point_size: float | None = None,
        point_alpha: float | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            return path
        point_size, point_alpha = self._scatter_style(
            xy.shape[0],
            point_size,
            point_alpha,
        )
        finite_idx = np.flatnonzero(finite)
        finite_idx = finite_idx[np.argsort(values[finite_idx], kind="stable")]

        with plt.rc_context(self._NATURE_RC):
            fig, ax = plt.subplots(figsize=(4.4, 3.8), facecolor="white")
            if (~finite).any():
                ax.scatter(
                    xy[~finite, 0],
                    xy[~finite, 1],
                    c="#D9D9D9",
                    s=point_size,
                    alpha=min(point_alpha, 0.35),
                    linewidths=0,
                    antialiaseds=False,
                    rasterized=True,
                )
            sc = ax.scatter(
                xy[finite_idx, 0],
                xy[finite_idx, 1],
                c=values[finite_idx],
                cmap="viridis",
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                antialiaseds=False,
                rasterized=True,
            )
            ax.set_title(f"{title} | n={xy.shape[0]}", loc="left", pad=4)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.margins(0.02)
            for spine in ax.spines.values():
                spine.set_visible(False)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.outline.set_linewidth(0.4)
            fig.savefig(path, dpi=self.dpi, facecolor="white", transparent=False)
            plt.close(fig)

        print(f"[plot] {path}")
        return path

    def _event_representation(self, adata: ad.AnnData) -> np.ndarray:
        return self._event_embeddings(adata)

    def plot_events(
        self,
        adata: ad.AnnData,
        *,
        plot_max_events: int | None = None,
        metric_max_events: int | None = None,
        gt_key: str = "ground_truth",
        label_columns: Sequence[str] | None = None,
        knn_k: int = 15,
        point_size: float | None = None,
        point_alpha: float | None = None,
    ) -> dict[str, Any]:
        """
        Write event UMAPs, per-event metadata CSVs, and partition metrics.

        Event UMAPs use a reproducible module-balanced sample when
        ``plot_max_events`` is positive. Metric sampling is controlled separately.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stale_plot in self.output_dir.glob("events_*.png"):
            stale_plot.unlink()
        cell_plot = self.output_dir / "translator.png"
        if cell_plot.is_file():
            cell_plot.unlink()
        labels = tuple(label_columns or self.DEFAULT_EVENT_LABELS)
        meta = self.event_metadata(adata, gt_key=gt_key).reset_index(drop=True)
        features = self._event_representation(adata)
        if features.shape[0] != meta.shape[0]:
            raise ValueError(
                "translator feature count does not match event metadata "
                f"({features.shape[0]} vs {meta.shape[0]})"
            )
        n_total = meta.shape[0]
        max_plot = self.plot_max_events if plot_max_events is None else int(plot_max_events)
        sample_labels = (
            meta["module_label"].astype(str).to_numpy()
            if "module_label" in meta
            else np.zeros(n_total, dtype=str)
        )
        sample_idx = balanced_sample_indices(
            sample_labels,
            max_events=max_plot,
            seed=self.seed,
        )
        features = features[sample_idx]
        plot_meta = meta.iloc[sample_idx].copy()
        outputs: dict[str, Any] = {
            "plots": [],
            "metadata": [],
            "metrics": None,
            "n_total_events": n_total,
            "n_plotted_events": int(sample_idx.size),
        }
        print(
            f"[translator] plotting {sample_idx.size}/{n_total} candidate events"
        )
        event_names = plot_meta["event_id"].map(lambda i: f"event_{int(i)}").to_numpy()
        xy, louvain = self._event_layouts(
            features,
            event_names,
        )
        plot_meta["umap_1"] = xy[:, 0]
        plot_meta["umap_2"] = xy[:, 1]
        plot_meta["louvain"] = louvain
        meta["plot_sample"] = False
        meta["umap_1"] = np.nan
        meta["umap_2"] = np.nan
        meta["louvain"] = ""
        meta.loc[sample_idx, "plot_sample"] = True
        meta.loc[sample_idx, "umap_1"] = xy[:, 0]
        meta.loc[sample_idx, "umap_2"] = xy[:, 1]
        meta.loc[sample_idx, "louvain"] = louvain
        metadata_path = self.output_dir / "events_metadata.csv"
        meta.to_csv(metadata_path, index=False)
        outputs["metadata"].append(metadata_path)
        print(f"[table] {metadata_path}")

        for label_col in (*labels, "louvain"):
            if label_col not in plot_meta:
                continue
            values = plot_meta[label_col].astype(str).to_numpy()
            if pd.unique(values).size < 2:
                continue
            path = self.output_dir / f"events_by_{label_col}.png"
            outputs["plots"].append(
                self._save_categorical_event_umap(
                    xy,
                    values,
                    path,
                    f"translator events by {label_col}",
                    point_size=point_size,
                    point_alpha=point_alpha,
                )
            )

        for value_col in ("event_influence",):
            if value_col not in plot_meta:
                continue
            values = pd.to_numeric(plot_meta[value_col], errors="coerce").to_numpy()
            if np.isfinite(values).sum() < 2:
                continue
            path = self.output_dir / f"events_{value_col}.png"
            outputs["plots"].append(
                self._save_continuous_event_umap(
                    xy,
                    values,
                    path,
                    "translator events | inferred signalling influence",
                    point_size=point_size,
                    point_alpha=point_alpha,
                )
            )

        metrics = compute_partition_metrics(
            features,
            plot_meta,
            louvain,
            label_columns=labels,
            max_events=(
                self.metric_max_events
                if metric_max_events is None
                else int(metric_max_events)
            ),
            knn_k=knn_k,
            n_pcs=self.n_pcs,
            seed=self.seed,
        )
        metrics_path = self.output_dir / "event_partition_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs["metrics"] = metrics_path
        print(f"[table] {metrics_path}")

        return outputs

    def plot_report(
        self,
        adata: ad.AnnData,
        *,
        plot_max_events: int | None = None,
        metric_max_events: int | None = None,
        gt_key: str = "ground_truth",
        label_columns: Sequence[str] | None = None,
        knn_k: int = 15,
        point_size: float | None = None,
        point_alpha: float | None = None,
    ) -> dict[str, Any]:
        """Write the default four event-level Translator UMAP diagnostics."""
        return self.plot_events(
            adata,
            plot_max_events=plot_max_events,
            metric_max_events=metric_max_events,
            gt_key=gt_key,
            label_columns=label_columns,
            knn_k=knn_k,
            point_size=point_size,
            point_alpha=point_alpha,
        )

    def run_path(self, path: Union[str, Path]) -> Path:
        return self.plot(ad.read_h5ad(path))

    def run_report_path(self, path: Union[str, Path], **kwargs: Any) -> dict[str, Any]:
        return self.plot_report(ad.read_h5ad(path), **kwargs)
