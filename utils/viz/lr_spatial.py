"""Ligand–receptor spatial communication maps."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, to_hex, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Polygon, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import Voronoi, QhullError, cKDTree

from utils.tools.clustering import active_base_seed


def _normalize_lr_pair_key(key: str) -> Tuple[str, str]:
    text = key.strip()
    if "->" in text:
        lig, rec = text.split("->", 1)
    elif ":" in text:
        lig, rec = text.split(":", 1)
    else:
        raise ValueError(f"Invalid LR pair {key!r}; use 'Lig->Rec' or 'Lig:Rec'")
    lig_norm = lig.strip()
    rec_components = sorted(r.strip() for r in rec.split(":") if r.strip())
    return lig_norm, ":".join(rec_components)


def parse_lr_pairs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    return [_normalize_lr_pair_key(key) for key in specs]


def _module_lr_names(ccc: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve per-module ligand and receptor names from compact labels."""
    if "lr_pairs" not in ccc:
        raise KeyError("CCC table must contain 'lr_pairs'")
    labels = np.asarray(ccc["lr_pairs"], dtype=str)
    ligands = np.empty(labels.shape[0], dtype=object)
    receptors = np.empty(labels.shape[0], dtype=object)
    for index, label in enumerate(labels):
        ligand, receptor = _normalize_lr_pair_key(str(label))
        ligands[index] = ligand
        receptors[index] = receptor
    return ligands, receptors


def build_edge_influence_from_ccc(
    ccc: Mapping[str, object],
    n_obs: int,
    *,
    min_event_influence: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, int], List[Tuple[str, str]]]:
    """
    Convert CellVOCAL ``{prefix}_ccc`` event tables to dense edge influences.

    Each event contributes ``event_influence`` on ``(sender, receiver, lr_pair)``.
    Multiple modules sharing the same ligand–receptor pair are merged with ``max``.
    """
    sender = np.asarray(ccc["sender"], dtype=np.int64)
    receiver = np.asarray(ccc["receiver"], dtype=np.int64)
    module = np.asarray(ccc["module"], dtype=np.int64)
    event_influence = np.asarray(ccc["event_influence"], dtype=np.float64)
    ligands, receptors = _module_lr_names(ccc)

    lr_pairs: List[Tuple[str, str]] = []
    edge_type_map: Dict[str, int] = {}
    module_to_idx: Dict[int, int] = {}

    for mod_idx, (lig, rec) in enumerate(zip(ligands, receptors)):
        lig_s = str(lig).strip()
        rec_s = str(rec).strip()
        key = f"{lig_s}:{rec_s}"
        if key not in edge_type_map:
            edge_type_map[key] = len(lr_pairs)
            lr_pairs.append((lig_s, rec_s))
        module_to_idx[int(mod_idx)] = edge_type_map[key]

    influence = np.zeros((n_obs, n_obs, len(lr_pairs)), dtype=np.float32)
    keep = event_influence > float(min_event_influence)
    for s, r, m, score in zip(sender[keep], receiver[keep], module[keep], event_influence[keep]):
        lr_idx = module_to_idx[int(m)]
        prev = influence[s, r, lr_idx]
        if score > prev:
            influence[s, r, lr_idx] = float(score)
    return influence, edge_type_map, lr_pairs


def select_edges_by_top_fraction(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the highest-scoring positive edges for one LR pair."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores > 0.0
    if not np.any(positive):
        return rows[:0], cols[:0], scores[:0]

    rows_p = np.asarray(rows[positive], dtype=np.int64)
    cols_p = np.asarray(cols[positive], dtype=np.int64)
    scores_p = scores[positive]

    frac = float(top_fraction)
    if frac >= 1.0:
        return rows_p, cols_p, scores_p

    n_keep = max(1, int(np.ceil(frac * scores_p.size)))
    order = np.argsort(-scores_p, kind="stable")
    pick = order[:n_keep]
    return rows_p[pick], cols_p[pick], scores_p[pick]


class LRSpatialPlotter:
    """CellVOCAL adapter for ligand–receptor spatial communication maps."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        prefix: str = "cv",
        spatial_key: str = "spatial",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self._viz = LigandReceptorSpatialVisualizer(spatial_key=spatial_key)

    @staticmethod
    def _ccc(adata: ad.AnnData, prefix: str) -> dict:
        key = f"{prefix}_ccc"
        ccc = adata.uns.get(key)
        if ccc is None:
            raise KeyError(f"Missing adata.uns[{key!r}]")
        return ccc

    def _prepare(
        self,
        adata: ad.AnnData,
        *,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], List[Tuple[str, str]]]:
        pos_edge_influence, edge_type_map, discovered_pairs = build_edge_influence_from_ccc(
            self._ccc(adata, self.prefix),
            adata.n_obs,
            min_event_influence=0.0,
        )
        lr_pairs = discovered_pairs if ligand_receptor_pairs is None else ligand_receptor_pairs
        return pos_edge_influence, edge_type_map, lr_pairs

    def plot(
        self,
        adata: ad.AnnData,
        *,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None,
        top_fraction: float = 0.05,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
    ) -> list[Path]:
        pos_edge_influence, edge_type_map, lr_pairs = self._prepare(
            adata,
            ligand_receptor_pairs=ligand_receptor_pairs,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self._viz.plot_ligand_receptor_spatial_distribution(
            adata,
            ligand_receptor_pairs=lr_pairs,
            edge_type_map=edge_type_map,
            pos_edge_influence=pos_edge_influence,
            top_fraction=top_fraction,
            save_path=str(self.output_dir),
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
        )

    def plot_by_region(
        self,
        adata: ad.AnnData,
        *,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None,
        top_fraction: float = 0.05,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        region_key: str = "clusters",
    ) -> list[Path]:
        pos_edge_influence, edge_type_map, lr_pairs = self._prepare(
            adata,
            ligand_receptor_pairs=ligand_receptor_pairs,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self._viz.plot_ligand_receptor_spatial_distribution_by_region(
            adata,
            ligand_receptor_pairs=lr_pairs,
            edge_type_map=edge_type_map,
            pos_edge_influence=pos_edge_influence,
            top_fraction=top_fraction,
            save_path=str(self.output_dir),
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
            region_key=region_key,
        )


def get_colour_scheme(palette_name: str, num_colours: int) -> List[str]:
    """
    Return a list of `num_colours` hex colors from a matplotlib colormap.
    """
    cmap = plt.get_cmap(palette_name)
    xs = np.linspace(0.0, 1.0, num_colours, endpoint=True)
    return [to_hex(cmap(x), keep_alpha=False) for x in xs]


class LigandReceptorSpatialVisualizer:
    """Spatial communication maps from dense LR edge scores."""

    def __init__(self, *, spatial_key: str = "spatial") -> None:
        self.spatial_key = spatial_key

    def plot_ligand_receptor_spatial_distribution(
        self,
        adata: ad.AnnData,
        *,
        ligand_receptor_pairs: List[Tuple[str, str]],
        edge_type_map: Dict[str, int],
        pos_edge_influence: np.ndarray,
        top_fraction: float = 0.05,
        save_path: str,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
    ) -> list[Path]:
        coords = self._extract_spatial_coordinates(adata)
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        for lig, rec, edges_arr, scores_arr, vmin_used, vmax_used in self._iter_lr_figures(
            coords=coords,
            lr_pairs=ligand_receptor_pairs,
            edge_type_map=edge_type_map,
            pos_edge_influence=np.asarray(pos_edge_influence, dtype=np.float32),
            top_fraction=top_fraction,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
        ):
            safe_name = f"{lig}_{rec}".replace(" ", "_").replace("/", "-").replace(":", "-")
            out_path = save_dir / f"lr_spatial_{safe_name}.svg"
            self._plot_spatial_comm_map(
                coords=coords,
                edges=edges_arr,
                edge_score=scores_arr,
                figsize=figsize or (7.0, 6.6),
                bg_color="white",
                cmap="Reds",
                cmap_min=0.08,
                vmin=vmin_used,
                vmax=vmax_used,
                invert_y=False,
                save=str(out_path),
                dpi=300,
            )
            saved.append(out_path)

        if not saved:
            warnings.warn(
                "No LR spatial figures written: no positive edges for the requested "
                f"pairs (top_fraction={top_fraction}).",
                stacklevel=2,
            )
        return saved

    def plot_ligand_receptor_spatial_distribution_by_region(
        self,
        adata: ad.AnnData,
        *,
        ligand_receptor_pairs: List[Tuple[str, str]],
        edge_type_map: Dict[str, int],
        pos_edge_influence: np.ndarray,
        top_fraction: float = 0.05,
        save_path: str,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        region_key: str = "clusters",
    ) -> list[Path]:
        if region_key not in adata.obs:
            raise ValueError(
                f"region_key {region_key!r} not found in adata.obs; set region_key first."
            )

        coords = self._extract_spatial_coordinates(adata)
        region_labels = adata.obs[region_key].astype(str).to_numpy()
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        for lig, rec, edges_arr, scores_arr, vmin_used, vmax_used in self._iter_lr_figures(
            coords=coords,
            lr_pairs=ligand_receptor_pairs,
            edge_type_map=edge_type_map,
            pos_edge_influence=np.asarray(pos_edge_influence, dtype=np.float32),
            top_fraction=top_fraction,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
        ):
            safe_name = f"{lig}_{rec}".replace(" ", "_").replace("/", "-").replace(":", "-")
            out_path = save_dir / f"lr_spatial_region_{safe_name}.svg"
            self._plot_spatial_comm_map_by_region(
                coords=coords,
                edges=edges_arr,
                edge_score=scores_arr,
                region_labels=region_labels,
                figsize=figsize or (7.0, 6.6),
                bg_color="white",
                vmin=vmin_used,
                vmax=vmax_used,
                invert_y=False,
                save=str(out_path),
                dpi=300,
            )
            saved.append(out_path)

        if not saved:
            warnings.warn(
                "No LR spatial-by-region figures written: no positive edges for the "
                f"requested pairs (top_fraction={top_fraction}).",
                stacklevel=2,
            )
        return saved

    def _extract_spatial_coordinates(self, adata: ad.AnnData) -> np.ndarray:
        key = self.spatial_key
        if key not in adata.obsm:
            raise KeyError(f"Missing obsm[{key!r}]")
        return np.asarray(adata.obsm[key][:, :2], dtype=float)

    def _create_lr_to_edge_idx(
        self,
        edge_type_map: Dict[str, int],
        lr_pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], int]:
        lr_to_edge_idx: Dict[Tuple[str, str], int] = {}
        for lr_name, idx in edge_type_map.items():
            lig, rec = lr_name.split(":", 1)
            lr_pair = (lig.strip(), rec.strip())
            if lr_pair in lr_pairs:
                lr_to_edge_idx[lr_pair] = idx
        return lr_to_edge_idx

    def _iter_lr_figures(
        self,
        *,
        coords: np.ndarray,
        lr_pairs: List[Tuple[str, str]],
        edge_type_map: Dict[str, int],
        pos_edge_influence: np.ndarray,
        top_fraction: float,
        max_outgoing_edges_per_node: int,
        max_incoming_edges_per_node: int,
    ):
        if not lr_pairs:
            raise ValueError("ligand_receptor_pairs is empty")
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

        lr_to_edge_idx = self._create_lr_to_edge_idx(edge_type_map, lr_pairs)
        n_points = coords.shape[0]
        max_deg = max(1, min(max_outgoing_edges_per_node, max_incoming_edges_per_node))

        for lig, rec in lr_pairs:
            if (lig, rec) not in lr_to_edge_idx:
                raise ValueError(f"LR pair {(lig, rec)!r} not found in edge_type_map")

            edge_influence = pos_edge_influence[:, :, lr_to_edge_idx[(lig, rec)]]
            if edge_influence.shape[:2] != (n_points, n_points):
                raise ValueError(
                    f"edge_influence shape {edge_influence.shape[:2]} does not match n_points={n_points}"
                )

            rows, cols = np.triu_indices(n_points, k=1)
            scores_arr = np.maximum(
                edge_influence[rows, cols].astype(np.float64, copy=False),
                edge_influence[cols, rows].astype(np.float64, copy=False),
            )
            rows, cols, scores_arr = select_edges_by_top_fraction(
                rows,
                cols,
                scores_arr,
                top_fraction,
            )
            if scores_arr.size == 0:
                continue

            edges_arr = np.column_stack([rows, cols]).astype(np.int64)
            edges_arr, scores_arr = self._limit_edges_per_node(
                n_nodes=n_points,
                edges=edges_arr,
                edge_score=scores_arr,
                max_degree=max_deg,
            )
            if edges_arr.size == 0:
                continue

            vmin_used = float(np.min(scores_arr))
            vmax_used = float(np.percentile(scores_arr, 95))
            if vmax_used <= vmin_used:
                vmax_used = vmin_used + 1.0

            yield lig, rec, edges_arr, scores_arr, vmin_used, vmax_used

    def _compute_node_score(
        self,
        n_nodes: int,
        edges: np.ndarray,
        edge_score: np.ndarray,
    ) -> np.ndarray:
        """Aggregate edge scores to node-level hotspot score using max incident edge score."""
        node_score = np.zeros(n_nodes, dtype=float)
        for (i, j), s in zip(edges, edge_score):
            if s > node_score[i]:
                node_score[i] = s
            if s > node_score[j]:
                node_score[j] = s
        return node_score

    def _limit_edges_per_node(
        self,
        n_nodes: int,
        edges: np.ndarray,
        edge_score: np.ndarray,
        max_degree: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedily keep highest-scoring undirected edges with a per-node degree cap.

        The algorithm sorts edges by descending score and accepts an edge (i, j)
        only if both nodes currently have degree < max_degree.
        """
        if edges.size == 0 or max_degree <= 0:
            return edges[:0], edge_score[:0]

        order = np.argsort(edge_score)[::-1]
        edges_sorted = edges[order]
        scores_sorted = edge_score[order]

        degree = np.zeros(n_nodes, dtype=int)
        kept_edges: List[Tuple[int, int]] = []
        kept_scores: List[float] = []

        for (i, j), s in zip(edges_sorted, scores_sorted):
            if degree[i] >= max_degree or degree[j] >= max_degree:
                continue
            kept_edges.append((int(i), int(j)))
            kept_scores.append(float(s))
            degree[i] += 1
            degree[j] += 1

        if not kept_edges:
            return edges[:0], edge_score[:0]

        return np.asarray(kept_edges, dtype=int), np.asarray(kept_scores, dtype=float)

    def _voronoi_finite_polygons_2d(
        self,
        vor: "Voronoi",
        radius: Optional[float] = None,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions: List[List[int]] = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

        all_ridges: Dict[int, List[Tuple[int, int, int]]] = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]

            if len(region) == 0:
                new_regions.append([])
                continue

            if all(v >= 0 for v in region):
                new_regions.append(region)
                continue

            ridges = all_ridges.get(p1, [])
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent) + 1e-12
                normal = np.array([-tangent[1], tangent[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            if len(new_region) == 0:
                new_regions.append([])
                continue

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = [v for _, v in sorted(zip(angles, new_region))]
            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def _chaikin_smooth_closed(
        self,
        points: np.ndarray,
        refinements: int = 2,
    ) -> np.ndarray:
        """Chaikin smoothing for closed polygons."""
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return pts

        for _ in range(refinements):
            new_pts = []
            n = pts.shape[0]
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            pts = np.asarray(new_pts, dtype=float)
        return pts

    def _make_one_irregular_cell(
        self,
        seed_point: np.ndarray,
        region_polygon: np.ndarray,
        local_radius: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Convert a Voronoi polygon into a more irregular cell-like polygon.

        Compared with a smoother version:
        - stronger anisotropy;
        - more edge perturbation points;
        - less smoothing to preserve irregularity;
        - extra radial harmonic distortion.
        """
        poly = np.asarray(region_polygon, dtype=float)
        if poly.shape[0] < 3:
            return poly

        seed_point = np.asarray(seed_point, dtype=float)

        # 1) clamp region by local spacing
        vec = poly - seed_point
        dist = np.linalg.norm(vec, axis=1)
        max_r = max(local_radius * 1.30, 1e-6)
        scale = np.minimum(1.0, max_r / (dist + 1e-12))
        poly = seed_point + vec * scale[:, None]

        # 2) global shrink: leave a little gap, but not too much
        shrink = 0.84 + 0.08 * rng.random()
        poly = seed_point + shrink * (poly - seed_point)

        # 3) anisotropic deformation to make shapes less uniform
        theta0 = rng.uniform(0, 2 * np.pi)
        major_axis = np.array([np.cos(theta0), np.sin(theta0)])
        minor_axis = np.array([-major_axis[1], major_axis[0]])
        major_scale = 1.02 + 0.18 * rng.random()
        minor_scale = 0.82 + 0.12 * rng.random()

        vv = poly - seed_point
        proj_major = vv @ major_axis
        proj_minor = vv @ minor_axis
        poly = (
            seed_point
            + np.outer(proj_major * major_scale, major_axis)
            + np.outer(proj_minor * minor_scale, minor_axis)
        )

        # 4) insert more irregular points along edges
        pts = []
        n = poly.shape[0]
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            pts.append(p0)

            edge = p1 - p0
            L = np.linalg.norm(edge) + 1e-12
            tangent = edge / L
            perp = np.array([-tangent[1], tangent[0]])

            # 2 or 3 inner points per edge
            n_inner = 2 + int(rng.random() > 0.45)

            for k in range(1, n_inner + 1):
                t = k / (n_inner + 1)
                base = (1 - t) * p0 + t * p1

                radial = base - seed_point
                radial /= np.linalg.norm(radial) + 1e-12

                amp = local_radius * (0.07 + 0.18 * rng.random())

                side_sign = 1.0 if rng.random() > 0.5 else -1.0
                direction = (
                    (0.75 + 0.20 * rng.random()) * radial
                    + side_sign * (0.18 + 0.35 * rng.random()) * perp
                )
                direction /= np.linalg.norm(direction) + 1e-12

                pts.append(base + amp * direction)

        pts = np.asarray(pts, dtype=float)

        # 5) lighter smoothing: preserve irregular boundaries
        pts = self._chaikin_smooth_closed(pts, refinements=1)

        # 6) extra harmonic radial distortion
        vec = pts - seed_point
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        rr = np.linalg.norm(vec, axis=1)

        f1 = int(rng.integers(3, 6))
        f2 = int(rng.integers(6, 10))
        ph1 = rng.uniform(0, 2 * np.pi)
        ph2 = rng.uniform(0, 2 * np.pi)

        mod = (
            1.0
            + 0.10 * np.sin(f1 * ang + ph1)
            + 0.06 * np.sin(f2 * ang + ph2)
        )
        rr = rr * np.clip(mod, 0.82, 1.22)

        max_final_r = local_radius * 1.36
        rr = np.minimum(rr, max_final_r)

        pts = np.c_[
            seed_point[0] + rr * np.cos(ang),
            seed_point[1] + rr * np.sin(ang),
        ]
        return pts

    def _build_irregular_tissue_patches(
        self,
        coords: np.ndarray,
    ) -> Optional[List[Polygon]]:
        if coords.shape[0] < 4:
            return None

        pts = np.asarray(coords, dtype=float).copy()
        x_range = float(pts[:, 0].max() - pts[:, 0].min())
        y_range = float(pts[:, 1].max() - pts[:, 1].min())
        jitter_scale = max(x_range, y_range, 1.0) * 1e-8
        rng0 = np.random.default_rng(active_base_seed())
        pts += rng0.normal(loc=0.0, scale=jitter_scale, size=pts.shape)

        try:
            vor = Voronoi(pts)
            regions, vertices = self._voronoi_finite_polygons_2d(vor)
        except QhullError:
            return None

        tree = cKDTree(coords)
        k = min(5, len(coords))
        dists, _ = tree.query(coords, k=k)
        if dists.ndim == 1:
            local_radius = np.full(len(coords), np.median(dists) * 0.52)
        else:
            neigh = dists[:, 1:] if dists.shape[1] > 1 else dists
            local_radius = np.median(neigh, axis=1) * 0.52

        if len(local_radius) > 1:
            lo = np.percentile(local_radius, 5)
            hi = np.percentile(local_radius, 95)
            local_radius = np.clip(local_radius, lo, hi)

        patches: List[Polygon] = []
        for idx, region in enumerate(regions):
            rng = np.random.default_rng(active_base_seed() + 1000 + idx)

            if len(region) < 3:
                theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
                rr = local_radius[idx] * (
                    0.72
                    + 0.18 * np.sin(theta * (3 + (idx % 3)) + idx * 0.31)
                    + 0.08 * np.sin(theta * 7 + idx * 0.17)
                )
                poly = np.c_[
                    coords[idx, 0] + rr * np.cos(theta),
                    coords[idx, 1] + rr * np.sin(theta),
                ]
                patches.append(Polygon(poly, closed=True))
                continue

            region_polygon = vertices[region]
            irr_poly = self._make_one_irregular_cell(
                seed_point=coords[idx],
                region_polygon=region_polygon,
                local_radius=float(local_radius[idx]),
                rng=rng,
            )
            patches.append(Polygon(irr_poly, closed=True))

        return patches

    def _truncate_cmap(
        self,
        cmap: Union[str, LinearSegmentedColormap],
        minval: float = 0.08,
        maxval: float = 1.0,
        n: int = 256,
    ) -> LinearSegmentedColormap:
        """
        Truncate a colormap so that the lower end is not pure white.

        For example, cmap='Reds', minval=0.08 means the displayed minimum
        starts from a very light pink rather than white.
        """
        base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        minval = float(np.clip(minval, 0.0, 0.95))
        maxval = float(np.clip(maxval, minval + 1e-6, 1.0))
        colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(
            f"trunc_{base.name}_{minval:.2f}_{maxval:.2f}",
            colors,
        )

    def _style_nature_colorbar(
        self,
        cbar,
        *,
        vmin: float,
        vmax: float,
    ) -> None:
        """
        Style colorbar in a cleaner Nature-like manner:
        - slim bar
        - no outer box
        - few ticks
        - small font
        """
        cbar.outline.set_visible(False)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        if vmax <= vmin:
            ticks = np.array([vmin], dtype=float)
        else:
            ticks = np.linspace(vmin, vmax, 4)

        cbar.set_ticks(ticks)

        value_range = float(vmax - vmin)
        if value_range < 0.1:
            fmt = "%.3f"
        elif value_range < 1:
            fmt = "%.2f"
        elif value_range < 10:
            fmt = "%.1f"
        else:
            fmt = "%.0f"

        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        cbar.ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=2.2,
            width=0.6,
            color="#4d4d4d",
            labelcolor="#4d4d4d",
            labelsize=8,
            pad=2,
        )

    def _draw_curved_edges(
        self,
        ax,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        alpha_min: float = 0.35,
        alpha_max: float = 0.95,
        lw_min: float = 0.22,
        lw_max: float = 1.40,
        curve_scale: float = 0.18,
        zorder: float = 4.0,
    ) -> None:
        """
        Draw thicker black curved edges with quadratic Bézier curves.
        """
        if edges.size == 0:
            return

        order = np.argsort(edge_score)
        edges = edges[order]
        edge_score = edge_score[order]

        seg = coords[edges[:, 1]] - coords[edges[:, 0]]
        seg_len = np.linalg.norm(seg, axis=1)
        positive = seg_len > 1e-12
        median_len = float(np.median(seg_len[positive])) if np.any(positive) else 1.0
        median_len = max(median_len, 1e-6)

        for (i, j), s in zip(edges, edge_score):
            p0 = coords[i]
            p1 = coords[j]
            d = p1 - p0
            L = np.linalg.norm(d)
            if L <= 1e-12:
                continue

            tangent = d / L
            perp = np.array([-tangent[1], tangent[0]])

            sign = 1.0 if (((int(i) * 73856093) ^ (int(j) * 19349663)) & 1) else -1.0

            strength = (np.clip(s, vmin, vmax) - vmin) / max(vmax - vmin, 1e-12)
            len_factor = np.clip(L / median_len, 0.55, 1.55)
            offset = sign * perp * (curve_scale * len_factor * L)

            mid = 0.5 * (p0 + p1)
            control = mid + offset

            lw = lw_min + strength * (lw_max - lw_min)
            alpha = alpha_min + strength * (alpha_max - alpha_min)

            path = MplPath(
                [p0, control, p1],
                [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
            )

            under = PathPatch(
                path,
                facecolor="none",
                edgecolor="black",
                lw=lw * 1.35,
                alpha=alpha * 0.22,
                capstyle="round",
                joinstyle="round",
                zorder=zorder - 0.05,
            )
            ax.add_patch(under)

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor="black",
                lw=lw,
                alpha=alpha,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
            ax.add_patch(patch)

    def _plot_spatial_comm_map(
        self,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        *,
        figsize: Tuple[float, float] = (7, 6.6),
        bg_color: str = "white",
        cmap: str = "Reds",
        cmap_min: float = 0.08,
        vmin: float = 0.0,
        vmax: float = 7.0,
        invert_y: bool = False,
        save: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """
        Draw a spatial communication map with:
        - one different irregular tissue block per point;
        - red hotspot fills;
        - thick black curved lines.

        Key change:
        - the lowest color is no longer pure white;
        - low-score cells are rendered in a very light pink.
        """
        coords = np.asarray(coords, dtype=float)
        edges = np.asarray(edges, dtype=int)
        edge_score = np.asarray(edge_score, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be an array of shape (N, 2)")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be an array of shape (M, 2)")
        if len(edge_score) != len(edges):
            raise ValueError("edge_score length must match number of edges")

        cmap_obj = self._truncate_cmap(cmap, minval=cmap_min, maxval=1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        node_score = self._compute_node_score(len(coords), edges, edge_score)

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)

        patches = self._build_irregular_tissue_patches(coords)

        if patches is not None and len(patches) == len(coords):
            # Keep low values as very light red, do not force them to pure white
            cell_values = np.clip(node_score, vmin, vmax)
            cell_colors = cmap_obj(norm(cell_values))

            pc = PatchCollection(
                patches,
                facecolor=cell_colors,
                edgecolor="#2f2f2f",
                linewidths=0.22,
                antialiaseds=True,
                alpha=0.98,
                zorder=1,
            )
            ax.add_collection(pc)
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=np.clip(node_score, vmin, vmax),
                cmap=cmap_obj,
                norm=norm,
                s=14,
                edgecolors="#2f2f2f",
                linewidths=0.18,
                zorder=1,
            )

        hot_mask = edge_score > vmin
        if np.any(hot_mask):
            self._draw_curved_edges(
                ax=ax,
                coords=coords,
                edges=edges[hot_mask],
                edge_score=edge_score[hot_mask],
                vmin=vmin,
                vmax=vmax,
                alpha_min=0.35,
                alpha_max=0.95,
                lw_min=0.22,
                lw_max=1.0,
                curve_scale=0.2,
                zorder=4.0,
            )

        pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
        pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
        ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        if invert_y:
            ax.invert_yaxis()

        ax.set_aspect("equal")
        ax.axis("off")

        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.032,
            pad=0.020,
            aspect=35,
            shrink=0.96,
        )
        self._style_nature_colorbar(cbar, vmin=vmin, vmax=vmax)

        plt.tight_layout()

        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            if str(save).lower().endswith(".svg"):
                plt.rcParams["svg.fonttype"] = "none"
            plt.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        plt.close(fig)

    def _plot_spatial_comm_map_by_region(
        self,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        region_labels: np.ndarray,
        *,
        figsize: Tuple[float, float] = (7, 6.6),
        bg_color: str = "white",
        vmin: float = 0.0,
        vmax: float = 7.0,
        invert_y: bool = False,
        save: Optional[str] = None,
        dpi: int = 300,
        region_palette: str = "tab20",
    ) -> None:
        """
        Draw spatial communication map with cells colored by region (categorical).

        Same layout and curved edges as _plot_spatial_comm_map; cell fill uses
        one color per region and a legend instead of a score colorbar.
        """
        coords = np.asarray(coords, dtype=float)
        edges = np.asarray(edges, dtype=int)
        edge_score = np.asarray(edge_score, dtype=float)
        region_labels = np.asarray(region_labels)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be an array of shape (N, 2)")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be an array of shape (M, 2)")
        if len(edge_score) != len(edges):
            raise ValueError("edge_score length must match number of edges")
        if len(region_labels) != len(coords):
            raise ValueError("region_labels length must match number of points")

        unique_regions = sorted(np.unique(region_labels).tolist())
        n_regions = len(unique_regions)
        palette = get_colour_scheme(region_palette, max(n_regions, 1))
        region_to_color = {r: palette[i % len(palette)] for i, r in enumerate(unique_regions)}
        cell_colors = np.array([region_to_color[r] for r in region_labels])

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)

        patches = self._build_irregular_tissue_patches(coords)

        if patches is not None and len(patches) == len(coords):
            pc = PatchCollection(
                patches,
                facecolor=cell_colors,
                edgecolor="#2f2f2f",
                linewidths=0.22,
                antialiaseds=True,
                alpha=0.98,
                zorder=1,
            )
            ax.add_collection(pc)
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=cell_colors,
                s=14,
                edgecolors="#2f2f2f",
                linewidths=0.18,
                zorder=1,
            )

        if edges.size > 0 and np.any(edge_score > vmin):
            hot_mask = edge_score > vmin
            self._draw_curved_edges(
                ax=ax,
                coords=coords,
                edges=edges[hot_mask],
                edge_score=edge_score[hot_mask],
                vmin=vmin,
                vmax=vmax,
                alpha_min=0.35,
                alpha_max=0.95,
                lw_min=0.22,
                lw_max=1.0,
                curve_scale=0.2,
                zorder=4.0,
            )

        pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
        pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
        ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        if invert_y:
            ax.invert_yaxis()

        ax.set_aspect("equal")
        ax.axis("off")

        legend_handles = [
            Patch(facecolor=region_to_color[r], edgecolor="#2f2f2f", label=str(r))
            for r in unique_regions
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
        )

        plt.tight_layout()

        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            if str(save).lower().endswith(".svg"):
                plt.rcParams["svg.fonttype"] = "none"
            plt.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        plt.close(fig)
