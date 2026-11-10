"""Shared CCC/LR expression core for simulation panels."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

EVENT_ATTRIBUTION_MODES = (
    "organizer_dominant",
    "collective_redundant",
    "contact_dependent",
    "graded_paracrine",
    "developmental_switch",
)

MODE_DESCRIPTIONS = {
    "organizer_dominant": "One organizer-like sender dominates.",
    "collective_redundant": "Several nearby senders contribute redundantly.",
    "contact_dependent": "Short-range contact-like simulated event influence.",
    "graded_paracrine": "Smooth ligand-distance paracrine simulated event influence.",
    "developmental_switch": "Sender program switches with receiver stage.",
}


def save_h5ad(adata: ad.AnnData, path: str | Path) -> Path:
    """Write h5ad robustly, including non-ASCII target paths on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        adata.write_h5ad(path)
    except OSError:
        if str(path).isascii():
            raise
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            adata.write_h5ad(temporary)
            if path.exists():
                path.unlink()
            shutil.move(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()
    return path


@dataclass(frozen=True)
class SimulationConfig:
    n_cells: int = 3000
    n_genes: int = 500
    n_modules: int = 4
    target_genes_per_module: int = 24
    n_sender_neighbors: int = 90
    max_hard_negative_events: int = 20000
    t_max: float = 20.0
    tissue_length: float = 1.0
    tissue_width: float = 0.28
    developmental_angle: float = np.pi / 4.0
    communication_velocity_weight: float = 0.5
    expression_effect_strength: float = 0.16
    sender_ligand_feedback_strength: float = 0.75
    lr_program_strength: float = 6.0
    lr_program_sharpness: float = 2.5
    lr_program_active_fraction: float = 0.20
    lr_program_background: float = 0.01
    module_receptor_suppression: float = 0.02
    noise_level: float = 0.10
    unspliced_dropout: float = 0.75
    spliced_dropout: float = 0.45
    dropout_expression_protection: float = 2.5
    lr_dropout_scale: float = 0.08
    lr_observation_anchor_strength: float = 0.85
    seed: int = 42


EventRecord = tuple[
    int,
    int,
    str,
    str,
    str,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]

_GROUND_TRUTH_EVENT_FIELDS = (
    "sender_id",
    "receiver_id",
    "module_id",
    "label",
    "negative_type",
    "effect_magnitude",
    "distance",
    "ligand_score",
    "receptor_score",
)


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    lo = np.nanmin(x, axis=0)
    hi = np.nanmax(x, axis=0)
    denom = np.maximum(hi - lo, eps)
    return np.clip((x - lo) / denom, 0.0, 1.0)


def _rotate(coords: np.ndarray, angle: float) -> np.ndarray:
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return coords @ matrix.T


def _state_labels(latent: np.ndarray) -> np.ndarray:
    return np.where(
        latent < 1.0 / 3.0,
        "early",
        np.where(latent < 2.0 / 3.0, "transition", "late"),
    ).astype(object)


def _normalize_vector(values: Any, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.min(arr[finite])
    hi = np.max(arr[finite])
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _assign_regions(coords: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else np.zeros(coords.shape[0])
    x_hi = x >= np.median(x)
    y_hi = y >= np.median(y)
    region_id = x_hi.astype(int) + 2 * y_hi.astype(int)
    return np.array([f"region_{idx}" for idx in region_id], dtype=object)


def _linear_development_tissue(
    n_cells: int,
    t_max: float,
    tissue_length: float,
    tissue_width: float,
    developmental_angle: float,
    rng: np.random.Generator,
    lateral_drift_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent = np.sort(rng.beta(1.8, 1.8, size=n_cells))
    true_time = latent * t_max

    longitudinal_noise = rng.normal(0.0, 0.012 * tissue_length, n_cells)
    transverse_sd = tissue_width * (0.18 + 0.12 * latent)
    transverse = rng.normal(0.0, transverse_sd)
    unrotated = np.column_stack(
        [tissue_length * latent + longitudinal_noise, transverse]
    )
    coords = _rotate(unrotated, developmental_angle).astype(np.float32)

    forward = np.array(
        [np.cos(developmental_angle), np.sin(developmental_angle)],
        dtype=np.float64,
    )
    speed = tissue_length / t_max
    velocity = speed * forward[None, :]
    if lateral_drift_strength > 0.0:
        normal = np.array([-forward[1], forward[0]], dtype=np.float64)
        lateral_drift = (
            lateral_drift_strength
            * speed
            * np.sin(2.0 * np.pi * latent)
            * np.tanh(transverse / max(tissue_width * 0.25, 1e-8))
        )
        velocity = velocity + lateral_drift[:, None] * normal[None, :]
    else:
        velocity = np.repeat(velocity, n_cells, axis=0)

    return (
        true_time.astype(np.float32),
        latent.astype(np.float32),
        coords,
        velocity.astype(np.float32),
        transverse.astype(np.float32),
    )


def _developmental_spatial_pattern(
    latent: np.ndarray,
    coords: np.ndarray,
    n_genes: int,
    rng: np.random.Generator,
    baseline_weight: float,
    temporal_weight: float,
    spatial_weight: float,
    width_high: float,
    spatial_preference_sd: float,
    spatial_center: str = "median",
    monotonic_gene_count: int = 0,
) -> np.ndarray:
    peak = rng.uniform(0.05, 0.95, size=n_genes)
    width = rng.uniform(0.12, width_high, size=n_genes)
    temporal = np.exp(
        -0.5 * ((latent[:, None] - peak[None, :]) / width[None, :]) ** 2
    )

    if monotonic_gene_count > 0:
        n_monotonic = min(int(monotonic_gene_count), n_genes)
        half = n_monotonic // 2
        temporal[:, :half] = 1.0 / (
            1.0 + np.exp((latent[:, None] - peak[None, :half]) / 0.08)
        )
        temporal[:, half:n_monotonic] = 1.0 / (
            1.0 + np.exp(-(latent[:, None] - peak[None, half:n_monotonic]) / 0.08)
        )

    spatial_axis = (
        coords[:, 1] if coords.shape[1] > 1 else np.zeros(coords.shape[0])
    ).astype(np.float64, copy=False)
    if spatial_center == "mean":
        spatial_axis = spatial_axis - float(np.mean(spatial_axis))
    elif spatial_center == "median":
        spatial_axis = spatial_axis - float(np.median(spatial_axis))
    else:
        raise ValueError("spatial_center must be 'mean' or 'median'")
    spatial_axis = spatial_axis / max(float(np.std(spatial_axis)), 1e-8)
    spatial_preference = rng.normal(0.0, spatial_preference_sd, size=n_genes)
    spatial = np.exp(
        -0.5 * (spatial_axis[:, None] - spatial_preference[None, :]) ** 2
    )

    pattern = baseline_weight + temporal_weight * temporal + spatial_weight * spatial
    return np.clip(pattern, 0.02, None).astype(np.float32)


def _nearest_cell(coords: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(coords - point[None, :], axis=1)))


def _spatial_neighbor_candidates(
    coords: np.ndarray,
    n_neighbors: int,
    sender_indices: np.ndarray | None = None,
    receiver_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match CellVOCAL spatial edges: kneighbors_graph(...).maximum(g.T)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = min(int(n_neighbors), max(1, coords.shape[0] - 1))
    graph = kneighbors_graph(
        coords,
        n_neighbors=n,
        mode="distance",
        include_self=False,
    )
    graph = graph.maximum(graph.T).tocsr()

    if sender_indices is None:
        sender_allowed = np.ones(coords.shape[0], dtype=bool)
    else:
        sender_allowed = np.zeros(coords.shape[0], dtype=bool)
        sender_allowed[np.asarray(sender_indices, dtype=np.int64)] = True
    receivers = (
        np.arange(coords.shape[0], dtype=np.int64)
        if receiver_indices is None
        else np.asarray(receiver_indices, dtype=np.int64)
    )

    rows_senders: list[np.ndarray] = []
    rows_distances: list[np.ndarray] = []
    max_degree = 0
    for receiver in receivers:
        start, end = graph.indptr[int(receiver)], graph.indptr[int(receiver) + 1]
        cols = graph.indices[start:end]
        vals = graph.data[start:end]
        keep = sender_allowed[cols] & (cols != int(receiver))
        row_senders = cols[keep].astype(np.int64, copy=False)
        row_distances = vals[keep].astype(np.float32, copy=False)
        if row_senders.size:
            order = np.argsort(row_distances, kind="stable")
            row_senders = row_senders[order]
            row_distances = row_distances[order]
        rows_senders.append(row_senders)
        rows_distances.append(row_distances)
        max_degree = max(max_degree, int(row_senders.size))

    out_senders = np.full((len(rows_senders), max_degree), -1, dtype=np.int64)
    out_distances = np.full((len(rows_senders), max_degree), np.nan, dtype=np.float32)
    for row, (senders, distances) in enumerate(
        zip(rows_senders, rows_distances, strict=True)
    ):
        if senders.size:
            out_senders[row, : senders.size] = senders
            out_distances[row, : distances.size] = distances
    return out_senders, out_distances


def _expression_distance_sender_share(
    ligand: np.ndarray,
    senders: np.ndarray,
    distance: np.ndarray,
    distance_scale: float,
) -> np.ndarray:
    weight = np.exp(-0.5 * (distance / max(distance_scale, 1e-8)) ** 2)
    evidence = np.maximum(ligand[senders], 1e-6) * weight
    total = float(np.sum(evidence))
    if total <= 1e-12:
        return np.full(senders.size, 1.0 / max(senders.size, 1))
    return evidence / total


def _concat_tables(tables: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tables[0].keys()
    return {
        key: np.concatenate([np.asarray(table[key]) for table in tables])
        for key in keys
    }


def _generate_linear_development(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_time, latent, coords, intrinsic_velocity, transverse = (
        _linear_development_tissue(
            config.n_cells,
            config.t_max,
            config.tissue_length,
            config.tissue_width,
            config.developmental_angle,
            rng,
            lateral_drift_strength=0.08,
        )
    )

    states = _state_labels(latent)
    lateral_region = (transverse >= np.median(transverse)).astype(int)
    longitudinal_region = np.digitize(latent, [1.0 / 3.0, 2.0 / 3.0])
    regions = np.asarray(
        [
            f"stage_{stage}_side_{side}"
            for stage, side in zip(longitudinal_region, lateral_region, strict=True)
        ],
        dtype=object,
    )
    return true_time, latent, coords, intrinsic_velocity, states, regions


def _generate_baseline_transcription(
    latent: np.ndarray,
    coords: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_genes = config.n_genes
    alpha0 = rng.lognormal(mean=0.35, sigma=0.55, size=n_genes)
    beta = rng.lognormal(mean=-0.05, sigma=0.30, size=n_genes)
    gamma = rng.lognormal(mean=-0.15, sigma=0.30, size=n_genes)

    n_monotonic = max(2, n_genes // 5)
    rho = _developmental_spatial_pattern(
        latent,
        coords,
        n_genes,
        rng,
        baseline_weight=0.08,
        temporal_weight=0.72,
        spatial_weight=0.20,
        width_high=0.42,
        spatial_preference_sd=0.55,
        spatial_center="median",
        monotonic_gene_count=n_monotonic,
    )
    transcription_drive = (alpha0[None, :] * rho).astype(np.float32)

    ton = rng.uniform(0.0, 0.08 * config.t_max, size=n_genes)
    toff = rng.uniform(0.55 * config.t_max, config.t_max, size=n_genes)
    return transcription_drive, beta, gamma, ton, toff


def _sample_without_overlap(
    rng: np.random.Generator,
    pool: np.ndarray,
    size: int,
    used: set[int],
    n_genes: int,
) -> np.ndarray:
    pool = np.asarray([idx for idx in pool if int(idx) not in used], dtype=int)
    if pool.size < size:
        pool = np.asarray([idx for idx in range(n_genes) if idx not in used], dtype=int)
    if pool.size == 0:
        pool = np.arange(n_genes)
    chosen = np.sort(rng.choice(pool, size=min(size, pool.size), replace=False))
    used.update(int(idx) for idx in chosen)
    return chosen.astype(np.int64)


def _build_modules(
    transcription_drive: np.ndarray,
    latent: np.ndarray,
    states: np.ndarray,
    regions: np.ndarray,
    coords: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
]:
    n_genes = config.n_genes
    n_modules = min(config.n_modules, max(1, n_genes // 3))
    expr = _normalize(transcription_drive)
    mean_expr = expr.mean(axis=0)
    high_pool = np.flatnonzero(mean_expr >= np.quantile(mean_expr, 0.55))
    gene_names = np.asarray([f"G{idx:04d}" for idx in range(n_genes)], dtype=object)
    state_patterns = (
        np.asarray(["early", "transition"], dtype=object),
        np.asarray(["transition"], dtype=object),
        np.asarray(["transition", "late"], dtype=object),
        np.asarray(["late"], dtype=object),
    )
    range_types = ("local", "regional", "long_range")
    all_regions = np.unique(regions.astype(str))
    tissue_scale = max(float(np.linalg.norm(np.ptp(coords, axis=0))), 1e-6)

    modules: dict[str, dict[str, Any]] = {}
    pair_ids: list[str] = []
    pair_modules: list[str] = []
    pair_ligands: list[str] = []
    pair_receptors: list[str] = []
    used_lr: set[int] = set()

    for module_idx in range(n_modules):
        module_id = f"module_{module_idx}"
        compatible_states = state_patterns[module_idx % len(state_patterns)]
        range_type = range_types[module_idx % len(range_types)]
        allowed_regions = np.asarray([all_regions[module_idx % len(all_regions)]], dtype=object)

        state_mask = np.isin(states.astype(str), compatible_states.astype(str))
        region_mask = np.isin(regions.astype(str), allowed_regions.astype(str))
        focus = state_mask if range_type == "long_range" else state_mask & region_mask
        if not np.any(focus):
            focus = state_mask
        background = ~focus
        focus_mean = expr[focus].mean(axis=0)
        background_mean = expr[background].mean(axis=0) if np.any(background) else np.zeros(n_genes)
        receptor_contrast = np.maximum(focus_mean - background_mean, 0.0)
        receptor_pool = np.argsort(receptor_contrast + 0.05 * focus_mean)[
            -max(4, n_modules * 2) :
        ]

        ligand_idx = _sample_without_overlap(rng, high_pool, 1, used_lr, n_genes)
        receptor_idx = _sample_without_overlap(rng, receptor_pool, 1, used_lr, n_genes)
        target_pool = np.asarray([idx for idx in range(n_genes) if idx not in used_lr], dtype=int)
        target_idx = np.sort(
            rng.choice(
                target_pool,
                size=min(config.target_genes_per_module, target_pool.size),
                replace=False,
            )
        ).astype(np.int64)
        effect_direction = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=target_idx.size,
            p=[0.25, 0.75],
        )

        ligand = str(gene_names[ligand_idx[0]])
        receptor = str(gene_names[receptor_idx[0]])
        pair_id = f"{module_id}:{ligand}->{receptor}"
        distance_decay = {
            "local": 0.10,
            "regional": 0.22,
            "long_range": 0.45,
        }[range_type] * tissue_scale

        modules[module_id] = {
            "module_id": module_id,
            "module_name": f"Communication module {module_idx}",
            "ligand_gene": ligand,
            "receptor_gene": receptor,
            "ligand_genes": gene_names[ligand_idx],
            "receptor_genes": gene_names[receptor_idx],
            "target_genes": gene_names[target_idx],
            "ligand_gene_indices": ligand_idx,
            "receptor_gene_indices": receptor_idx,
            "target_gene_indices": target_idx,
            "effect_direction": effect_direction,
            "effect_strength": float(rng.uniform(0.75, 1.25)),
            "range_type": range_type,
            "distance_decay": float(distance_decay),
            "compatible_receiver_states": compatible_states,
            "allowed_regions": allowed_regions,
            "concept_name": (
                "developmental_state"
                if module_idx % 3 == 0
                else "regional_signal"
                if module_idx % 3 == 1
                else "spatial_gradient"
            ),
        }
        pair_ids.append(pair_id)
        pair_modules.append(module_id)
        pair_ligands.append(ligand)
        pair_receptors.append(receptor)

    lr_pairs = {
        "pair_id": np.asarray(pair_ids, dtype=object),
        "module_id": np.asarray(pair_modules, dtype=object),
        "ligand_gene": np.asarray(pair_ligands, dtype=object),
        "receptor_gene": np.asarray(pair_receptors, dtype=object),
    }
    return modules, lr_pairs


def _mean_gene_score(expr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return (0.05 + expr[:, indices].mean(axis=1)).astype(np.float32)


def _lr_gene_indices(modules: dict[str, dict[str, Any]]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for module in modules.values():
        arrays.append(np.asarray(module["ligand_gene_indices"], dtype=np.int64))
        arrays.append(np.asarray(module["receptor_gene_indices"], dtype=np.int64))
    if not arrays:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(arrays)).astype(np.int64)


def _module_focus(
    module: dict[str, Any],
    states: np.ndarray,
    regions: np.ndarray,
) -> np.ndarray:
    state_mask = np.isin(states.astype(str), module["compatible_receiver_states"].astype(str))
    if module["range_type"] == "long_range":
        return state_mask
    region_mask = np.isin(regions.astype(str), module["allowed_regions"].astype(str))
    focus = state_mask & region_mask
    return focus if np.any(focus) else state_mask


def _standardized_spatial_axis(coords: np.ndarray) -> np.ndarray:
    centered = coords - np.median(coords, axis=0, keepdims=True)
    axis = centered[:, 1]
    return axis / max(float(np.std(axis)), 1e-8)


def _gaussian_field(values: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((values - center) / max(width, 1e-8)) ** 2)


def _mode_lr_program_field(
    mode: str,
    module_idx: int,
    n_modules: int,
    latent: np.ndarray,
    spatial_axis: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    """Mode-specific spatial/temporal LR availability.

    CellVOCAL candidate CCC maps are driven mainly by LR expression gates plus
    spatial edges.  Keeping this field identical across modes makes the final
    LR spatial plots look identical even if the hidden simulated event influence model
    differs, so each mode gets a distinct sender/receiver availability program.
    """
    frac = (module_idx + 0.5) / max(float(n_modules), 1.0)
    spatial_center = -1.25 + 2.5 * frac
    temporal_centers = np.linspace(0.18, 0.82, max(n_modules, 1))
    temporal_center = float(temporal_centers[min(module_idx, temporal_centers.size - 1)])

    if mode == "organizer_dominant":
        core = _gaussian_field(latent, temporal_center, 0.10) * _gaussian_field(
            spatial_axis,
            spatial_center,
            0.48,
        )
        halo = _gaussian_field(latent, temporal_center, 0.18) * _gaussian_field(
            spatial_axis,
            spatial_center,
            0.85,
        )
        field = core if kind == "ligand" else 0.35 * halo + 0.65 * core
    elif mode == "collective_redundant":
        wave_t = 0.5 + 0.5 * np.sin(2.0 * np.pi * (latent + 0.17 * module_idx))
        wave_s = 0.5 + 0.5 * np.cos(1.7 * spatial_axis - 0.9 * module_idx)
        patch = _gaussian_field(latent, temporal_center, 0.24) + _gaussian_field(
            spatial_axis,
            spatial_center,
            0.95,
        )
        field = 0.28 * patch + 0.42 * wave_t + 0.30 * wave_s
    elif mode == "contact_dependent":
        field = _gaussian_field(latent, temporal_center, 0.085) * _gaussian_field(
            spatial_axis,
            spatial_center,
            0.40,
        )
    elif mode == "graded_paracrine":
        direction = 1.0 if module_idx % 2 == 0 else -1.0
        grad = 1.0 / (
            1.0
            + np.exp(
                -direction
                * (spatial_axis - 0.35 * np.sin(module_idx + 1.0))
                / 0.65
            )
        )
        temporal = _gaussian_field(latent, temporal_center, 0.22)
        field = 0.68 * grad + 0.32 * temporal
        if kind == "receptor":
            field = 0.55 * field + 0.45 * _gaussian_field(
                spatial_axis,
                spatial_center,
                1.10,
            )
    elif mode == "developmental_switch":
        switch = 1.0 / (1.0 + np.exp(-(latent - 0.5) / 0.055))
        early = _gaussian_field(latent, 0.22 + 0.06 * (module_idx % 2), 0.13)
        late = _gaussian_field(latent, 0.78 - 0.06 * (module_idx % 2), 0.13)
        stage = (1.0 - switch) * early if module_idx % 2 == 0 else switch * late
        side = _gaussian_field(
            spatial_axis,
            -0.85 if module_idx < max(n_modules, 1) / 2 else 0.85,
            0.75,
        )
        field = 0.72 * stage + 0.28 * side
    else:
        field = np.ones_like(latent, dtype=np.float64)

    field = _normalize(np.asarray(field, dtype=np.float64))
    return np.clip(field, 0.0, 1.0)


def _sender_receiver_coupling(mode: str) -> float:
    return {
        "organizer_dominant": 0.20,
        "collective_redundant": 0.55,
        "contact_dependent": 0.88,
        "graded_paracrine": 0.25,
        "developmental_switch": 0.45,
    }.get(mode, 0.45)


def _exclusive_receiver_module_weights(
    modules: dict[str, dict[str, Any]],
    latent: np.ndarray,
    states: np.ndarray,
    regions: np.ndarray,
    coords: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Assign each receiver cell a small, capacity-balanced set of modules.

    The capacity balance prevents one LR module from dominating all receivers.
    With 4 modules each module still owns about 25% of receivers, so a strict
    CellVOCAL q=0.8 top-20% receptor gate remains more selective than full
    receiver-module coverage.
    """
    module_ids = list(modules)
    n_cells = states.size
    n_modules = len(module_ids)
    spatial_axis = _standardized_spatial_axis(coords)
    scores = np.zeros((n_cells, n_modules), dtype=np.float64)
    for module_pos, module_id in enumerate(module_ids):
        module = modules[module_id]
        state_ok = np.isin(
            states.astype(str),
            module["compatible_receiver_states"].astype(str),
        )
        region_ok = np.isin(
            regions.astype(str),
            module["allowed_regions"].astype(str),
        )
        competence = np.where(state_ok, 1.0, 0.02)
        if module["range_type"] != "long_range":
            competence *= np.where(region_ok, 1.0, 0.08)
        mode_bias = _mode_lr_program_field(
            mode,
            module_pos,
            n_modules,
            latent,
            spatial_axis,
            kind="receptor",
        )
        # Deterministic tiny tie-breaker distributes equal-compatibility modules
        # without adding a new random stream.
        tie = 1e-4 * np.sin((module_pos + 1) * 17.0 + latent * 31.0)
        scores[:, module_pos] = competence * (0.15 + 1.85 * mode_bias) + tie

    weights = np.zeros_like(scores, dtype=np.float32)
    preference = np.argsort(scores, axis=1)[:, ::-1]
    if n_modules > 1:
        confidence = scores[np.arange(n_cells), preference[:, 0]] - scores[
            np.arange(n_cells), preference[:, 1]
        ]
    else:
        confidence = np.ones(n_cells, dtype=np.float64)
    cell_order = np.argsort(-confidence, kind="stable")
    capacity = int(np.ceil(n_cells / max(n_modules, 1)))
    remaining = np.full(n_modules, capacity, dtype=np.int64)

    for cell in cell_order:
        for module_pos in preference[cell]:
            if remaining[int(module_pos)] <= 0:
                continue
            weights[int(cell), int(module_pos)] = 1.0
            remaining[int(module_pos)] -= 1
            break
        if not np.any(weights[int(cell)]):
            module_pos = int(np.argmax(remaining))
            weights[int(cell), module_pos] = 1.0
            remaining[module_pos] -= 1
    return weights


def _top_fraction_program_field(
    field: np.ndarray,
    *,
    active_fraction: float,
    background: float,
) -> np.ndarray:
    """Convert a smooth program into a sparse high-expression plateau."""
    x = np.asarray(field, dtype=np.float64)
    x = np.maximum(x, 0.0)
    if x.size == 0:
        return x
    denom = max(float(np.quantile(x, 0.95)), 1e-8)
    norm = np.clip(x / denom, 0.0, None)
    frac = float(np.clip(active_fraction, 1.0 / max(x.size, 1), 1.0))
    bg = float(np.clip(background, 0.0, 1.0))
    cutoff = float(np.quantile(norm, max(0.0, 1.0 - frac)))
    active = norm >= cutoff
    out = bg * np.clip(norm, 0.0, 1.0)
    if np.any(active):
        vals = norm[active]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi > lo:
            scaled = (vals - lo) / (hi - lo)
        else:
            scaled = np.ones_like(vals)
        out[active] = 0.75 + 0.25 * scaled
    return out.astype(np.float64, copy=False)


def _apply_lr_expression_program(
    mode: str,
    baseline_drive: np.ndarray,
    modules: dict[str, dict[str, Any]],
    latent: np.ndarray,
    states: np.ndarray,
    regions: np.ndarray,
    coords: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Encode LR availability before simulated event influence is calculated.

    This is the key fix: the same LR transcription program drives both the
    simulated event influence ground truth and the generated RNA snapshot.
    """
    drive = baseline_drive.astype(np.float32, copy=True)
    module_ids = list(modules)
    exclusive_weights = _exclusive_receiver_module_weights(
        modules,
        latent,
        states,
        regions,
        coords,
        mode,
    )
    tissue_scale = max(float(np.linalg.norm(np.ptp(coords, axis=0))), 1e-6)
    spatial_axis = _standardized_spatial_axis(coords)
    coupling = _sender_receiver_coupling(mode)

    for module_idx, module_id in enumerate(module_ids):
        module = modules[module_id]
        lig_idx = np.asarray(module["ligand_gene_indices"], dtype=np.int64)
        rec_idx = np.asarray(module["receptor_gene_indices"], dtype=np.int64)
        focus = _module_focus(module, states, regions)
        ligand_mode_field = _mode_lr_program_field(
            mode,
            module_idx,
            len(module_ids),
            latent,
            spatial_axis,
            kind="ligand",
        )
        receptor_mode_field = _mode_lr_program_field(
            mode,
            module_idx,
            len(module_ids),
            latent,
            spatial_axis,
            kind="receptor",
        )

        compatible_state = np.isin(
            states.astype(str),
            module["compatible_receiver_states"].astype(str),
        ).astype(np.float64)
        compatible_region = np.isin(
            regions.astype(str),
            module["allowed_regions"].astype(str),
        ).astype(np.float64)
        if module["range_type"] == "long_range":
            receptor_field = 0.20 + 0.80 * compatible_state
        else:
            receptor_field = 0.08 + 0.62 * compatible_state + 0.30 * compatible_region
        receptor_field = np.clip(receptor_field, 0.05, None)
        receptor_field *= 0.06 + 0.94 * receptor_mode_field

        if module["range_type"] == "local":
            center = np.median(coords[focus], axis=0) if np.any(focus) else np.median(coords, axis=0)
            dist = np.linalg.norm(coords - center[None, :], axis=1)
            ligand_field = np.exp(-dist / max(0.12 * tissue_scale, 1e-8))
        elif module["range_type"] == "regional":
            preference = -0.8 + 1.6 * ((module_idx % 4) / 3.0)
            ligand_field = np.exp(
                -0.5 * ((spatial_axis - preference) / 0.75) ** 2
            )
        else:
            ligand_field = 0.35 + 0.65 * np.sin(np.pi * latent + 0.4 * module_idx) ** 2

        ownership = np.asarray(exclusive_weights[:, module_idx], dtype=np.float64)
        sender_support = coupling * ownership + (1.0 - coupling) * ligand_mode_field
        ligand_field *= 0.06 + 0.94 * sender_support
        ligand_field = np.clip(ligand_field * (0.35 + 0.65 * compatible_state), 0.02, None)
        ligand_field = _top_fraction_program_field(
            ligand_field,
            active_fraction=config.lr_program_active_fraction,
            background=config.lr_program_background,
        )
        receptor_field = receptor_field / max(float(np.quantile(receptor_field, 0.95)), 1e-8)
        receptor_mask = (
            float(config.module_receptor_suppression)
            + (1.0 - float(config.module_receptor_suppression)) * ownership
        )
        receptor_norm = np.clip(receptor_field * receptor_mask, 0.0, None)
        receptor_program = float(config.lr_program_background) * receptor_norm
        owned = ownership > 0.0
        if np.any(owned):
            owned_vals = receptor_norm[owned]
            lo = float(np.min(owned_vals))
            hi = float(np.max(owned_vals))
            if hi > lo:
                owned_scaled = (owned_vals - lo) / (hi - lo)
            else:
                owned_scaled = np.ones_like(owned_vals)
            receptor_program[owned] = 0.85 + 0.15 * owned_scaled
        receptor_field = receptor_program

        strength = float(config.lr_program_strength)
        sharp = float(config.lr_program_sharpness)
        for idx in lig_idx:
            base = np.quantile(drive[:, idx], 0.90)
            drive[:, idx] += (base + 1.0) * strength * np.maximum(ligand_field, 0.0) ** sharp
        for idx in rec_idx:
            base = np.quantile(drive[:, idx], 0.90)
            drive[:, idx] *= receptor_mask.astype(np.float32)
            drive[:, idx] += (base + 1.0) * strength * np.maximum(receptor_field, 0.0) ** sharp

    # Keep all LR genes among the most informative genes for QC/prior filtering.
    for idx in _lr_gene_indices(modules):
        floor = float(np.quantile(drive[:, idx], 0.75))
        drive[:, idx] = np.maximum(drive[:, idx], 0.15 * floor)
    return np.clip(drive, 0.0, None).astype(np.float32)


def _module_activation_mask(
    modules: dict[str, dict[str, Any]],
    receptor_scores: dict[str, np.ndarray],
    states: np.ndarray,
    regions: np.ndarray,
    receiver_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    module_ids = list(modules)
    gate = np.zeros((states.size, len(module_ids)), dtype=np.float32)
    for receiver in receiver_indices:
        score = np.zeros(len(module_ids), dtype=np.float64)
        for pos, module_id in enumerate(module_ids):
            module = modules[module_id]
            state_ok = str(states[receiver]) in set(module["compatible_receiver_states"].astype(str))
            region_ok = str(regions[receiver]) in set(module["allowed_regions"].astype(str))
            competence = 1.0 if state_ok else 0.05
            if module["range_type"] != "long_range":
                competence *= 1.0 if region_ok else 0.12
            score[pos] = competence * float(receptor_scores[module_id][receiver]) + rng.uniform(0.0, 1e-4)
        active = int(np.argmax(score))
        gate[receiver, active] = 1.0
    return gate


def _neighbor_candidates(
    coords: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _spatial_neighbor_candidates(
        coords,
        n_neighbors,
        sender_indices=sender_indices,
        receiver_indices=receiver_indices,
    )


def _organizer_affinity(
    coords: np.ndarray,
    ligand_score: np.ndarray,
    sender_indices: np.ndarray,
) -> np.ndarray:
    seed_sender = int(sender_indices[np.argmax(ligand_score[sender_indices])])
    distance = np.linalg.norm(coords - coords[seed_sender], axis=1)
    scale = max(float(np.linalg.norm(np.ptp(coords, axis=0))) * 0.12, 1e-6)
    return np.exp(-distance / scale).astype(np.float32)


def _sample_sender_count(
    rng: np.random.Generator,
    counts: tuple[int, ...],
    probabilities: tuple[float, ...],
    available: int,
) -> int:
    if available <= 0:
        return 0
    valid = np.asarray([count for count in counts if count <= available], dtype=np.int64)
    if valid.size == 0:
        return 1
    probs = np.asarray(probabilities[: len(counts)], dtype=np.float64)
    probs = probs[np.asarray(counts) <= available]
    if probs.size != valid.size or probs.sum() <= 0.0:
        probs = np.full(valid.size, 1.0 / valid.size)
    else:
        probs = probs / probs.sum()
    return int(rng.choice(valid, p=probs))


def _topk_sender_shares(
    raw: np.ndarray,
    n_keep: int,
    rng: np.random.Generator,
    *,
    temperature: float,
    score_jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_keep = min(max(int(n_keep), 1), raw.size)
    if n_keep <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float32)
    scores = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    if score_jitter > 0.0:
        scores = scores * rng.lognormal(mean=0.0, sigma=float(score_jitter), size=scores.size)
    if np.all(scores <= 1e-12):
        scores = rng.random(scores.size) + 1e-6
    keep = np.argpartition(-scores, n_keep - 1)[:n_keep]
    kept = np.maximum(scores[keep], 1e-12)
    logits = np.log(kept) / max(float(temperature), 1e-3)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    sender_share = weights / max(float(weights.sum()), 1e-12)
    return keep, sender_share.astype(np.float32)


def _mode_sender_share_and_activation(
    mode: str,
    candidate_senders: np.ndarray,
    candidate_distances: np.ndarray,
    receiver: int,
    latent: np.ndarray,
    ligand_score: np.ndarray,
    organizer_affinity: np.ndarray,
    distance_decay: float,
    tissue_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    lig = np.maximum(ligand_score[candidate_senders].astype(np.float64), 1e-6)
    dist = candidate_distances.astype(np.float64)
    distance_scale = max(distance_decay * float(rng.lognormal(0.0, 0.18)), 1e-8)
    standard_distance = np.exp(-dist / distance_scale)

    if mode == "organizer_dominant":
        focus_power = float(rng.uniform(2.5, 4.8))
        raw = lig ** float(rng.uniform(1.8, 3.2))
        raw *= (0.04 + organizer_affinity[candidate_senders]) ** focus_power
        raw *= np.exp(-dist / max(float(rng.uniform(0.18, 0.42)) * tissue_scale, 1e-8))
        max_positive = _sample_sender_count(rng, (1, 2, 3), (0.42, 0.43, 0.15), raw.size)
        temperature = float(rng.uniform(0.26, 0.58))
        activation = 0.25 + 0.75 * float(np.clip(np.max(raw), 0.0, 1.0))
    elif mode == "collective_redundant":
        evidence = np.sqrt(lig * standard_distance)
        max_positive = _sample_sender_count(
            rng,
            (4, 5, 6, 7, 8, 9, 10, 11, 12),
            (0.05, 0.08, 0.12, 0.16, 0.18, 0.16, 0.12, 0.08, 0.05),
            evidence.size,
        )
        raw = evidence ** float(rng.uniform(0.45, 0.90))
        temperature = float(rng.uniform(0.90, 1.80))
        keep0 = np.argpartition(-raw, max_positive - 1)[:max_positive]
        active_fraction = float(np.mean(lig[keep0] > np.median(lig)))
        activation = 0.35 + 0.65 * (1.0 - np.exp(-max_positive / 4.0)) * (0.6 + 0.4 * active_fraction)
    elif mode == "contact_dependent":
        contact = np.exp(-dist / max(float(rng.uniform(0.014, 0.045)) * tissue_scale, 1e-8))
        raw = lig ** float(rng.uniform(1.0, 1.8)) * contact
        max_positive = _sample_sender_count(rng, (1, 2, 3), (0.20, 0.55, 0.25), raw.size)
        temperature = float(rng.uniform(0.45, 0.95))
        activation = 0.20 + 0.80 * float(np.clip(np.max(raw), 0.0, 1.0))
    elif mode == "graded_paracrine":
        raw = (lig * standard_distance) ** float(rng.uniform(1.0, 2.4))
        max_positive = _sample_sender_count(
            rng,
            (3, 4, 5, 6, 7, 8, 9),
            (0.08, 0.13, 0.18, 0.22, 0.18, 0.13, 0.08),
            raw.size,
        )
        temperature = float(rng.uniform(0.75, 1.50))
        top = np.sort(raw)[-min(max_positive, raw.size) :]
        activation = 0.25 + 0.75 * float(1.0 - np.exp(-top.sum() / 2.5))
    elif mode == "developmental_switch":
        receiver_time = float(latent[receiver])
        late_mix = 1.0 / (1.0 + np.exp(-(receiver_time - rng.normal(0.5, 0.035)) / 0.07))
        sender_time = latent[candidate_senders]
        early_program = np.exp(-0.5 * ((sender_time - rng.normal(0.18, 0.035)) / rng.uniform(0.08, 0.16)) ** 2)
        late_program = np.exp(-0.5 * ((sender_time - rng.normal(0.82, 0.035)) / rng.uniform(0.08, 0.16)) ** 2)
        stage_program = (1.0 - late_mix) * early_program + late_mix * late_program
        raw = lig * np.exp(-dist / max(0.30 * tissue_scale, 1e-8))
        raw *= 0.03 + stage_program
        max_positive = _sample_sender_count(rng, (2, 3, 4, 5, 6), (0.14, 0.24, 0.28, 0.22, 0.12), raw.size)
        temperature = float(rng.uniform(0.55, 1.15))
        top = np.sort(raw)[-min(max_positive, raw.size) :]
        activation = 0.25 + 0.75 * float(1.0 - np.exp(-top.sum() / 2.0))
    else:
        raise ValueError(f"Unknown event-attribution mode {mode!r}")

    if max_positive <= 0 or raw.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float32), 0.0
    keep, sender_share = _topk_sender_shares(
        raw,
        max_positive,
        rng,
        temperature=temperature,
        score_jitter=0.16,
    )
    activation *= float(rng.lognormal(mean=0.0, sigma=0.12))
    return keep, sender_share.astype(np.float32), float(np.clip(activation, 0.05, 1.0))


def _communication_direction(
    coords: np.ndarray,
    intrinsic_velocity: np.ndarray,
    sender: int,
    receiver: int,
) -> np.ndarray:
    forward = intrinsic_velocity[receiver].astype(np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    displacement = (coords[receiver] - coords[sender]).astype(np.float64)
    displacement /= max(float(np.linalg.norm(displacement)), 1e-8)
    if float(np.dot(displacement, forward)) < 0.0:
        displacement = -displacement
    direction = 0.75 * forward + 0.25 * displacement
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    return direction.astype(np.float32)


def _event_table(records: list[EventRecord]) -> dict[str, np.ndarray]:
    fields = {
        "sender_id": (0, np.int64),
        "receiver_id": (1, np.int64),
        "module_id": (2, object),
        "label": (3, object),
        "negative_type": (4, object),
        "simulated_event_influence": (5, np.float32),
        "module_activation_prob": (6, np.float32),
        "sender_share": (7, np.float32),
        "effect_magnitude": (8, np.float32),
        "distance": (9, np.float32),
        "ligand_score": (10, np.float32),
        "receptor_score": (11, np.float32),
    }
    if not records:
        return {name: np.asarray([], dtype=dtype) for name, (_, dtype) in fields.items()}
    return {
        name: np.asarray([row[pos] for row in records], dtype=dtype)
        for name, (pos, dtype) in fields.items()
    }


def _hard_negative_records(
    modules: dict[str, dict[str, Any]],
    module_activation_mask: np.ndarray,
    positive_keys: set[tuple[int, int, str]],
    neighbor_senders: np.ndarray,
    neighbor_distances: np.ndarray,
    receiver_indices: np.ndarray,
    ligand_scores: dict[str, np.ndarray],
    receptor_scores: dict[str, np.ndarray],
    max_events: int,
    rng: np.random.Generator,
) -> list[EventRecord]:
    if max_events <= 0:
        return []
    module_ids = list(modules)
    module_pos = {module_id: pos for pos, module_id in enumerate(module_ids)}
    receiver_pos = {int(receiver): pos for pos, receiver in enumerate(receiver_indices)}
    records: list[EventRecord] = []
    seen: set[tuple[int, int, str]] = set()
    categories = (
        "spatial_close_wrong_module",
        "low_ligand_active_module",
        "receptor_incompetent",
    )
    attempts = 0
    while len(records) < max_events and attempts < max_events * 30:
        attempts += 1
        receiver = int(rng.choice(receiver_indices))
        rpos = receiver_pos[receiver]
        valid_locals = np.flatnonzero(neighbor_senders[rpos] >= 0)
        if valid_locals.size == 0:
            continue
        local = int(rng.choice(valid_locals))
        sender = int(neighbor_senders[rpos, local])
        if sender == receiver:
            continue
        category = categories[len(records) % len(categories)]

        if category == "spatial_close_wrong_module":
            inactive = [
                module_id
                for module_id in module_ids
                if module_activation_mask[receiver, module_pos[module_id]] <= 0.0
            ]
            if not inactive:
                continue
            module_id = str(rng.choice(inactive))
        elif category == "low_ligand_active_module":
            active = [
                module_id
                for module_id in module_ids
                if module_activation_mask[receiver, module_pos[module_id]] > 0.0
            ]
            if not active:
                continue
            module_id = str(rng.choice(active))
            row_senders = neighbor_senders[rpos]
            row_senders = row_senders[row_senders >= 0]
            if row_senders.size == 0:
                continue
            sender = int(row_senders[np.argmin(ligand_scores[module_id][row_senders])])
        else:
            module_id = str(rng.choice(module_ids))
            low_receivers = receiver_indices[
                np.argsort(receptor_scores[module_id][receiver_indices])[
                    : min(200, receiver_indices.size)
                ]
            ]
            receiver = int(rng.choice(low_receivers))
            rpos = receiver_pos[receiver]
            valid_locals = np.flatnonzero(neighbor_senders[rpos] >= 0)
            if valid_locals.size == 0:
                continue
            local = int(rng.choice(valid_locals))
            sender = int(neighbor_senders[rpos, local])

        key = (sender, receiver, module_id)
        if key in positive_keys or key in seen or sender == receiver:
            continue
        seen.add(key)
        matches = np.flatnonzero(neighbor_senders[rpos] == sender)
        distance = float(neighbor_distances[rpos, matches[0]]) if matches.size else float("nan")
        records.append(
            (
                sender,
                receiver,
                module_id,
                "hard_negative",
                category,
                0.0,
                0.0,
                0.0,
                0.0,
                distance,
                float(ligand_scores[module_id][sender]),
                float(receptor_scores[module_id][receiver]),
            )
        )
    return records


def _export_ground_truth_events(events: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop hidden simulated event influence fields before writing simulation ground truth."""
    return {name: np.asarray(events[name]) for name in _GROUND_TRUTH_EVENT_FIELDS}


def _communication_event_summary(events: dict[str, np.ndarray], latent: np.ndarray) -> dict[str, float]:
    positive = np.asarray(events["label"], dtype=str) == "positive"
    if not np.any(positive):
        return {}
    sender = events["sender_id"][positive].astype(np.int64)
    receiver = events["receiver_id"][positive].astype(np.int64)
    module = events["module_id"][positive].astype(str)
    ligand = events["ligand_score"][positive].astype(np.float64)
    receptor = events["receptor_score"][positive].astype(np.float64)
    lr_expression = ligand * receptor
    distance = events["distance"][positive].astype(np.float64)

    groups: dict[tuple[int, str], list[int]] = {}
    for idx, key in enumerate(zip(receiver, module, strict=True)):
        groups.setdefault((int(key[0]), str(key[1])), []).append(idx)
    top_share: list[float] = []
    effective_senders: list[float] = []
    for idxs in groups.values():
        weights = lr_expression[np.asarray(idxs, dtype=int)]
        weights = weights / max(float(weights.sum()), 1e-12)
        top_share.append(float(np.max(weights)))
        effective_senders.append(float(1.0 / np.sum(weights**2)))

    early = latent[receiver] < 0.4
    late = latent[receiver] > 0.6
    return {
        "n_positive_events": float(np.sum(positive)),
        "n_receiver_module_groups": float(len(groups)),
        "mean_lr_expression": float(np.mean(lr_expression)),
        "mean_top_sender_lr_share": float(np.mean(top_share)),
        "mean_effective_senders_by_lr": float(np.mean(effective_senders)),
        "mean_sender_receiver_distance": float(np.nanmean(distance)),
        "early_mean_sender_time": float(np.mean(latent[sender[early]])) if np.any(early) else float("nan"),
        "late_mean_sender_time": float(np.mean(latent[sender[late]])) if np.any(late) else float("nan"),
    }


def _build_communication_truth(
    mode: str,
    true_time: np.ndarray,
    latent: np.ndarray,
    coords: np.ndarray,
    intrinsic_velocity: np.ndarray,
    states: np.ndarray,
    regions: np.ndarray,
    baseline_drive: np.ndarray,
    lr_expression_drive: np.ndarray,
    modules: dict[str, dict[str, Any]],
    lr_pairs: dict[str, np.ndarray],
    config: SimulationConfig,
    event_influence_rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    expr = _normalize(lr_expression_drive)
    module_ids = list(modules)
    ligand_scores = {
        module_id: _mean_gene_score(expr, modules[module_id]["ligand_gene_indices"])
        for module_id in module_ids
    }
    receptor_scores = {
        module_id: _mean_gene_score(expr, modules[module_id]["receptor_gene_indices"])
        for module_id in module_ids
    }

    is_sender = latent <= 0.98
    is_receiver = latent >= 0.02
    sender_indices = np.flatnonzero(is_sender)
    receiver_indices = np.flatnonzero(is_receiver)
    gate = _module_activation_mask(
        modules,
        receptor_scores,
        states,
        regions,
        receiver_indices,
        event_influence_rng,
    )
    neighbor_senders, neighbor_distances = _neighbor_candidates(
        coords,
        sender_indices,
        receiver_indices,
        config.n_sender_neighbors,
    )
    organizer_fields = {
        module_id: _organizer_affinity(coords, ligand_scores[module_id], sender_indices)
        for module_id in module_ids
    }

    tissue_scale = max(float(np.linalg.norm(np.ptp(coords, axis=0))), 1e-6)
    raw_comm_velocity = np.zeros_like(intrinsic_velocity, dtype=np.float32)
    expression_effect = np.zeros_like(baseline_drive, dtype=np.float32)
    positive_records: list[EventRecord] = []
    positive_keys: set[tuple[int, int, str]] = set()

    for receiver_pos, receiver in enumerate(receiver_indices):
        row_senders = neighbor_senders[receiver_pos]
        row_distances = neighbor_distances[receiver_pos]
        valid = (row_senders >= 0) & (row_senders != receiver) & np.isfinite(row_distances)
        row_senders = row_senders[valid]
        row_distances = row_distances[valid]
        if row_senders.size == 0:
            continue

        for module_pos, module_id in enumerate(module_ids):
            base_gate = float(gate[receiver, module_pos])
            if base_gate <= 0.0:
                continue
            module = modules[module_id]
            keep, sender_share, activation = _mode_sender_share_and_activation(
                mode,
                row_senders,
                row_distances,
                int(receiver),
                latent,
                ligand_scores[module_id],
                organizer_fields[module_id],
                float(module["distance_decay"]),
                tissue_scale,
                event_influence_rng,
            )
            if keep.size == 0:
                continue
            senders = row_senders[keep]
            distances = row_distances[keep]
            module_activation_prob = base_gate * activation
            simulated_event_influence = module_activation_prob * sender_share
            target_idx = module["target_gene_indices"]
            target_direction = module["effect_direction"]
            ligand_idx = module["ligand_gene_indices"]
            strength = float(module["effect_strength"])
            receptor_score = float(receptor_scores[module_id][receiver])

            for sender, distance, p_sender, a in zip(
                senders,
                distances,
                sender_share,
                simulated_event_influence,
                strict=True,
            ):
                sender = int(sender)
                a = float(a)
                direction = _communication_direction(coords, intrinsic_velocity, sender, int(receiver))
                raw_comm_velocity[receiver] += (a * strength * direction).astype(np.float32)
                expression_effect[receiver, target_idx] += (
                    a * strength * target_direction
                ).astype(np.float32)
                expression_effect[sender, ligand_idx] += np.float32(
                    a
                    * abs(strength)
                    * float(config.sender_ligand_feedback_strength)
                )
                positive_keys.add((sender, int(receiver), module_id))
                positive_records.append(
                    (
                        sender,
                        int(receiver),
                        module_id,
                        "positive",
                        "",
                        a,
                        float(module_activation_prob),
                        float(p_sender),
                        abs(strength),
                        float(distance),
                        float(ligand_scores[module_id][sender]),
                        receptor_score,
                    )
                )

    comm_norm = np.linalg.norm(raw_comm_velocity, axis=1)
    intrinsic_norm = np.linalg.norm(intrinsic_velocity, axis=1)
    if np.any(comm_norm > 1e-8):
        scale = (
            config.communication_velocity_weight
            * float(np.percentile(intrinsic_norm[intrinsic_norm > 1e-8], 95))
            / max(float(np.percentile(comm_norm[comm_norm > 1e-8], 95)), 1e-8)
        )
    else:
        scale = 0.0
    communication_velocity = (raw_comm_velocity * scale).astype(np.float32)
    true_velocity = (intrinsic_velocity + communication_velocity).astype(np.float32)

    nonzero_effect = np.abs(expression_effect)
    nonzero_effect = nonzero_effect[nonzero_effect > 1e-8]
    nonzero_drive = lr_expression_drive[lr_expression_drive > 1e-8]
    if nonzero_effect.size:
        effect_scale = (
            config.expression_effect_strength
            * float(np.percentile(nonzero_drive, 75))
            / max(float(np.percentile(nonzero_effect, 95)), 1e-8)
        )
    else:
        effect_scale = 0.0
    expression_effect = (expression_effect * effect_scale).astype(np.float32)

    negative_budget = min(config.max_hard_negative_events, max(1000, len(positive_records)))
    negative_records = _hard_negative_records(
        modules,
        gate,
        positive_keys,
        neighbor_senders,
        neighbor_distances,
        receiver_indices,
        ligand_scores,
        receptor_scores,
        negative_budget,
        event_influence_rng,
    )
    events = _event_table(positive_records + negative_records)
    summary = _communication_event_summary(events, latent)

    obs = {
        "latent_time": latent.astype(np.float32),
        "receiver_state": states,
        "region": regions,
        "cell_type": np.asarray(
            [f"{state}_{region}" for state, region in zip(states, regions, strict=True)],
            dtype=object,
        ),
        "sender_type": np.asarray([f"sender_{state}" for state in states], dtype=object),
        "is_sender": is_sender,
        "is_receiver": is_receiver,
        "event_attribution_mode": np.full(states.size, mode, dtype=object),
    }
    obsm = {
        "spatial": coords.copy(),
        "intrinsic_spatial_velocity": intrinsic_velocity.copy(),
        "communication_spatial_velocity": communication_velocity,
        "true_spatial_velocity": true_velocity,
        "velocity_spatial": true_velocity.copy(),
    }
    uns = {
        "modules": modules,
        "lr_pairs": lr_pairs,
        "events": _export_ground_truth_events(events),
        "communication_summary": summary,
        "lr_expression_program": "lr_programmed_transcription_drive",
        "expression_effect_scale": float(effect_scale),
        "lr_dropout_scale": float(config.lr_dropout_scale),
        "lr_observation_anchor_strength": float(config.lr_observation_anchor_strength),
        "communication_spatial_scale": float(scale),
        "mode": mode,
        "random_seed": int(config.seed),
    }
    return obs, obsm, expression_effect, uns


def _induction_solution(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    duration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    duration = np.maximum(duration, 0.0)
    exp_beta = np.exp(-duration * beta)
    exp_gamma = np.exp(-duration * gamma)
    u = alpha / beta * (1.0 - exp_beta)
    diff = gamma - beta
    regular = np.abs(diff) > 1e-5
    s = np.empty_like(alpha, dtype=np.float64)
    s[:, regular[0]] = (
        alpha[:, regular[0]] / gamma[:, regular[0]] * (1.0 - exp_gamma[:, regular[0]])
        + alpha[:, regular[0]]
        / diff[:, regular[0]]
        * (exp_gamma[:, regular[0]] - exp_beta[:, regular[0]])
    )
    if np.any(~regular[0]):
        idx = ~regular[0]
        s[:, idx] = (
            alpha[:, idx] / gamma[:, idx] * (1.0 - exp_gamma[:, idx])
            - alpha[:, idx] * duration[:, idx] * exp_gamma[:, idx]
        )
    return u, s


def _rna_snapshot(
    true_time: np.ndarray,
    transcription_drive: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    ton: np.ndarray,
    toff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t = true_time[:, None].astype(np.float64)
    alpha = transcription_drive.astype(np.float64)
    beta2 = beta[None, :].astype(np.float64)
    gamma2 = gamma[None, :].astype(np.float64)
    ton2 = ton[None, :]
    toff2 = toff[None, :]
    on_duration = np.clip(np.minimum(t, toff2) - ton2, 0.0, None)
    u_on, s_on = _induction_solution(alpha, beta2, gamma2, on_duration)

    off_duration = np.clip(t - toff2, 0.0, None)
    exp_beta = np.exp(-off_duration * beta2)
    exp_gamma = np.exp(-off_duration * gamma2)
    u = u_on * exp_beta
    diff = gamma2 - beta2
    regular = np.abs(diff) > 1e-5
    s = np.empty_like(alpha, dtype=np.float64)
    s[:, regular[0]] = (
        s_on[:, regular[0]] * exp_gamma[:, regular[0]]
        + beta2[:, regular[0]]
        * u_on[:, regular[0]]
        / diff[:, regular[0]]
        * (exp_beta[:, regular[0]] - exp_gamma[:, regular[0]])
    )
    if np.any(~regular[0]):
        idx = ~regular[0]
        s[:, idx] = (s_on[:, idx] + beta2[:, idx] * u_on[:, idx] * off_duration[:, idx]) * exp_gamma[:, idx]
    before_on = t < ton2
    u[before_on] = 0.0
    s[before_on] = 0.0
    return np.clip(u, 0.0, None), np.clip(s, 0.0, None)


def _stabilize_lr_kinetics(
    beta: np.ndarray,
    gamma: np.ndarray,
    ton: np.ndarray,
    toff: np.ndarray,
    lr_indices: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make LR genes observed-expression ranks track their transcription drive.

    Simulated event influences are defined from the LR expression program.  If LR
    genes use arbitrary gene-specific on/off kinetics, final spliced counts can
    be rank-distorted even though the simulated biology says those LR programs
    define communication availability.
    """
    if lr_indices.size == 0:
        return beta, gamma, ton, toff
    beta2 = beta.astype(np.float64, copy=True)
    gamma2 = gamma.astype(np.float64, copy=True)
    ton2 = ton.astype(np.float64, copy=True)
    toff2 = toff.astype(np.float64, copy=True)
    idx = np.asarray(lr_indices, dtype=np.int64)
    beta2[idx] = max(float(np.median(beta)), 1.10)
    gamma2[idx] = max(float(np.median(gamma)), 0.90)
    ton2[idx] = -0.25 * float(config.t_max)
    toff2[idx] = 2.0 * float(config.t_max)
    return beta2, gamma2, ton2, toff2


def _expression_dependent_dropout(
    x: np.ndarray,
    base_dropout: float,
    protection: float,
    rng: np.random.Generator,
    *,
    protected_gene_indices: np.ndarray | None = None,
    protected_dropout_scale: float = 1.0,
) -> np.ndarray:
    """Apply dropout before saving, with high-expression LR cells naturally protected."""
    if base_dropout <= 0.0:
        return x
    positive = x[x > 0.0]
    if positive.size == 0:
        return x
    scale = max(float(np.percentile(positive, 75)), 1e-8)
    p_drop = float(base_dropout) * np.exp(-float(protection) * x / scale)
    p_drop = np.clip(p_drop, 0.02, float(base_dropout))
    if protected_gene_indices is not None:
        idx = np.asarray(protected_gene_indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < x.shape[1])]
        if idx.size:
            p_drop[:, idx] *= float(np.clip(protected_dropout_scale, 0.0, 1.0))
    return x * (rng.random(x.shape) >= p_drop)


def _negative_binomial_counts(
    mean: np.ndarray,
    dispersion: float,
    dropout_rate: float,
    dropout_expression_protection: float,
    rng: np.random.Generator,
    *,
    protected_gene_indices: np.ndarray | None = None,
    protected_dropout_scale: float = 1.0,
) -> np.ndarray:
    shape = max(float(dispersion), 1e-3)
    gamma_scale = np.maximum(mean, 1e-8) / shape
    rate = rng.gamma(shape=shape, scale=gamma_scale)
    counts = rng.poisson(rate).astype(np.float32)
    return _expression_dependent_dropout(
        counts,
        dropout_rate,
        dropout_expression_protection,
        rng,
        protected_gene_indices=protected_gene_indices,
        protected_dropout_scale=protected_dropout_scale,
    ).astype(np.float32)


def _scale_like_observation(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    target_q = float(np.percentile(target[target > 0.0], 90)) if np.any(target > 0.0) else 0.0
    obs_q = float(np.percentile(observed[observed > 0.0], 90)) if np.any(observed > 0.0) else 0.0
    if target_q <= 1e-8 or obs_q <= 1e-8:
        return np.maximum(target, 0.0)
    return np.maximum(target, 0.0) * (obs_q / target_q)


def _anchor_lr_observation_to_drive(
    u: np.ndarray,
    s: np.ndarray,
    final_drive: np.ndarray,
    lr_indices: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep final LR observations consistent with the drive used for CCC scores."""
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0 or lr_indices.size == 0:
        return u, s
    u2 = u.astype(np.float64, copy=True)
    s2 = s.astype(np.float64, copy=True)
    for idx in np.asarray(lr_indices, dtype=np.int64):
        if idx < 0 or idx >= final_drive.shape[1]:
            continue
        target_s = _scale_like_observation(final_drive[:, idx], s2[:, idx])
        target_u = _scale_like_observation(final_drive[:, idx], u2[:, idx])
        s2[:, idx] = (1.0 - weight) * s2[:, idx] + weight * target_s
        u2[:, idx] = (1.0 - weight) * u2[:, idx] + weight * target_u
    return np.clip(u2, 0.0, None), np.clip(s2, 0.0, None)


def simulate_event_attribution_dataset(
    mode: str,
    config: SimulationConfig = SimulationConfig(),
) -> ad.AnnData:
    if mode not in EVENT_ATTRIBUTION_MODES:
        raise ValueError(f"Unknown event-attribution mode {mode!r}")
    if config.n_cells < 20:
        raise ValueError("n_cells must be at least 20")
    if config.n_genes < 12:
        raise ValueError("n_genes must be at least 12")

    spatial_rng = np.random.default_rng(config.seed + 101)
    expression_rng = np.random.default_rng(config.seed + 202)
    module_rng = np.random.default_rng(config.seed + 303)
    event_influence_rng = np.random.default_rng(config.seed + 404)
    noise_rng = np.random.default_rng(config.seed + 505)

    true_time, latent, coords, intrinsic_velocity, states, regions = _generate_linear_development(
        config,
        spatial_rng,
    )
    baseline_drive, beta, gamma, ton, toff = _generate_baseline_transcription(
        latent,
        coords,
        config,
        expression_rng,
    )
    modules, lr_pairs = _build_modules(
        baseline_drive,
        latent,
        states,
        regions,
        coords,
        config,
        module_rng,
    )
    lr_indices = _lr_gene_indices(modules)
    lr_programmed_drive = _apply_lr_expression_program(
        mode,
        baseline_drive,
        modules,
        latent,
        states,
        regions,
        coords,
        config,
    )
    obs_truth, obsm_truth, expression_effect, ground_truth = _build_communication_truth(
        mode,
        true_time,
        latent,
        coords,
        intrinsic_velocity,
        states,
        regions,
        baseline_drive,
        lr_programmed_drive,
        modules,
        lr_pairs,
        config,
        event_influence_rng,
    )
    final_drive = np.clip(lr_programmed_drive + expression_effect, 0.0, None)
    beta, gamma, ton, toff = _stabilize_lr_kinetics(beta, gamma, ton, toff, lr_indices, config)
    u, s = _rna_snapshot(true_time, final_drive, beta, gamma, ton, toff)

    u += noise_rng.normal(0.0, config.noise_level, size=u.shape) * np.sqrt(u + 0.05)
    s += noise_rng.normal(0.0, config.noise_level, size=s.shape) * np.sqrt(s + 0.05)
    u = np.clip(u, 0.0, None)
    s = np.clip(s, 0.0, None)
    u = _expression_dependent_dropout(
        u,
        config.unspliced_dropout,
        config.dropout_expression_protection,
        noise_rng,
        protected_gene_indices=lr_indices,
        protected_dropout_scale=config.lr_dropout_scale,
    )
    s = _expression_dependent_dropout(
        s,
        config.spliced_dropout,
        config.dropout_expression_protection,
        noise_rng,
        protected_gene_indices=lr_indices,
        protected_dropout_scale=config.lr_dropout_scale,
    )
    u, s = _anchor_lr_observation_to_drive(
        u,
        s,
        final_drive,
        lr_indices,
        config.lr_observation_anchor_strength,
    )

    gene_names = np.asarray([f"G{idx:04d}" for idx in range(config.n_genes)])
    obs = {"true_time": true_time.astype(np.float32), **obs_truth}
    var = {
        "true_alpha_baseline": np.mean(baseline_drive, axis=0).astype(np.float32),
        "true_alpha_lr_programmed": np.mean(lr_programmed_drive, axis=0).astype(np.float32),
        "true_beta": beta.astype(np.float32),
        "true_gamma": gamma.astype(np.float32),
        "true_ton": ton.astype(np.float32),
        "true_toff": toff.astype(np.float32),
    }
    adata = ad.AnnData(
        X=sparse.csr_matrix(s.astype(np.float32)),
        obs=obs,
        var=var,
        layers={
            "unspliced": sparse.csr_matrix(u.astype(np.float32)),
            "spliced": sparse.csr_matrix(s.astype(np.float32)),
            "true_rho": (
                final_drive
                / np.maximum(np.mean(lr_programmed_drive, axis=0, keepdims=True), 1e-8)
            ).astype(np.float32),
            "baseline_transcription_drive": sparse.csr_matrix(baseline_drive),
            "lr_programmed_transcription_drive": sparse.csr_matrix(lr_programmed_drive),
            "communication_expression_effect": sparse.csr_matrix(expression_effect),
        },
    )
    adata.var_names = gene_names
    adata.obs_names = np.asarray([str(idx) for idx in range(config.n_cells)])
    adata.obsm["X_spatial"] = coords
    for key, value in obsm_truth.items():
        adata.obsm[key] = value
    adata.uns["ground_truth"] = ground_truth
    return adata

