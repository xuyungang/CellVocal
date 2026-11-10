"""Differential CCC simulation pipe and implementation.

The simulator perturbs receiver activity and sender choice through hidden
internal variables

    a_ji^m = q_i^m * p_ji^m,

where q is receiver-module activity and p is the sender distribution within a
receiver-module group. These variables are used only to generate expression and
spatial effects; generated h5ad files export interaction labels, not q/p/a
simulation truth. Four scenarios independently manipulate activity and
sender choice:

* null: neither q nor p changes;
* activity: q changes while p is fixed;
* sender_shift: sender ligand source and sender choice change while q is fixed;
* mixed: both q and p change.

Sender shifts use nested receiver-module group prefixes while retaining a
small, fixed LR vocabulary. The perturbed ligand field is changed only in
sender neighborhoods touched by selected groups, and the same observable field
is used to derive p. The simulator does not import CellVOCAL, but it shares
low-level LR observation and event-summary helpers with the static CCC
simulator so all benchmarks expose the same RNA conventions.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.simulation.core import (
    save_h5ad,
    _anchor_lr_observation_to_drive as _ccc_anchor_lr_observation_to_drive,
    _communication_event_summary as _ccc_communication_event_summary,
    _concat_tables,
    _developmental_spatial_pattern,
    _expression_distance_sender_share,
    _linear_development_tissue,
    _lr_gene_indices as _ccc_lr_gene_indices,
    _nearest_cell,
    _negative_binomial_counts as _ccc_negative_binomial_counts,
    _spatial_neighbor_candidates,
)


SCENARIOS = ("null", "activity", "sender_shift", "mixed")
SENDER_SHIFT_REFERENCE_STRENGTHS = (0.25, 0.50, 1.00)
BENCHMARK_VERSION = "differential_v2_observable_compact"


@dataclass(frozen=True)
class DifferentialSimulationConfig:
    """Compact defaults for the observable differential-CCC benchmark.

    Perturbations change receiver activity, sender shares, or both. Defaults
    concentrate the observable LR and target-gene signal in four paired
    replicates per condition.
    """

    n_cells_per_sample: int = 240
    n_genes: int = 180
    n_modules: int = 4
    target_genes_per_module: int = 24
    n_sender_neighbors: int = 12
    max_active_modules_per_receiver: int = 2
    n_replicates: int = 4
    n_perturbed_modules: int = 4
    sender_shift_selection_seed: int = 1701
    t_max: float = 20.0
    tissue_length: float = 1.0
    tissue_width: float = 0.28
    developmental_angle: float = np.pi / 4.0
    sample_gap: float = 1.5
    expression_scale: float = 4.0
    dispersion: float = 12.0
    unspliced_fraction: float = 0.45
    dropout_rate: float = 0.25
    unspliced_dropout: float = 0.40
    spliced_dropout: float = 0.20
    dropout_expression_protection: float = 3.5
    lr_dropout_scale: float = 0.02
    lr_observation_anchor_strength: float = 0.98
    sender_shift_concentration: float = 1.0
    differential_event_threshold: float = 0.015
    sender_share_threshold: float = 0.005
    seed: int = 42


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    scenario: str
    affected_fraction: float
    effect_strength: float


def experiment_specs() -> Tuple[ExperimentSpec, ...]:
    """Return the fixed 28-experiment benchmark panel."""
    specs: List[ExperimentSpec] = [
        ExperimentSpec("N00", "null", 0.0, 0.0)
    ]
    fractions = (0.05, 0.20, 0.40)
    strengths = (0.25, 0.50, 1.00)
    prefixes = {
        "activity": "A",
        "sender_shift": "R",
        "mixed": "M",
    }
    for scenario in ("activity", "sender_shift", "mixed"):
        index = 1
        for fraction in fractions:
            for strength in strengths:
                specs.append(
                    ExperimentSpec(
                        f"{prefixes[scenario]}{index:02d}",
                        scenario,
                        fraction,
                        strength,
                    )
                )
                index += 1
    return tuple(specs)


EXPERIMENT_IDS = tuple(spec.experiment_id for spec in experiment_specs())


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    lookup = {spec.experiment_id: spec for spec in experiment_specs()}
    try:
        return lookup[str(experiment_id).upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment {experiment_id!r}; expected one of {EXPERIMENT_IDS}"
        ) from exc


def _canonical_tissue(
    config: DifferentialSimulationConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_time, latent, coords, velocity, _ = _linear_development_tissue(
        config.n_cells_per_sample,
        config.t_max,
        config.tissue_length,
        config.tissue_width,
        config.developmental_angle,
        rng,
    )
    return true_time, latent, coords, velocity


def _base_expression_mean(
    latent: np.ndarray,
    coords: np.ndarray,
    config: DifferentialSimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    amplitude = rng.lognormal(0.0, 0.45, config.n_genes)
    pattern = _developmental_spatial_pattern(
        latent,
        coords,
        config.n_genes,
        rng,
        baseline_weight=0.12,
        temporal_weight=0.68,
        spatial_weight=0.20,
        width_high=0.40,
        spatial_preference_sd=0.65,
        spatial_center="mean",
    )
    return (
        config.expression_scale
        * amplitude[None, :]
        * pattern
    ).astype(np.float32)


def _module_definitions(
    base_mean: np.ndarray,
    coords: np.ndarray,
    latent: np.ndarray,
    config: DifferentialSimulationConfig,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    required = config.n_modules * 2
    if config.n_genes < required + 4:
        raise ValueError(
            "n_genes is too small for distinct ligand/receptor genes: "
            f"need at least {required + 4}"
        )
    gene_names = np.asarray([f"G{idx:04d}" for idx in range(config.n_genes)])
    modules: Dict[str, Dict[str, Any]] = {}
    pair_ids: List[str] = []
    pair_modules: List[str] = []
    pair_ligands: List[str] = []
    pair_receptors: List[str] = []
    used = set(range(required))
    target_pool = np.asarray(
        [idx for idx in range(config.n_genes) if idx not in used],
        dtype=np.int64,
    )
    target_size = min(
        config.target_genes_per_module,
        target_pool.size // config.n_modules,
    )
    if target_size < 1:
        raise ValueError(
            "n_genes is too small to allocate a distinct target gene "
            "to every module"
        )
    target_order = rng.permutation(target_pool)
    span = np.ptp(coords, axis=0)
    lo = np.min(coords, axis=0)

    grid_columns = int(np.ceil(np.sqrt(config.n_modules)))
    grid_rows = int(np.ceil(config.n_modules / grid_columns))
    for module_index in range(config.n_modules):
        module_id = f"module_{module_index}"
        ligand_idx = module_index
        receptor_idx = config.n_modules + module_index
        target_start = module_index * target_size
        target_idx = np.sort(
            target_order[target_start : target_start + target_size]
        ).astype(np.int64)
        grid_column = module_index % grid_columns
        grid_row = module_index // grid_columns
        column_fraction = grid_column / max(grid_columns - 1, 1)
        row_fraction = grid_row / max(grid_rows - 1, 1)
        source_a_point = lo + np.asarray(
            [
                (0.12 + 0.70 * column_fraction) * span[0],
                (0.20 + 0.58 * row_fraction) * span[1],
            ]
        )
        source_b_point = lo + np.asarray(
            [
                (0.82 - 0.64 * column_fraction) * span[0],
                (0.78 - 0.54 * row_fraction) * span[1],
            ]
        )
        source_a = _nearest_cell(coords, source_a_point)
        source_b = _nearest_cell(coords, source_b_point)
        center_a = coords[source_a].astype(np.float64)
        center_b = coords[source_b].astype(np.float64)
        tissue_scale = max(float(np.linalg.norm(span)), 1e-8)
        source_sigma = tissue_scale * (0.08 + 0.025 * (module_index % 3))
        field_a = np.exp(
            -0.5
            * (
                np.linalg.norm(coords - center_a[None, :], axis=1)
                / source_sigma
            )
            ** 2
        )
        field_b = np.exp(
            -0.5
            * (
                np.linalg.norm(coords - center_b[None, :], axis=1)
                / source_sigma
            )
            ** 2
        )
        receptor_peak = float(
            np.quantile(
                latent,
                (module_index + 0.5) / config.n_modules,
            )
        )
        receptor_field = np.exp(
            -0.5 * ((latent - receptor_peak) / 0.22) ** 2
        )
        ligand_control = 0.10 + 4.0 * field_a
        ligand_alternative = 0.10 + 4.0 * field_b
        receptor_expression = 0.10 + 3.0 * receptor_field
        base_mean[:, ligand_idx] = ligand_control
        base_mean[:, receptor_idx] = receptor_expression
        direction = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=target_idx.size,
            p=[0.25, 0.75],
        )
        ligand = str(gene_names[ligand_idx])
        receptor = str(gene_names[receptor_idx])
        pair_id = f"{module_id}:{ligand}->{receptor}"
        differential_range_type = "contact" if module_index % 2 == 0 else "diffusion"
        range_type = "local" if differential_range_type == "contact" else "regional"
        distance_scale = float(
            tissue_scale
            * (0.035 if differential_range_type == "contact" else 0.18)
        )
        modules[module_id] = {
            "module_id": module_id,
            "module_name": f"Differential communication module {module_index}",
            "ligand_gene": ligand,
            "receptor_gene": receptor,
            "ligand_genes": gene_names[ligand_idx : ligand_idx + 1],
            "receptor_genes": gene_names[receptor_idx : receptor_idx + 1],
            "ligand_gene_indices": np.asarray([ligand_idx], dtype=np.int64),
            "receptor_gene_indices": np.asarray([receptor_idx], dtype=np.int64),
            "target_gene_indices": target_idx,
            "target_genes": gene_names[target_idx],
            "effect_direction": direction,
            "effect_strength": float(rng.uniform(0.25, 0.60)),
            "range_type": range_type,
            "differential_range_type": differential_range_type,
            "distance_scale": distance_scale,
            "distance_decay": distance_scale,
            "source_a_cell": int(source_a),
            "source_b_cell": int(source_b),
            "ligand_control": ligand_control.astype(np.float32),
            "ligand_alternative": ligand_alternative.astype(np.float32),
            "receptor_expression": receptor_expression.astype(np.float32),
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


def _sender_ligand_program(
    module: Mapping[str, Any],
    effect_strength: float,
    concentration: float,
) -> np.ndarray:
    """Return the observable ligand field used to derive perturbed conditional sender shares."""
    mix = float(np.clip(effect_strength * concentration, 0.0, 1.0))
    return (
        (1.0 - mix) * np.asarray(module["ligand_control"], dtype=np.float32)
        + mix * np.asarray(module["ligand_alternative"], dtype=np.float32)
    ).astype(np.float32)


def _eligible_sender_shift_modules(
    group_module: np.ndarray,
    module_ids: List[str],
    max_modules: int,
) -> Tuple[str, ...]:
    """Return active modules eligible for sender-shift perturbation."""
    eligible_positions = list(range(min(max_modules, len(module_ids))))
    return tuple(
        module_ids[position]
        for position in eligible_positions
        if np.any(group_module == position)
    )


def _reference_sender_switch_rates(
    group_receiver: np.ndarray,
    group_module: np.ndarray,
    module_ids: List[str],
    modules: Dict[str, Dict[str, Any]],
    neighbors: np.ndarray,
    distances: np.ndarray,
    config: DifferentialSimulationConfig,
) -> np.ndarray:
    """Measure intrinsic switchability at all benchmark effect strengths."""
    switched = np.zeros(
        (len(module_ids), len(SENDER_SHIFT_REFERENCE_STRENGTHS)),
        dtype=np.int64,
    )
    totals = np.zeros(len(module_ids), dtype=np.int64)
    for receiver, module_position in zip(
        group_receiver,
        group_module,
        strict=True,
    ):
        receiver = int(receiver)
        module_position = int(module_position)
        module = modules[module_ids[module_position]]
        sender = neighbors[receiver]
        distance = distances[receiver]
        valid = (sender >= 0) & np.isfinite(distance)
        sender = sender[valid]
        distance = distance[valid]
        if sender.size == 0:
            continue
        p_control = _expression_distance_sender_share(
            module["ligand_control"],
            sender,
            distance,
            float(module["distance_scale"]),
        )
        totals[module_position] += 1
        control_top_sender = sender[np.argmax(p_control)]
        for strength_index, strength in enumerate(
            SENDER_SHIFT_REFERENCE_STRENGTHS
        ):
            reference_ligand = _sender_ligand_program(
                module,
                strength,
                config.sender_shift_concentration,
            )
            p_reference = _expression_distance_sender_share(
                reference_ligand,
                sender,
                distance,
                float(module["distance_scale"]),
            )
            switched[module_position, strength_index] += int(
                control_top_sender != sender[np.argmax(p_reference)]
            )
    rates = np.full(
        (len(module_ids), len(SENDER_SHIFT_REFERENCE_STRENGTHS)),
        np.nan,
        dtype=np.float32,
    )
    valid = totals > 0
    rates[valid] = switched[valid] / totals[valid, None]
    return rates


def _sender_shift_group_order(
    group_receiver: np.ndarray,
    group_module: np.ndarray,
    module_ids: List[str],
    modules: Dict[str, Dict[str, Any]],
    latent: np.ndarray,
    max_modules: int,
    selection_seed: int,
) -> np.ndarray:
    """Interleave spatially ordered groups from a small fixed LR vocabulary."""
    eligible_positions = [
        position
        for position in range(min(max_modules, len(module_ids)))
        if np.any(group_module == position)
    ]
    if not eligible_positions:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(selection_seed)
    module_order = list(
        np.asarray(eligible_positions, dtype=np.int64)[
            rng.permutation(len(eligible_positions))
        ]
    )
    ordered_by_module: Dict[int, np.ndarray] = {}
    for module_position in module_order:
        group_ids = np.flatnonzero(group_module == module_position)
        receivers = group_receiver[group_ids]
        module_id = module_ids[int(module_position)]
        source_b = int(modules[module_id]["source_b_cell"])
        distance = np.abs(
            latent[receivers] - float(latent[source_b])
        )
        tie_break = rng.random(group_ids.size)
        order = np.lexsort((tie_break, distance))
        ordered_by_module[int(module_position)] = group_ids[order]

    interleaved: List[int] = []
    max_groups = max(
        values.size
        for values in ordered_by_module.values()
    )
    for offset in range(max_groups):
        for module_position in module_order:
            values = ordered_by_module[int(module_position)]
            if offset < values.size:
                interleaved.append(int(values[offset]))
    return np.asarray(interleaved, dtype=np.int64)


def _select_sender_shift_groups(
    group_order: np.ndarray,
    affected_fraction: float,
    n_groups: int,
) -> np.ndarray:
    """Select an exact nested group prefix for the requested fraction."""
    if affected_fraction <= 0.0 or group_order.size == 0:
        return np.asarray([], dtype=np.int64)
    n_selected = int(round(float(affected_fraction) * n_groups))
    n_selected = max(1, min(n_selected, group_order.size))
    return np.asarray(group_order[:n_selected], dtype=np.int64)


def _localized_sender_ligand_programs(
    affected_group: np.ndarray,
    group_receiver: np.ndarray,
    group_module: np.ndarray,
    module_ids: List[str],
    modules: Dict[str, Dict[str, Any]],
    neighbors: np.ndarray,
    distances: np.ndarray,
    effect_strength: float,
    config: DifferentialSimulationConfig,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build observable local ligand shifts for selected receiver groups."""
    programs = {
        module_id: np.asarray(
            modules[module_id]["ligand_control"],
            dtype=np.float32,
        ).copy()
        for module_id in module_ids
    }
    touched_sender_count = np.zeros(len(module_ids), dtype=np.int64)
    for module_position, module_id in enumerate(module_ids):
        selected_groups = np.flatnonzero(
            affected_group & (group_module == module_position)
        )
        if selected_groups.size == 0:
            continue
        touched = np.zeros(programs[module_id].size, dtype=bool)
        control_program = np.asarray(
            modules[module_id]["ligand_control"],
            dtype=np.float32,
        )
        full_program = _sender_ligand_program(
            modules[module_id],
            effect_strength,
            config.sender_shift_concentration,
        )
        for group_index in selected_groups:
            receiver = int(group_receiver[group_index])
            valid = (
                (neighbors[receiver] >= 0)
                & np.isfinite(distances[receiver])
            )
            sender = neighbors[receiver][valid]
            if sender.size == 0:
                continue
            touched[int(sender[np.argmax(control_program[sender])])] = True
            touched[int(sender[np.argmax(full_program[sender])])] = True
        programs[module_id][touched] = full_program[touched]
        touched_sender_count[module_position] = int(np.sum(touched))
    return programs, touched_sender_count


def _communication_truth(
    spec: ExperimentSpec,
    modules: Dict[str, Dict[str, Any]],
    neighbors: np.ndarray,
    distances: np.ndarray,
    latent: np.ndarray,
    config: DifferentialSimulationConfig,
) -> Dict[str, Any]:
    module_ids = list(modules)
    n_cells = latent.size
    receptor_scores = np.column_stack(
        [modules[module_id]["receptor_expression"] for module_id in module_ids]
    )
    active = np.zeros((n_cells, len(module_ids)), dtype=bool)
    n_active = min(config.max_active_modules_per_receiver, len(module_ids))
    top = np.argpartition(-receptor_scores, n_active - 1, axis=1)[:, :n_active]
    rows = np.arange(n_cells)[:, None]
    active[rows, top] = True

    group_receiver, group_module = np.nonzero(active)
    n_groups = group_receiver.size
    module_group_count = np.asarray(
        [
            np.sum(group_module == position)
            for position in range(len(module_ids))
        ],
        dtype=np.int64,
    )
    eligible_sender_shift_modules = _eligible_sender_shift_modules(
        group_module,
        module_ids,
        config.n_perturbed_modules,
    )
    reference_sender_switch_rates = _reference_sender_switch_rates(
        group_receiver,
        group_module,
        module_ids,
        modules,
        neighbors,
        distances,
        config,
    )
    sender_shift_group_order = _sender_shift_group_order(
        group_receiver,
        group_module,
        module_ids,
        modules,
        latent,
        config.n_perturbed_modules,
        config.sender_shift_selection_seed,
    )
    eligible_modules = set(eligible_sender_shift_modules)
    eligible = np.asarray(
        [module_ids[int(module)] in eligible_modules for module in group_module],
        dtype=bool,
    )
    affected_group = np.zeros(n_groups, dtype=bool)
    sender_shift_modules: Tuple[str, ...] = ()
    sender_shift_group_ids = np.asarray([], dtype=np.int64)
    if spec.scenario in ("sender_shift", "mixed"):
        sender_shift_group_ids = _select_sender_shift_groups(
            sender_shift_group_order,
            spec.affected_fraction,
            n_groups,
        )
        affected_group[sender_shift_group_ids] = True
        selected_module_positions = set(
            int(position)
            for position in group_module[sender_shift_group_ids]
        )
        sender_shift_modules = tuple(
            module_id
            for position, module_id in enumerate(module_ids)
            if position in selected_module_positions
        )
    elif spec.scenario == "activity":
        n_affected = int(round(spec.affected_fraction * n_groups))
        candidate = np.flatnonzero(eligible)
        # Prefer contiguous receiver patches near each module's alternative
        # ligand source, matching the affected-location design of CytoSignal.
        score = np.empty(candidate.size, dtype=np.float64)
        for pos, group_idx in enumerate(candidate):
            receiver = int(group_receiver[group_idx])
            module_id = module_ids[int(group_module[group_idx])]
            source_b = int(modules[module_id]["source_b_cell"])
            score[pos] = -abs(float(latent[receiver] - latent[source_b]))
        chosen = candidate[np.argsort(-score)[: min(n_affected, candidate.size)]]
        affected_group[chosen] = True

    if spec.scenario in ("sender_shift", "mixed"):
        (
            perturbed_ligand_programs,
            module_touched_sender_count,
        ) = _localized_sender_ligand_programs(
            affected_group,
            group_receiver,
            group_module,
            module_ids,
            modules,
            neighbors,
            distances,
            spec.effect_strength,
            config,
        )
    else:
        perturbed_ligand_programs = {
            module_id: np.asarray(
                modules[module_id]["ligand_control"],
                dtype=np.float32,
            ).copy()
            for module_id in module_ids
        }
        module_touched_sender_count = np.zeros(
            len(module_ids),
            dtype=np.int64,
        )

    event_sender: List[int] = []
    event_receiver: List[int] = []
    event_module: List[str] = []
    event_group: List[int] = []
    p_control: List[float] = []
    p_perturbed: List[float] = []
    q_control: List[float] = []
    q_perturbed: List[float] = []
    a_control: List[float] = []
    a_perturbed: List[float] = []
    event_distance: List[float] = []

    group_q0 = np.zeros(n_groups, dtype=np.float32)
    group_q1 = np.zeros(n_groups, dtype=np.float32)
    group_top0 = np.zeros(n_groups, dtype=np.int64)
    group_top1 = np.zeros(n_groups, dtype=np.int64)

    for group_index, (receiver, module_pos) in enumerate(
        zip(group_receiver, group_module, strict=True)
    ):
        receiver = int(receiver)
        module_id = module_ids[int(module_pos)]
        module = modules[module_id]
        sender = neighbors[receiver]
        distance = distances[receiver]
        valid = (sender >= 0) & np.isfinite(distance)
        sender = sender[valid]
        distance = distance[valid]
        if sender.size == 0:
            continue
        p0 = _expression_distance_sender_share(
            module["ligand_control"],
            sender,
            distance,
            float(module["distance_scale"]),
        )
        is_affected = bool(affected_group[group_index])
        p1 = _expression_distance_sender_share(
            perturbed_ligand_programs[module_id],
            sender,
            distance,
            float(module["distance_scale"]),
        )

        receptor = float(module["receptor_expression"][receiver])
        q0 = 0.20 + 0.75 * receptor / (
            receptor + float(np.median(module["receptor_expression"]))
        )
        change_q = spec.scenario in ("activity", "mixed") and is_affected
        q1 = q0 * (1.0 + spec.effect_strength) if change_q else q0
        group_q0[group_index] = q0
        group_q1[group_index] = q1
        group_top0[group_index] = int(sender[np.argmax(p0)])
        group_top1[group_index] = int(sender[np.argmax(p1)])

        event_sender.extend(int(value) for value in sender)
        event_receiver.extend([receiver] * sender.size)
        event_module.extend([module_id] * sender.size)
        event_group.extend([group_index] * sender.size)
        p_control.extend(float(value) for value in p0)
        p_perturbed.extend(float(value) for value in p1)
        q_control.extend([q0] * sender.size)
        q_perturbed.extend([q1] * sender.size)
        a_control.extend(float(q0 * value) for value in p0)
        a_perturbed.extend(float(q1 * value) for value in p1)
        event_distance.extend(float(value) for value in distance)

    p0_arr = np.asarray(p_control, dtype=np.float32)
    p1_arr = np.asarray(p_perturbed, dtype=np.float32)
    q0_arr = np.asarray(q_control, dtype=np.float32)
    q1_arr = np.asarray(q_perturbed, dtype=np.float32)
    a0_arr = np.asarray(a_control, dtype=np.float32)
    a1_arr = np.asarray(a_perturbed, dtype=np.float32)
    delta_a = a1_arr - a0_arr
    activity_signal = np.abs((q1_arr - q0_arr) * p0_arr)
    sender_signal = np.abs(p1_arr - p0_arr)
    activity_differential = (
        activity_signal >= config.differential_event_threshold
    )
    sender_differential = (
        sender_signal >= config.sender_share_threshold
    )
    differential = activity_differential | sender_differential
    event_group_arr = np.asarray(event_group, dtype=np.int64)
    group_sender_share_changed = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(
        group_sender_share_changed,
        event_group_arr,
        sender_differential,
    )
    group_top_sender_switched = group_top0 != group_top1
    module_affected_group_count = np.asarray(
        [
            np.sum(affected_group & (group_module == position))
            for position in range(len(module_ids))
        ],
        dtype=np.int64,
    )
    module_top_sender_switched_count = np.asarray(
        [
            np.sum(
                affected_group
                & group_top_sender_switched
                & (group_module == position)
            )
            for position in range(len(module_ids))
        ],
        dtype=np.int64,
    )
    module_sender_share_changed_count = np.asarray(
        [
            np.sum(
                group_sender_share_changed
                & (group_module == position)
            )
            for position in range(len(module_ids))
        ],
        dtype=np.int64,
    )
    module_affected_top_sender_switch_rate = np.full(
        len(module_ids),
        np.nan,
        dtype=np.float32,
    )
    has_affected = module_affected_group_count > 0
    module_affected_top_sender_switch_rate[has_affected] = (
        module_top_sender_switched_count[has_affected]
        / module_affected_group_count[has_affected]
    )
    affected_top_sender_switch_rate = (
        float(np.mean(group_top_sender_switched[affected_group]))
        if np.any(affected_group)
        else float("nan")
    )
    return {
        "event_sender_id": np.asarray(event_sender, dtype=np.int64),
        "event_receiver_id": np.asarray(event_receiver, dtype=np.int64),
        "event_module_id": np.asarray(event_module, dtype=object),
        "event_group_id": event_group_arr,
        "event_distance": np.asarray(event_distance, dtype=np.float32),
        "p_control": p0_arr,
        "p_perturbed": p1_arr,
        "q_control": q0_arr,
        "q_perturbed": q1_arr,
        "a_control": a0_arr,
        "a_perturbed": a1_arr,
        "delta_p": (p1_arr - p0_arr).astype(np.float32),
        "delta_q": (q1_arr - q0_arr).astype(np.float32),
        "delta_simulated_event_influence": delta_a.astype(np.float32),
        "is_differential_event": differential,
        "group_receiver_id": group_receiver.astype(np.int64),
        "group_module_id": np.asarray(
            [module_ids[int(pos)] for pos in group_module],
            dtype=object,
        ),
        "group_is_affected": affected_group,
        "group_q_control": group_q0,
        "group_q_perturbed": group_q1,
        "group_top_sender_control": group_top0,
        "group_top_sender_perturbed": group_top1,
        "group_top_sender_switched": group_top_sender_switched,
        "group_sender_share_changed": (
            group_sender_share_changed
        ),
        "eligible_sender_shift_module_ids": np.asarray(
            eligible_sender_shift_modules,
            dtype="U32",
        ),
        "sender_shift_group_order": np.asarray(
            sender_shift_group_order,
            dtype=np.int64,
        ),
        "sender_shift_group_ids": np.asarray(
            sender_shift_group_ids,
            dtype=np.int64,
        ),
        "sender_shift_module_ids": np.asarray(
            sender_shift_modules,
            dtype="U32",
        ),
        "module_audit_ids": np.asarray(module_ids, dtype="U32"),
        "module_group_count": module_group_count,
        "module_affected_group_count": module_affected_group_count,
        "module_top_sender_switched_count": module_top_sender_switched_count,
        "module_sender_share_changed_count": (
            module_sender_share_changed_count
        ),
        "module_touched_sender_count": module_touched_sender_count,
        "module_affected_top_sender_switch_rate": (
            module_affected_top_sender_switch_rate
        ),
        "module_reference_top_sender_switch_rate": (
            reference_sender_switch_rates
        ),
        "affected_top_sender_switch_rate": affected_top_sender_switch_rate,
        "realized_sender_share_change_fraction": float(
            np.mean(group_sender_share_changed)
        ),
        "sender_share_change_spillover_fraction": float(
            np.mean(
                group_sender_share_changed
                & ~affected_group
            )
        ),
        "module_perturbed_ligand_programs": perturbed_ligand_programs,
        "effect_strength": float(spec.effect_strength),
    }


def _target_effect(
    condition: str,
    truth: Dict[str, Any],
    modules: Dict[str, Dict[str, Any]],
    n_cells: int,
    n_genes: int,
) -> np.ndarray:
    effect = np.zeros((n_cells, n_genes), dtype=np.float32)
    q_key = "group_q_control" if condition == "control" else "group_q_perturbed"
    for receiver, module_id, q in zip(
        truth["group_receiver_id"],
        truth["group_module_id"],
        truth[q_key],
        strict=True,
    ):
        module = modules[str(module_id)]
        effect[int(receiver), module["target_gene_indices"]] += (
            float(q)
            * float(module["effect_strength"])
            * module["effect_direction"]
        ).astype(np.float32)
    return effect


def _communication_velocities(
    truth: Dict[str, Any],
    coords: np.ndarray,
    intrinsic_velocity: np.ndarray,
) -> Dict[str, np.ndarray]:
    raw_by_condition: Dict[str, np.ndarray] = {}
    for condition, interaction_score in (
        ("control", truth["a_control"]),
        ("perturbed", truth["a_perturbed"]),
    ):
        raw = np.zeros_like(intrinsic_velocity, dtype=np.float32)
        for sender, receiver, score in zip(
            truth["event_sender_id"],
            truth["event_receiver_id"],
            interaction_score,
            strict=True,
        ):
            delta = coords[int(receiver)] - coords[int(sender)]
            norm = float(np.linalg.norm(delta))
            direction = (
                delta / norm
                if norm > 1e-8
                else intrinsic_velocity[int(receiver)]
            )
            forward = intrinsic_velocity[int(receiver)]
            if float(np.dot(direction, forward)) < 0.0:
                direction = -direction
            raw[int(receiver)] += float(score) * direction
        raw_by_condition[condition] = raw

    raw_norm = np.concatenate(
        [
            np.linalg.norm(raw, axis=1)
            for raw in raw_by_condition.values()
        ]
    )
    intrinsic_norm = np.linalg.norm(intrinsic_velocity, axis=1)
    if np.any(raw_norm > 1e-8):
        scale = (
            0.35
            * float(np.percentile(intrinsic_norm[intrinsic_norm > 1e-8], 95))
            / max(float(np.percentile(raw_norm[raw_norm > 1e-8], 95)), 1e-8)
        )
    else:
        scale = 0.0
    return {
        condition: (raw * scale).astype(np.float32)
        for condition, raw in raw_by_condition.items()
    }


def _condition_lr_program(
    condition: str,
    base_mean: np.ndarray,
    modules: Dict[str, Dict[str, Any]],
    truth: Dict[str, Any],
    spec: ExperimentSpec,
    config: DifferentialSimulationConfig,
) -> np.ndarray:
    mean = np.asarray(base_mean, dtype=np.float32).copy()
    affected_lookup = {
        (int(receiver), str(module))
        for receiver, module, affected in zip(
            truth["group_receiver_id"],
            truth["group_module_id"],
            truth["group_is_affected"],
            strict=True,
        )
        if affected
    }
    for module_id, module in modules.items():
        ligand_idx = int(module["ligand_gene_indices"][0])
        receptor_idx = int(module["receptor_gene_indices"][0])
        if condition == "perturbed":
            mean[:, ligand_idx] = truth[
                "module_perturbed_ligand_programs"
            ][module_id]
        else:
            mean[:, ligand_idx] = module["ligand_control"]
        mean[:, receptor_idx] = module["receptor_expression"]
        if (
            condition == "perturbed"
            and spec.scenario in ("activity", "mixed")
        ):
            receiver_ids = np.asarray(
                [
                    receiver
                    for receiver, candidate_module in affected_lookup
                    if candidate_module == module_id
                ],
                dtype=np.int64,
            )
            if receiver_ids.size:
                mean[receiver_ids, receptor_idx] *= 1.0 + spec.effect_strength
    return np.clip(mean, 0.02, None)


def _sample_events(
    condition: str,
    sample_offset: int,
    sample_id: str,
    replicate: int,
    truth: Dict[str, Any],
    modules: Dict[str, Dict[str, Any]],
    condition_mean: np.ndarray,
) -> Dict[str, np.ndarray]:
    sender = truth["event_sender_id"]
    receiver = truth["event_receiver_id"]
    module = truth["event_module_id"]
    group = truth["event_group_id"]
    ligand_score = np.asarray(
        [
            condition_mean[
                int(sender_id),
                int(modules[str(module_id)]["ligand_gene_indices"][0]),
            ]
            for sender_id, module_id in zip(sender, module, strict=True)
        ],
        dtype=np.float32,
    )
    receptor_score = np.asarray(
        [
            condition_mean[
                int(receiver_id),
                int(modules[str(module_id)]["receptor_gene_indices"][0]),
            ]
            for receiver_id, module_id in zip(receiver, module, strict=True)
        ],
        dtype=np.float32,
    )
    effect_magnitude = (
        np.asarray(truth["group_is_affected"], dtype=np.float32)[group]
        * float(truth["effect_strength"])
    ).astype(np.float32)
    return {
        "sender_id": (sender + sample_offset).astype(np.int64),
        "receiver_id": (receiver + sample_offset).astype(np.int64),
        "canonical_event_id": np.arange(sender.size, dtype=np.int64),
        "canonical_sender_id": sender.astype(np.int64),
        "canonical_receiver_id": receiver.astype(np.int64),
        "event_group_id": group.astype(np.int64),
        "module_id": module.astype(object),
        "sample_id": np.full(sender.size, sample_id, dtype=object),
        "condition": np.full(sender.size, condition, dtype=object),
        "replicate": np.full(sender.size, replicate, dtype=np.int64),
        "label": np.full(sender.size, "positive", dtype=object),
        "negative_type": np.full(sender.size, "", dtype=object),
        "is_differential_event": np.asarray(
            truth["is_differential_event"],
            dtype=bool,
        ),
        "is_affected_group": np.asarray(
            truth["group_is_affected"],
            dtype=bool,
        )[group],
        "effect_magnitude": effect_magnitude,
        "distance": np.asarray(truth["event_distance"], dtype=np.float32),
        "ligand_score": ligand_score,
        "receptor_score": receptor_score,
    }


def _public_differential_truth(
    truth: Dict[str, Any],
    spec: ExperimentSpec,
    config: DifferentialSimulationConfig,
) -> Dict[str, Any]:
    """Export differential labels without hidden simulated event influences."""
    return {
        "experiment_id": spec.experiment_id,
        "scenario": spec.scenario,
        "affected_fraction": float(spec.affected_fraction),
        "effect_strength": float(spec.effect_strength),
        "realized_affected_fraction": float(
            np.mean(np.asarray(truth["group_is_affected"], dtype=bool))
        ),
        "perturbed_module_ids": np.asarray(
            truth["sender_shift_module_ids"],
            dtype="U32",
        ),
        "eligible_sender_shift_module_ids": np.asarray(
            truth["eligible_sender_shift_module_ids"],
            dtype="U32",
        ),
        "sender_shift_group_order": np.asarray(
            truth["sender_shift_group_order"],
            dtype=np.int64,
        ),
        "sender_shift_group_ids": np.asarray(
            truth["sender_shift_group_ids"],
            dtype=np.int64,
        ),
        "sender_shift_selection_seed": int(
            config.sender_shift_selection_seed
        ),
        "sender_shift_selection_strategy": (
            "nested_spatial_receiver_group_prefix"
        ),
        "sender_shift_reference_effect_strengths": np.asarray(
            SENDER_SHIFT_REFERENCE_STRENGTHS,
            dtype=np.float32,
        ),
        "nested_sender_shift_design": True,
        "n_receiver_module_groups": int(
            np.asarray(truth["group_is_affected"]).size
        ),
        "n_affected_receiver_module_groups": int(
            np.sum(np.asarray(truth["group_is_affected"], dtype=bool))
        ),
        "affected_top_sender_switch_rate": float(
            truth["affected_top_sender_switch_rate"]
        ),
        "realized_sender_share_change_fraction": float(
            truth["realized_sender_share_change_fraction"]
        ),
        "sender_share_change_spillover_fraction": float(
            truth["sender_share_change_spillover_fraction"]
        ),
        "module_audit": {
            "module_id": np.asarray(truth["module_audit_ids"], dtype="U32"),
            "group_count": np.asarray(
                truth["module_group_count"],
                dtype=np.int64,
            ),
            "affected_group_count": np.asarray(
                truth["module_affected_group_count"],
                dtype=np.int64,
            ),
            "top_sender_switched_count": np.asarray(
                truth["module_top_sender_switched_count"],
                dtype=np.int64,
            ),
            "sender_share_changed_count": np.asarray(
                truth["module_sender_share_changed_count"],
                dtype=np.int64,
            ),
            "touched_sender_count": np.asarray(
                truth["module_touched_sender_count"],
                dtype=np.int64,
            ),
            "affected_top_sender_switch_rate": np.asarray(
                truth["module_affected_top_sender_switch_rate"],
                dtype=np.float32,
            ),
            "reference_top_sender_switch_rate": np.asarray(
                truth["module_reference_top_sender_switch_rate"],
                dtype=np.float32,
            ),
        },
        "event_sender_id": np.asarray(truth["event_sender_id"], dtype=np.int64),
        "event_receiver_id": np.asarray(truth["event_receiver_id"], dtype=np.int64),
        "event_module_id": np.asarray(truth["event_module_id"], dtype=object),
        "event_group_id": np.asarray(truth["event_group_id"], dtype=np.int64),
        "event_distance": np.asarray(truth["event_distance"], dtype=np.float32),
        "is_differential_event": np.asarray(
            truth["is_differential_event"],
            dtype=bool,
        ),
        "group_receiver_id": np.asarray(truth["group_receiver_id"], dtype=np.int64),
        "group_module_id": np.asarray(truth["group_module_id"], dtype=object),
        "group_is_affected": np.asarray(truth["group_is_affected"], dtype=bool),
        "group_top_sender_switched": np.asarray(
            truth["group_top_sender_switched"],
            dtype=bool,
        ),
        "group_sender_share_changed": np.asarray(
            truth["group_sender_share_changed"],
            dtype=bool,
        ),
        "differential_event_threshold": float(config.differential_event_threshold),
        "sender_share_threshold": float(
            config.sender_share_threshold
        ),
    }


def simulate_differential_experiment(
    spec: ExperimentSpec | str,
    config: DifferentialSimulationConfig = DifferentialSimulationConfig(),
) -> ad.AnnData:
    """Generate one eight-sample differential benchmark AnnData."""
    if isinstance(spec, str):
        spec = get_experiment_spec(spec)
    if spec.scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {spec.scenario!r}; expected {SCENARIOS}")
    if config.n_cells_per_sample < 50:
        raise ValueError("n_cells_per_sample must be at least 50")
    if config.n_replicates < 2:
        raise ValueError("n_replicates must be at least 2")
    if config.n_modules < 1:
        raise ValueError("n_modules must be at least 1")
    if config.max_active_modules_per_receiver < 1:
        raise ValueError("max_active_modules_per_receiver must be at least 1")
    if config.n_perturbed_modules < 0:
        raise ValueError("n_perturbed_modules must be non-negative")
    if not 0.0 <= config.sender_shift_concentration <= 1.0:
        raise ValueError("sender_shift_concentration must be between 0 and 1")
    if config.differential_event_threshold < 0.0:
        raise ValueError("differential_event_threshold must be non-negative")
    if config.sender_share_threshold < 0.0:
        raise ValueError("sender_share_threshold must be non-negative")
    if not 0.0 <= spec.affected_fraction <= 1.0:
        raise ValueError("affected_fraction must be between 0 and 1")
    if not 0.0 <= spec.effect_strength <= 1.0:
        raise ValueError("effect_strength must be between 0 and 1")

    tissue_rng = np.random.default_rng(config.seed + 101)
    expression_rng = np.random.default_rng(config.seed + 202)
    module_rng = np.random.default_rng(config.seed + 303)
    true_time, latent, canonical_coords, intrinsic_velocity = _canonical_tissue(
        config,
        tissue_rng,
    )
    baseline_mean = _base_expression_mean(
        latent,
        canonical_coords,
        config,
        expression_rng,
    )
    base_mean = baseline_mean.copy()
    modules, lr_pairs = _module_definitions(
        base_mean,
        canonical_coords,
        latent,
        config,
        module_rng,
    )
    lr_gene_indices = _ccc_lr_gene_indices(modules)
    neighbors, distances = _spatial_neighbor_candidates(
        canonical_coords,
        config.n_sender_neighbors,
    )
    truth = _communication_truth(
        spec,
        modules,
        neighbors,
        distances,
        latent,
        config,
    )
    communication_velocity_by_condition = _communication_velocities(
        truth,
        canonical_coords,
        intrinsic_velocity,
    )

    matrices_u: List[sparse.csr_matrix] = []
    matrices_s: List[sparse.csr_matrix] = []
    baseline_means: List[sparse.csr_matrix] = []
    lr_programmed_means: List[sparse.csr_matrix] = []
    expression_effects: List[sparse.csr_matrix] = []
    means: List[sparse.csr_matrix] = []
    spatial_parts: List[np.ndarray] = []
    intrinsic_parts: List[np.ndarray] = []
    communication_parts: List[np.ndarray] = []
    true_velocity_parts: List[np.ndarray] = []
    obs_rows: List[Dict[str, Any]] = []
    event_tables: List[Dict[str, np.ndarray]] = []
    sample_index = 0
    replicate_rng = np.random.default_rng(config.seed + 404)
    replicate_library_factors = replicate_rng.lognormal(
        mean=0.0,
        sigma=0.12,
        size=config.n_replicates,
    )

    for condition_index, condition in enumerate(("control", "perturbed")):
        condition_program = _condition_lr_program(
            condition,
            base_mean,
            modules,
            truth,
            spec,
            config,
        )
        condition_effect = _target_effect(
            condition,
            truth,
            modules,
            condition_program.shape[0],
            condition_program.shape[1],
        )
        communication_velocity = communication_velocity_by_condition[condition]
        for replicate in range(config.n_replicates):
            sample_id = f"{condition}_{replicate}"
            rng = np.random.default_rng(
                config.seed
                + 10007 * (condition_index + 1)
                + 1009 * replicate
            )
            library_factor = float(replicate_library_factors[replicate])
            replicate_baseline = baseline_mean * library_factor
            replicate_program = condition_program * library_factor
            replicate_effect = condition_effect * library_factor
            replicate_mean = np.clip(replicate_program + replicate_effect, 0.02, None)
            spliced = _ccc_negative_binomial_counts(
                replicate_mean,
                config.dispersion,
                config.spliced_dropout,
                config.dropout_expression_protection,
                rng,
                protected_gene_indices=lr_gene_indices,
                protected_dropout_scale=config.lr_dropout_scale,
            )
            unspliced_mean = (
                config.unspliced_fraction * replicate_mean
                + 0.08
                * np.maximum(
                    np.gradient(replicate_mean, axis=0),
                    0.0,
                )
            )
            unspliced = _ccc_negative_binomial_counts(
                np.clip(unspliced_mean, 0.02, None),
                config.dispersion,
                config.unspliced_dropout,
                config.dropout_expression_protection,
                rng,
                protected_gene_indices=lr_gene_indices,
                protected_dropout_scale=config.lr_dropout_scale,
            )
            unspliced, spliced = _ccc_anchor_lr_observation_to_drive(
                unspliced,
                spliced,
                replicate_mean,
                lr_gene_indices,
                config.lr_observation_anchor_strength,
            )
            offset = np.asarray(
                [
                    sample_index
                    * (config.tissue_length + config.sample_gap),
                    0.0,
                ],
                dtype=np.float32,
            )
            sample_coords = (
                canonical_coords
                + offset[None, :]
                + rng.normal(0.0, 0.002, canonical_coords.shape)
            ).astype(np.float32)
            start = sample_index * config.n_cells_per_sample
            matrices_u.append(sparse.csr_matrix(unspliced))
            matrices_s.append(sparse.csr_matrix(spliced))
            baseline_means.append(
                sparse.csr_matrix(replicate_baseline.astype(np.float32))
            )
            lr_programmed_means.append(
                sparse.csr_matrix(replicate_program.astype(np.float32))
            )
            expression_effects.append(
                sparse.csr_matrix(replicate_effect.astype(np.float32))
            )
            means.append(sparse.csr_matrix(replicate_mean.astype(np.float32)))
            spatial_parts.append(sample_coords)
            intrinsic_parts.append(intrinsic_velocity.copy())
            communication_parts.append(communication_velocity.copy())
            true_velocity_parts.append(
                (intrinsic_velocity + communication_velocity).astype(np.float32)
            )
            for cell_id in range(config.n_cells_per_sample):
                obs_rows.append(
                    {
                        "sample_id": sample_id,
                        "condition": condition,
                        "replicate": str(replicate),
                        "canonical_cell_id": int(cell_id),
                        "true_time": float(true_time[cell_id]),
                        "latent_time": float(latent[cell_id]),
                        "clusters": str(
                            min(4, int(np.floor(5.0 * float(latent[cell_id]))))
                        ),
                    }
                )
            event_tables.append(
                _sample_events(
                    condition,
                    start,
                    sample_id,
                    replicate,
                    truth,
                    modules,
                    condition_program,
                )
            )
            sample_index += 1

    obs = {
        key: np.asarray([row[key] for row in obs_rows])
        for key in obs_rows[0]
    }
    gene_names = np.asarray([f"G{idx:04d}" for idx in range(config.n_genes)])
    unspliced_stack = sparse.vstack(matrices_u, format="csr")
    spliced_stack = sparse.vstack(matrices_s, format="csr")
    baseline_stack = sparse.vstack(baseline_means, format="csr")
    lr_programmed_stack = sparse.vstack(lr_programmed_means, format="csr")
    expression_effect_stack = sparse.vstack(expression_effects, format="csr")
    expected_stack = sparse.vstack(means, format="csr")
    true_alpha_baseline = np.asarray(baseline_stack.mean(axis=0)).ravel().astype(
        np.float32
    )
    true_alpha_lr_programmed = np.asarray(
        lr_programmed_stack.mean(axis=0)
    ).ravel().astype(np.float32)
    adata = ad.AnnData(
        X=spliced_stack,
        obs=obs,
        var={
            "gene_name": gene_names,
            "true_alpha_baseline": true_alpha_baseline,
            "true_alpha_lr_programmed": true_alpha_lr_programmed,
        },
        layers={
            "unspliced": unspliced_stack,
            "spliced": spliced_stack,
            "baseline_transcription_drive": baseline_stack,
            "lr_programmed_transcription_drive": lr_programmed_stack,
            "communication_expression_effect": expression_effect_stack,
            "expected_expression": expected_stack,
        },
    )
    adata.obs_names = np.asarray(
        [
            f"{row['sample_id']}_{row['canonical_cell_id']}"
            for row in obs_rows
        ]
    )
    adata.var_names = gene_names
    adata.obsm["X_spatial"] = np.concatenate(spatial_parts, axis=0)
    adata.obsm["spatial"] = adata.obsm["X_spatial"].copy()
    adata.obsm["intrinsic_spatial_velocity"] = np.concatenate(
        intrinsic_parts,
        axis=0,
    )
    adata.obsm["communication_spatial_velocity"] = np.concatenate(
        communication_parts,
        axis=0,
    )
    adata.obsm["true_spatial_velocity"] = np.concatenate(
        true_velocity_parts,
        axis=0,
    )
    events = _concat_tables(event_tables)
    communication_summary = _ccc_communication_event_summary(
        events,
        np.asarray(obs["latent_time"], dtype=np.float32),
    )
    adata.uns["ground_truth"] = {
        "modules": modules,
        "lr_pairs": lr_pairs,
        "events": events,
        "communication_summary": communication_summary,
        "lr_expression_program": "lr_programmed_transcription_drive",
        "lr_dropout_scale": float(config.lr_dropout_scale),
        "lr_observation_anchor_strength": float(
            config.lr_observation_anchor_strength
        ),
        "dropout_expression_protection": float(
            config.dropout_expression_protection
        ),
        "differential": _public_differential_truth(truth, spec, config),
        "design": {
            "experiment_id": spec.experiment_id,
            "scenario": spec.scenario,
            "affected_fraction": float(spec.affected_fraction),
            "realized_affected_fraction": float(
                np.mean(
                    np.asarray(
                        truth["group_is_affected"],
                        dtype=bool,
                    )
                )
            ),
            "effect_strength": float(spec.effect_strength),
            "n_replicates_per_condition": int(config.n_replicates),
            "n_samples": int(2 * config.n_replicates),
            "n_cells_per_sample": int(config.n_cells_per_sample),
            "n_modules": int(config.n_modules),
            "n_perturbed_modules": int(
                min(config.n_perturbed_modules, config.n_modules)
            ),
            "sender_shift_selection_seed": int(
                config.sender_shift_selection_seed
            ),
            "sender_shift_selection_strategy": (
                "nested_spatial_receiver_group_prefix"
            ),
            "sender_shift_reference_effect_strengths": np.asarray(
                SENDER_SHIFT_REFERENCE_STRENGTHS,
                dtype=np.float32,
            ),
            "nested_sender_shift_design": True,
            "paired_canonical_cells": True,
            "sample_coordinates_separated": True,
            "differential_axes": [
                "receiver_activity",
                "sender_choice",
            ],
        },
        "random_seed": int(config.seed),
    }
    adata.uns["benchmark_version"] = BENCHMARK_VERSION
    adata.uns["benchmark_configuration"] = {
        key: value
        for key, value in vars(config).items()
        if isinstance(value, (str, int, float, bool))
    }
    adata.uns["dataset_name"] = spec.experiment_id
    return adata


logger = logging.getLogger(__name__)

class DifferentialSimulationPipe:
    """Generate one merged eight-sample h5ad per differential experiment."""

    def __init__(
        self,
        output_dir: str | Path,
        config: DifferentialSimulationConfig = DifferentialSimulationConfig(),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def run(
        self,
        experiments: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        selected = list(EXPERIMENT_IDS if experiments is None else experiments)
        outputs: List[Path] = []
        for experiment_id in selected:
            spec = get_experiment_spec(experiment_id)
            logger.info(
                "Generating %s: scenario=%s fraction=%.2f strength=%.2f",
                spec.experiment_id,
                spec.scenario,
                spec.affected_fraction,
                spec.effect_strength,
            )
            adata = simulate_differential_experiment(spec, self.config)
            path = save_h5ad(
                adata,
                self.output_dir / f"{spec.experiment_id}.h5ad",
            )
            differential = adata.uns["ground_truth"]["differential"]
            realized_fraction = float(
                differential["realized_affected_fraction"]
            )
            switch_rate = float(
                differential["affected_top_sender_switch_rate"]
            )
            logger.info(
                "Wrote %s (%d samples, %d differential events, "
                "realized affected fraction %.3f, %d selected groups, "
                "sender-probability change fraction %.3f, "
                "affected top-sender switch rate %.3f)",
                path,
                int(adata.obs["sample_id"].nunique()),
                int(differential["is_differential_event"].sum()),
                realized_fraction,
                int(differential["n_affected_receiver_module_groups"]),
                float(
                    differential[
                        "realized_sender_share_change_fraction"
                    ]
                ),
                switch_rate,
            )
            if (
                spec.scenario in ("sender_shift", "mixed")
                and abs(realized_fraction - spec.affected_fraction) > 0.02
            ):
                logger.warning(
                    "%s realized affected fraction %.3f differs from "
                    "the nominal %.3f by more than 0.02",
                    spec.experiment_id,
                    realized_fraction,
                    spec.affected_fraction,
                )
            outputs.append(path)
        return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/data") / "differential" / "raw",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        choices=EXPERIMENT_IDS,
        help="Generate only this experiment; repeat to select several.",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=DifferentialSimulationConfig.n_cells_per_sample,
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=DifferentialSimulationConfig.n_genes,
    )
    parser.add_argument(
        "--n-modules",
        type=int,
        default=DifferentialSimulationConfig.n_modules,
    )
    parser.add_argument(
        "--n-perturbed-modules",
        type=int,
        default=DifferentialSimulationConfig.n_perturbed_modules,
    )
    parser.add_argument(
        "--sender-shift-selection-seed",
        type=int,
        default=DifferentialSimulationConfig.sender_shift_selection_seed,
        help=(
            "Seed for the fixed nested sender-shift group order. Use a "
            "different output directory for each seed replicate."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DifferentialSimulationConfig.n_replicates,
    )
    parser.add_argument("--seed", type=int, default=DifferentialSimulationConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = DifferentialSimulationConfig(
        n_cells_per_sample=args.n_cells,
        n_genes=args.n_genes,
        n_modules=args.n_modules,
        n_perturbed_modules=args.n_perturbed_modules,
        n_replicates=args.replicates,
        sender_shift_selection_seed=args.sender_shift_selection_seed,
        seed=args.seed,
    )
    DifferentialSimulationPipe(args.output_dir, config).run(args.experiments)


if __name__ == "__main__":
    main()
