"""Developmental-direction simulation pipe and implementation."""
from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import anndata
import anndata as ad
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.simulation.core import (
    EVENT_ATTRIBUTION_MODES,
    save_h5ad,
    _assign_regions,
    _normalize_vector,
    _state_labels,
    _anchor_lr_observation_to_drive as _ccc_anchor_lr_observation_to_drive,
    _apply_lr_expression_program as _ccc_apply_lr_expression_program,
    _build_communication_truth as _ccc_build_communication_truth,
    _build_modules as _ccc_build_modules,
    _expression_dependent_dropout as _ccc_expression_dependent_dropout,
    _lr_gene_indices as _ccc_lr_gene_indices,
    _stabilize_lr_kinetics as _ccc_stabilize_lr_kinetics,
)

_topovelo_ready = False


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for adding communication ground truth to synthetic data."""

    n_cells: int = 3000
    n_genes: int = 500
    n_modules: int = 4
    target_genes_per_module: int = 24
    max_active_modules_per_receiver: int = 2
    n_sender_neighbors: int = 90
    intrinsic_velocity_weight: float = 1.0
    communication_velocity_weight: float = 0.5
    simulated_event_influence_alpha: float = 2.0
    simulated_event_influence_beta: float = 5.0
    max_positive_senders_per_receiver_module: int = 4
    expression_effect_strength: float = 0.16
    max_hard_negative_events: int = 20000
    lr_gene_expr_quantile: float = 0.6
    event_attribution_mode: str = "graded_paracrine"
    t_max: float = 20.0
    sender_ligand_feedback_strength: float = 0.75
    lr_program_strength: float = 6.0
    lr_program_sharpness: float = 2.5
    lr_program_active_fraction: float = 0.20
    lr_program_background: float = 0.01
    module_receptor_suppression: float = 0.02
    unspliced_dropout: float = 0.75
    spliced_dropout: float = 0.45
    dropout_expression_protection: float = 2.5
    lr_dropout_scale: float = 0.08
    lr_observation_anchor_strength: float = 0.85
    seed: int = 42


@dataclass
class CommunicationTruth:
    obs: Dict[str, np.ndarray]
    obsm: Dict[str, np.ndarray]
    expression_effect: np.ndarray
    baseline_drive: np.ndarray
    lr_programmed_drive: np.ndarray
    lr_gene_indices: np.ndarray
    uns: Dict[str, Any]


@dataclass
class RnaSimulationResult:
    u: np.ndarray
    s: np.ndarray
    rho: np.ndarray
    rates: np.ndarray
    ton: np.ndarray
    toff: np.ndarray
    is_repressive: Optional[np.ndarray]
    communication: CommunicationTruth


def _ensure_topovelo() -> None:
    global _topovelo_ready
    if _topovelo_ready:
        return
    importlib.import_module("torch")
    try:
        importlib.import_module("torch_geometric")
    except ImportError:
        pass
    _topovelo_ready = True


def pred_su_numpy(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_topovelo()
    import topovelo as tpv

    return tpv.model.model_util.pred_su_numpy(*args, **kwargs)


def rotate(coords: np.ndarray, angle: float) -> np.ndarray:
    rot_mtx = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.einsum("ij, kj->ki", rot_mtx, coords)


def rotate_cov(cov: np.ndarray, angle: float) -> np.ndarray:
    rot_mtx = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(rot_mtx, np.matmul(cov, rot_mtx.T))


def draw_poisson_event(
    n: int,
    lamb: float,
    t_max: float,
    geom_param: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)
    count = 1
    t = [0.0]
    n_par = 1
    while count < n:
        n_des = sp.stats.geom.rvs(geom_param, size=n_par)
        n_par_next = int(np.sum(n_des))
        if count + n_par_next > n:
            n_par = n - count
            n_par_next = n - count
            n_des = np.array([1] * n_par_next)
        uniform_samples = np.random.uniform(size=(n_par_next,))
        delta_t = -1 / lamb * np.log(1 - uniform_samples)
        start = 0
        for i in range(n_par):
            t.extend(list(t[-i - 1] + delta_t[start : start + n_des[i]]))
            start += n_des[i]
        count += n_par_next
        n_par = n_par_next
    t_arr = np.array(t)
    t_arr = t_arr / t_arr.max() * t_max
    assert len(t_arr) == n
    return t_arr


def draw_init_pos(
    n: int, style: str = "linear", d: float = 1.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    if style == "linear":
        return (
            np.random.normal(scale=d * 0.01, size=(n,)),
            np.random.uniform(-d / 2, d / 2, size=(n,)),
        )
    if style == "disk":
        theta = np.random.uniform(0, 2 * np.pi, size=(n))
        r = np.random.uniform(0, d / 2, size=(n))
        return r, theta
    raise ValueError(f"Unknown init style: {style}")


def const_v(t_max: float, d: float) -> float:
    return d / t_max


def linear_v(t: np.ndarray, k: float, v0: float) -> np.ndarray:
    return k * t + v0


def disp_const_v(t: np.ndarray, v: float) -> np.ndarray:
    return v * t


def disp_binary_v(t: np.ndarray, v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    delta_disp = np.diff(np.concatenate((np.array([0.0]), t))) * v * direction
    return np.cumsum(delta_disp)


def disp_linear_v(t: np.ndarray, k: float, v0: float) -> np.ndarray:
    return 0.5 * k * t**2 + v0 * t


def cart2polar(coords: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(coords, axis=1)
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    theta[np.isnan(theta)] = 0
    return np.stack([r, theta], 1)


def polar2cart(coords: np.ndarray) -> np.ndarray:
    return np.stack(
        [coords[:, 0] * np.cos(coords[:, 1]), coords[:, 0] * np.sin(coords[:, 1])],
        1,
    )


def spatial_dynamics_1d(
    n_cell: int,
    n_init: int,
    angle: float,
    lamb: float = 10,
    width: float = 1.0,
    height: float = 1.0,
    t_max: float = 20,
    v0: float = 0.01,
    velocity_model: str = "const",
    geom_param: float = 0.8,
    beta_param: Optional[Dict[str, float]] = None,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if beta_param is None:
        beta_param = {"a": 5.0, "b": 5.0}
    np.random.seed(seed)
    x0, y0 = draw_init_pos(n_init, "linear", width, seed)
    batch_size = n_cell // n_init
    n_rem = n_cell - batch_size * n_init
    x, y = np.empty((n_cell)), np.empty((n_cell))
    vx, vy = np.empty((n_cell)), np.empty((n_cell))
    t = np.empty((n_cell))

    start = 0
    for i in range(n_init):
        n = batch_size + (i < n_rem)
        t_batch = draw_poisson_event(
            n, lamb, t_max, geom_param, np.random.randint(0, 4294967295)
        )
        t[start : start + n] = t_batch
        if velocity_model == "linear":
            k = (height - v0 * t_max) / (0.5 * t_max**2)
            vx_batch = linear_v(t_batch, k, v0)
            x[start : start + n] = disp_linear_v(t_batch, k, v0)
        else:
            vx_batch = const_v(t_max, height)
            x[start : start + n] = x0[i] + disp_const_v(t_batch, vx_batch)
        max_vy = const_v(t_max, width * 0.2)
        min_vy = const_v(t_max, width * 0.02)
        vy_batch = np.random.uniform(min_vy, max_vy, size=n)
        p = sp.stats.beta.rvs(beta_param["a"], beta_param["b"])
        bern_samples = (np.random.uniform(size=(n,)) > p).astype(int)
        b = (bern_samples == 1) + (bern_samples == 0) * (-1)

        y[start : start + n] = y0[i] + disp_binary_v(t_batch, vy_batch, b)
        vx[start : start + n] = vx_batch
        vy[start : start + n] = vy_batch * b
        start += n

    coords = rotate(np.stack([x, y], 1), angle)
    v = rotate(np.array(np.stack([vx, vy], 1)), angle)
    return t, coords, v


def spatial_dynamics_1d_bi(
    n_cell: int,
    n_init: int,
    angle: float,
    lamb: float = 10,
    width: float = 1.0,
    height: Tuple[float, float] = (1.0, 1.0),
    t_max: float = 20,
    v0: Tuple[float, float] = (0.01, 0.1),
    velocity_model: str = "const",
    geom_param: float = 0.8,
    beta_param: Optional[Dict[str, float]] = None,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if beta_param is None:
        beta_param = {"a": 5.0, "b": 5.0}
    np.random.seed(seed)
    seed_1 = np.random.randint(0, 4294967295)
    t_1, coords_1, v_1 = spatial_dynamics_1d(
        n_cell // 2,
        n_init // 2,
        angle,
        lamb,
        width,
        height[0],
        t_max,
        v0[0],
        velocity_model,
        geom_param,
        beta_param,
        seed_1,
        **kwargs,
    )
    seed_2 = np.random.randint(0, 4294967295)
    t_2, coords_2, v_2 = spatial_dynamics_1d(
        n_cell - n_cell // 2,
        n_init - n_init // 2,
        angle + np.pi,
        lamb,
        width,
        height[1],
        t_max,
        v0[1],
        velocity_model,
        geom_param,
        beta_param,
        seed_2,
        **kwargs,
    )
    return (
        np.concatenate((t_1, t_2)),
        np.concatenate((coords_1, coords_2)),
        np.concatenate((v_1, v_2)),
    )


def spatial_dynamics_radial(
    n_cell: int,
    n_init: int,
    lamb: float = 10,
    r_max: float = 1.0,
    r_init: float = 0.01,
    t_max: float = 20,
    vr_0: float = 0.01,
    velocity_model: str = "const",
    geom_param: float = 0.8,
    beta_param: Optional[Dict[str, float]] = None,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if beta_param is None:
        beta_param = {"a": 5.0, "b": 5.0}
    np.random.seed(seed)
    r0, theta0 = draw_init_pos(n_init, "disk", r_init, seed)
    batch_size = n_cell // n_init
    n_rem = n_cell - batch_size * n_init
    r, theta = np.empty((n_cell)), np.empty((n_cell))
    vr, vtheta = np.empty((n_cell)), np.empty((n_cell))
    t = np.empty((n_cell))

    start = 0
    for i in range(n_init):
        n = batch_size + (i < n_rem)
        t_batch = draw_poisson_event(
            n, lamb, t_max, geom_param, np.random.randint(0, 4294967295)
        )
        t[start : start + n] = t_batch
        if velocity_model == "linear":
            k = (r_max - vr_0 * t_max) / (0.5 * t_max**2)
            vr_batch = linear_v(t_batch, k, vr_0)
            r[start : start + n] = disp_linear_v(t_batch, k, vr_0)
        else:
            vr_batch = const_v(t_max, r_max)
            r[start : start + n] = r0[i] + disp_const_v(t_batch, vr_batch)
        max_vtheta = const_v(t_max, np.pi / 2)
        min_vtheta = const_v(t_max, np.pi / 12)
        vtheta_batch = np.random.uniform(min_vtheta, max_vtheta, size=n)
        p = sp.stats.beta.rvs(beta_param["a"], beta_param["b"])
        bern_samples = (np.random.uniform(size=(n,)) > p).astype(int)
        b = (bern_samples == 0) + (bern_samples == 1) * (-1)
        theta[start : start + n] = theta0[i] + disp_binary_v(t_batch, vtheta_batch, b)
        vr[start : start + n] = vr_batch
        vtheta[start : start + n] = vtheta_batch * b
        start += n

    coords = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    v = np.stack(
        [
            vr * np.cos(theta) - vtheta * r * np.sin(theta),
            vr * np.sin(theta) + vtheta * r * np.cos(theta),
        ],
        1,
    )
    return t, coords, v


def spatial_dynamics_multiradial(
    n_cell: int,
    n_init: int,
    n_part: int,
    centers: np.ndarray,
    lamb: float = 10,
    r_max: float = 1.0,
    r_init: float = 0.01,
    t_max: float = 20,
    vr_0: float = 0.01,
    velocity_model: str = "const",
    geom_param: float = 0.8,
    beta_param: Optional[Dict[str, float]] = None,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[list, list, list]:
    if beta_param is None:
        beta_param = {"a": 5.0, "b": 5.0}
    np.random.seed(seed)
    m = n_cell % n_part
    m0 = n_init % n_part
    t_out, coords_out, v_out = [], [], []
    for i in range(n_part):
        part_seed = np.random.randint(0, 4294967295)
        t_, coords_, v_ = spatial_dynamics_radial(
            n_cell // n_part + (i < m),
            n_init // n_part + (i < m0),
            lamb,
            r_max,
            r_init,
            t_max,
            vr_0,
            velocity_model,
            geom_param,
            beta_param,
            seed=part_seed,
            **kwargs,
        )
        coords_ = coords_ + centers[i]
        t_out.append(t_)
        coords_out.append(coords_)
        v_out.append(v_)
    return t_out, coords_out, v_out


def rho_spatial_gaussian(
    coords: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    mn = multivariate_normal(mu, cov)
    return mn.pdf(coords) / mn.pdf(mu)


def rho_spatial_bimodal(
    coords: np.ndarray,
    p: float,
    mu_1: np.ndarray,
    mu_2: np.ndarray,
    cov_1: np.ndarray,
    cov_2: np.ndarray,
) -> np.ndarray:
    mn1 = multivariate_normal(mu_1, cov_1)
    mn2 = multivariate_normal(mu_2, cov_2)
    pmax = max(
        p * mn1.pdf(mu_1) + (1 - p) * mn2.pdf(mu_1),
        p * mn1.pdf(mu_2) + (1 - p) * mn2.pdf(mu_2),
    )
    return (p * mn1.pdf(coords) + (1 - p) * mn2.pdf(coords)) / pmax


def sample_rates(
    mu: Sequence[float], cov: np.ndarray, n_sample: int, seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    return np.exp(np.random.multivariate_normal(mu, cov, size=n_sample))


def simulate_rna(
    t: np.ndarray,
    coords: np.ndarray,
    n_gene: int,
    n_repress_gene: int,
    mu_param: Sequence[float],
    cov_param: np.ndarray,
    t_max: float,
    sparsity: Tuple[float, float],
    angle: float,
    noise_level: float = 0.1,
    noise_type: str = "Gaussian",
    n_neighbors: int = 16,
    ton_cutoff: Sequence[float] = (0.0, 0.2),
    toff_cutoff: Sequence[float] = (0.3, 1.0),
    rho_spatial_pattern: str = "Gaussian",
    centers: Optional[np.ndarray] = None,
    seed: int = 42,
    intrinsic_spatial_velocity: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    ground_truth_config: SimulationConfig = SimulationConfig(),
) -> RnaSimulationResult:
    np.random.seed(seed)

    rates = sample_rates(mu_param, cov_param, n_gene, seed)
    ton = np.random.uniform(t_max * ton_cutoff[0], t_max * ton_cutoff[1], size=(n_gene))
    toff = np.random.uniform(
        t_max * toff_cutoff[0], t_max * toff_cutoff[1], size=(n_gene)
    )

    if rho_spatial_pattern == "Polar Gaussian":
        if centers is None:
            coords_polar = cart2polar(coords)
            peak_pos_polar = np.stack(
                [
                    np.random.uniform(0, coords_polar[:, 0].max(), size=(n_gene)),
                    np.random.uniform(
                        coords_polar[:, 1].min(), coords_polar[:, 1].max(), size=(n_gene)
                    ),
                ],
                1,
            )
            sigma_r = np.std(coords_polar[:, 0]) * np.exp(
                np.random.normal(1, 0.5, size=(n_gene))
            )
            sigma_theta = np.exp(np.random.normal(1.0, 1.0, size=(n_gene)))
            delta_theta = np.stack(
                [
                    np.arctan2(
                        np.sin(coords_polar[:, 1] - peak_pos_polar[j][1]),
                        np.cos(coords_polar[:, 1] - peak_pos_polar[j][1]),
                    )
                    for j in range(n_gene)
                ],
                1,
            )
            rho = np.stack(
                [
                    rho_spatial_gaussian(
                        np.stack(
                            [coords_polar[:, 0], 1 - np.cos(delta_theta[:, j])], 1
                        ),
                        np.array([peak_pos_polar[j][0], 0]),
                        np.diag([sigma_r[j] ** 2, sigma_theta[j] ** 2]),
                    )
                    for j in range(n_gene)
                ],
                1,
            )
        else:
            n_part = len(centers)
            angle_to_next = 2 * np.pi / n_part
            coords_polar = cart2polar(coords[0] - centers[0])
            peak_pos_polar = np.stack(
                [
                    np.random.uniform(0, coords_polar[:, 0].max(), size=(n_gene)),
                    np.random.uniform(
                        coords_polar[:, 1].min(), coords_polar[:, 1].max(), size=(n_gene)
                    ),
                ],
                1,
            )
            sigma_r = np.std(coords_polar[:, 0]) * np.exp(
                np.random.normal(1, 0.5, size=(n_gene))
            )
            sigma_theta = np.exp(np.random.normal(1.0, 1.0, size=(n_gene)))
            peak_pos = polar2cart(peak_pos_polar) + centers[0]
            rho_parts = []
            for part_i in range(n_part):
                peak_pos_polar_ = cart2polar(
                    rotate(peak_pos, part_i * angle_to_next) - centers[part_i]
                )
                coords_polar_ = cart2polar(coords[part_i] - centers[part_i])
                delta_theta = np.stack(
                    [
                        np.arctan2(
                            np.sin(coords_polar_[:, 1] - peak_pos_polar_[j][1]),
                            np.cos(coords_polar_[:, 1] - peak_pos_polar_[j][1]),
                        )
                        for j in range(n_gene)
                    ],
                    1,
                )
                rho_ = np.stack(
                    [
                        rho_spatial_gaussian(
                            np.stack(
                                [coords_polar_[:, 0], 1 - np.cos(delta_theta[:, j])], 1
                            ),
                            np.array([peak_pos_polar_[j][0], 0]),
                            np.diag([sigma_r[j] ** 2, sigma_theta[j] ** 2]),
                        )
                        for j in range(n_gene)
                    ],
                    1,
                )
                rho_parts.append(rho_)
            t = np.concatenate(t)
            coords = np.concatenate(coords, 0)
            rho = np.concatenate(rho_parts, 0)
    elif rho_spatial_pattern == "Gaussian (Bidirection)":
        peak_pos = rotate(
            np.stack(
                [
                    np.random.uniform(coords[:, 0].min(), coords[:, 0].max(), size=(n_gene)),
                    np.random.uniform(coords[:, 1].min(), coords[:, 1].max(), size=(n_gene)),
                ],
                1,
            ),
            angle,
        )
        sigma_x = np.std(coords[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        sigma_y = np.std(coords[:, 1]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        rho = np.stack(
            [
                rho_spatial_bimodal(
                    coords,
                    0.5,
                    peak_pos[j],
                    -peak_pos[j],
                    rotate_cov(np.diag([sigma_x[j] ** 2, sigma_y[j] ** 2]), angle),
                    rotate_cov(np.diag([sigma_x[j] ** 2, sigma_y[j] ** 2]), angle),
                )
                for j in range(n_gene)
            ],
            1,
        )
    else:
        peak_pos = rotate(
            np.stack(
                [
                    np.random.uniform(coords[:, 0].min(), coords[:, 0].max(), size=(n_gene)),
                    np.random.uniform(coords[:, 1].min(), coords[:, 1].max(), size=(n_gene)),
                ],
                1,
            ),
            angle,
        )
        sigma_x = np.std(coords[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        sigma_y = np.std(coords[:, 1]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        rho = np.stack(
            [
                rho_spatial_gaussian(
                    coords,
                    peak_pos[j],
                    rotate_cov(np.diag([sigma_x[j] ** 2, sigma_y[j] ** 2]), angle),
                )
                for j in range(n_gene)
            ],
            1,
        )

    n_cell = len(t)
    transcription_drive = (rates[:, 0].reshape(1, -1) * rho).astype(np.float32)

    if intrinsic_spatial_velocity is None:
        spatial_velocity = np.zeros((n_cell, coords.shape[1]), dtype=np.float32)
    elif isinstance(intrinsic_spatial_velocity, (list, tuple)):
        spatial_velocity = np.concatenate(intrinsic_spatial_velocity, axis=0).astype(
            np.float32,
            copy=False,
        )
    else:
        spatial_velocity = np.asarray(intrinsic_spatial_velocity, dtype=np.float32)

    communication = _build_communication_truth(
        np.asarray(t, dtype=np.float32),
        np.asarray(coords, dtype=np.float32),
        spatial_velocity,
        transcription_drive,
        ground_truth_config,
        seed,
    )
    lr_indices = np.asarray(communication.lr_gene_indices, dtype=np.int64)
    transcription_drive = np.clip(
        communication.lr_programmed_drive + communication.expression_effect,
        0.0,
        None,
    ).astype(np.float32)
    rho = (
        transcription_drive / np.maximum(rates[:, 0].reshape(1, -1), 1e-8)
    ).astype(np.float32)
    beta, gamma, ton, toff = _ccc_stabilize_lr_kinetics(
        rates[:, 1],
        rates[:, 2],
        ton,
        toff,
        lr_indices,
        replace(ground_truth_config, t_max=float(t_max)),
    )
    rates = rates.astype(np.float64, copy=True)
    rates[:, 1] = beta
    rates[:, 2] = gamma

    t = t.reshape(-1, 1)

    if n_repress_gene > 0:
        candidate_repress = np.setdiff1d(
            np.arange(n_gene, dtype=int),
            lr_indices,
            assume_unique=False,
        )
        if candidate_repress.size < n_repress_gene:
            candidate_repress = np.arange(n_gene, dtype=int)
        perm = np.random.permutation(candidate_repress)
        repress_idx = np.sort(perm[: min(n_repress_gene, perm.size)])
        ind_idx = np.setdiff1d(np.arange(n_gene, dtype=int), repress_idx, assume_unique=False)
        is_repressive = np.zeros((n_gene)).astype(bool)
        is_repressive[repress_idx] = 1

        u_ind_1, s_ind_1 = pred_su_numpy(
            np.clip(t - ton[ind_idx], 0, None),
            0,
            0,
            transcription_drive[:, ind_idx],
            rates[ind_idx, 1],
            rates[ind_idx, 2],
        )
        u0_, s0_ = pred_su_numpy(
            toff[ind_idx] - ton[ind_idx],
            0,
            0,
            rates[ind_idx, 0] * rho[:, ind_idx],
            rates[ind_idx, 1],
            rates[ind_idx, 2],
        )
        u_ind_2, s_ind_2 = pred_su_numpy(
            t - toff[ind_idx],
            u0_,
            s0_,
            0,
            rates[ind_idx, 1],
            rates[ind_idx, 2],
        )
        u, s = np.empty((n_cell, n_gene)), np.empty((n_cell, n_gene))
        u_repress, s_repress = pred_su_numpy(
            t,
            transcription_drive[:, repress_idx] / rates[repress_idx, 1],
            transcription_drive[:, repress_idx] / rates[repress_idx, 2],
            0,
            rates[repress_idx, 1],
            rates[repress_idx, 2],
        )
        u[:, repress_idx] = u_repress
        s[:, repress_idx] = s_repress
        mask = t < toff[ind_idx]
        u[:, ind_idx] = u_ind_1 * mask + u_ind_2 * (1 - mask)
        s[:, ind_idx] = s_ind_1 * mask + s_ind_2 * (1 - mask)
    else:
        u_ind_1, s_ind_1 = pred_su_numpy(
            np.clip(t - ton, 0, None),
            0,
            0,
            transcription_drive,
            rates[:, 1],
            rates[:, 2],
        )
        u0_, s0_ = pred_su_numpy(
            toff - ton,
            0,
            0,
            rates[:, 0] * rho,
            rates[:, 1],
            rates[:, 2],
        )
        u_ind_2, s_ind_2 = pred_su_numpy(
            t - toff,
            u0_,
            s0_,
            0,
            rates[:, 1],
            rates[:, 2],
        )
        mask = t < toff
        u = u_ind_1 * mask + u_ind_2 * (1 - mask)
        s = s_ind_1 * mask + s_ind_2 * (1 - mask)
        is_repressive = None

    if noise_type == "Spatial Gaussian":
        eps_u = np.random.normal(0.0, noise_level, size=u.shape)
        eps_s = np.random.normal(0.0, noise_level, size=u.shape)
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        dist, ind = knn.fit(coords).kneighbors()
        sigma = np.median(dist)
        coeff = np.exp(-(dist / sigma) ** 2)
        coeff = coeff / coeff.sum(1).reshape(-1, 1)
        noise_u = np.stack(
            [np.sum(coeff[i].reshape(-1, 1) * eps_u[ind[i]], 0) for i in range(n_cell)]
        )
        noise_s = np.stack(
            [np.sum(coeff[i].reshape(-1, 1) * eps_s[ind[i]], 0) for i in range(n_cell)]
        )
    elif noise_type == "Gaussian":
        noise_u = np.random.normal(0.0, np.clip(noise_level * u, 0.01, None), size=u.shape)
        noise_s = np.random.normal(0.0, np.clip(noise_level * s, 0.01, None), size=s.shape)
    else:
        noise_u = np.zeros_like(u)
        noise_s = np.zeros_like(s)

    dropout_rng = np.random.default_rng(seed + 505)
    u = _ccc_expression_dependent_dropout(
        np.clip(u + noise_u, 0, None),
        float(sparsity[0]),
        ground_truth_config.dropout_expression_protection,
        dropout_rng,
        protected_gene_indices=lr_indices,
        protected_dropout_scale=ground_truth_config.lr_dropout_scale,
    )
    s = _ccc_expression_dependent_dropout(
        np.clip(s + noise_s, 0, None),
        float(sparsity[1]),
        ground_truth_config.dropout_expression_protection,
        dropout_rng,
        protected_gene_indices=lr_indices,
        protected_dropout_scale=ground_truth_config.lr_dropout_scale,
    )
    u, s = _ccc_anchor_lr_observation_to_drive(
        u,
        s,
        transcription_drive,
        lr_indices,
        ground_truth_config.lr_observation_anchor_strength,
    )
    return RnaSimulationResult(u, s, rho, rates, ton, toff, is_repressive, communication)


def _build_adata(
    t: np.ndarray,
    coords: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    rho: np.ndarray,
    rates: np.ndarray,
    ton: np.ndarray,
    toff: np.ndarray,
    is_repressive: Optional[np.ndarray],
    communication: CommunicationTruth,
    extra_obs: Optional[Dict[str, np.ndarray]] = None,
) -> anndata.AnnData:
    obs: Dict[str, np.ndarray] = {"true_time": t, **communication.obs}
    if extra_obs:
        obs.update(extra_obs)

    var: Dict[str, np.ndarray] = {
        "true_alpha": rates[:, 0],
        "true_alpha_baseline": np.mean(communication.baseline_drive, axis=0).astype(
            np.float32
        ),
        "true_alpha_lr_programmed": np.mean(
            communication.lr_programmed_drive,
            axis=0,
        ).astype(np.float32),
        "true_beta": rates[:, 1],
        "true_gamma": rates[:, 2],
        "true_ton": ton,
        "true_toff": toff,
    }
    if is_repressive is not None:
        var["repressive_genes"] = is_repressive

    layers = {
        "unspliced": sp.sparse.csr_matrix(u),
        "spliced": sp.sparse.csr_matrix(s),
        "true_rho": rho,
        "baseline_transcription_drive": sp.sparse.csr_matrix(
            communication.baseline_drive
        ),
        "lr_programmed_transcription_drive": sp.sparse.csr_matrix(
            communication.lr_programmed_drive
        ),
        "communication_expression_effect": sp.sparse.csr_matrix(
            communication.expression_effect
        ),
    }

    adata = anndata.AnnData(s, obs, var, layers=layers)
    adata.var_names = np.asarray([f"G{idx:04d}" for idx in range(s.shape[1])])
    adata.obs_names = np.asarray([str(idx) for idx in range(s.shape[0])])
    adata.obsm["X_spatial"] = coords
    for key, value in communication.obsm.items():
        adata.obsm[key] = value
    adata.uns["ground_truth"] = communication.uns
    return adata


def _build_communication_truth(
    t: np.ndarray,
    coords: np.ndarray,
    intrinsic_spatial_velocity: np.ndarray,
    transcription_drive: np.ndarray,
    config: SimulationConfig,
    seed: int = 42,
) -> CommunicationTruth:
    mode = str(config.event_attribution_mode)
    if mode not in EVENT_ATTRIBUTION_MODES:
        raise ValueError(
            f"Unknown event_attribution_mode {mode!r}; expected one of {EVENT_ATTRIBUTION_MODES}"
        )

    coords = np.asarray(coords, dtype=np.float32)
    intrinsic_spatial_velocity = np.asarray(intrinsic_spatial_velocity, dtype=np.float32)
    baseline_drive = np.asarray(transcription_drive, dtype=np.float32)
    latent = _normalize_vector(t)
    states = _state_labels(latent)
    regions = _assign_regions(coords)
    ccc_config = replace(
        config,
        n_cells=int(baseline_drive.shape[0]),
        n_genes=int(baseline_drive.shape[1]),
        t_max=float(np.nanmax(t)) if np.size(t) else config.t_max,
        seed=int(seed),
    )

    module_rng = np.random.default_rng(seed + 303)
    event_influence_rng = np.random.default_rng(seed + 404)
    modules, lr_pairs = _ccc_build_modules(
        baseline_drive,
        latent,
        states,
        regions,
        coords,
        ccc_config,
        module_rng,
    )
    lr_programmed_drive = _ccc_apply_lr_expression_program(
        mode,
        baseline_drive,
        modules,
        latent,
        states,
        regions,
        coords,
        ccc_config,
    )
    obs, obsm, expression_effect, ground_truth = _ccc_build_communication_truth(
        mode,
        np.asarray(t, dtype=np.float32),
        latent,
        coords,
        intrinsic_spatial_velocity,
        states,
        regions,
        baseline_drive,
        lr_programmed_drive,
        modules,
        lr_pairs,
        ccc_config,
        event_influence_rng,
    )
    if config.intrinsic_velocity_weight != 1.0:
        true_velocity = (
            float(config.intrinsic_velocity_weight) * obsm["intrinsic_spatial_velocity"]
            + obsm["communication_spatial_velocity"]
        ).astype(np.float32)
        obsm["true_spatial_velocity"] = true_velocity
        obsm["velocity_spatial"] = true_velocity.copy()
        ground_truth["intrinsic_velocity_weight"] = float(config.intrinsic_velocity_weight)
    ground_truth["event_attribution_mode"] = mode
    return CommunicationTruth(
        obs=obs,
        obsm=obsm,
        expression_effect=expression_effect,
        baseline_drive=baseline_drive,
        lr_programmed_drive=lr_programmed_drive,
        lr_gene_indices=_ccc_lr_gene_indices(modules),
        uns=ground_truth,
    )


def simulation(
    n_cell: int,
    n_gene: int,
    n_init_cell: int,
    n_repress_gene: int,
    sparsity: Tuple[float, float],
    angle: float,
    d: float = 1.0,
    t_max: float = 20,
    growth_type: str = "1d",
    velocity_model: str = "const",
    v0: Union[float, Tuple[float, float]] = 0.01,
    geom_param: float = 0.5,
    beta_param: Optional[Dict[str, float]] = None,
    mu_param: Sequence[float] = (1.0, 0.0, 0.0),
    cov_param: Optional[np.ndarray] = None,
    noise_level: float = 0.5,
    noise_type: str = "Gaussian",
    n_neighbors: int = 16,
    ton_cutoff: Sequence[float] = (0.0, 0.1),
    toff_cutoff: Sequence[float] = (0.3, 1.0),
    rho_spatial_pattern: str = "Gaussian",
    n_part: int = 3,
    centers: Optional[np.ndarray] = None,
    seed: int = 42,
    ground_truth_config: SimulationConfig = SimulationConfig(),
) -> anndata.AnnData:
    if beta_param is None:
        beta_param = {"a": 5.0, "b": 5.0}
    if cov_param is None:
        cov_param = np.array(
            [[1.0, 0.05, 0.05], [0.05, 1.0, 0.05], [0.05, 0.05, 1.0]]
        )

    if growth_type == "1d":
        t, coords, v = spatial_dynamics_1d(
            n_cell,
            n_init_cell,
            angle,
            width=d,
            height=d,
            t_max=t_max,
            v0=float(v0),
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed,
        )
    elif growth_type == "1d (bidirection)":
        v0_pair = v0 if isinstance(v0, tuple) else (float(v0), float(v0))
        t, coords, v = spatial_dynamics_1d_bi(
            n_cell,
            n_init_cell,
            angle,
            width=d,
            height=(d, d),
            t_max=t_max,
            v0=v0_pair,
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed,
        )
    elif growth_type == "radial":
        t, coords, v = spatial_dynamics_radial(
            n_cell,
            n_init_cell,
            r_max=d,
            r_init=d * 0.01,
            t_max=20,
            vr_0=float(v0),
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed,
        )
    elif growth_type == "multi-radial":
        t, coords, v = spatial_dynamics_multiradial(
            n_cell,
            n_init_cell,
            n_part,
            centers,
            r_max=d,
            r_init=d * 0.01,
            t_max=t_max,
            vr_0=float(v0),
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown growth_type: {growth_type}")

    rna = simulate_rna(
        t,
        coords,
        n_gene,
        n_repress_gene,
        mu_param,
        cov_param,
        t_max,
        sparsity,
        angle,
        noise_level=noise_level,
        noise_type=noise_type,
        rho_spatial_pattern=rho_spatial_pattern,
        ton_cutoff=ton_cutoff,
        toff_cutoff=toff_cutoff,
        centers=centers,
        intrinsic_spatial_velocity=v,
        ground_truth_config=ground_truth_config,
    )

    if growth_type == "multi-radial":
        t = np.concatenate(t)
        coords = np.concatenate(coords)

    return _build_adata(
        t,
        coords,
        rna.u,
        rna.s,
        rna.rho,
        rna.rates,
        rna.ton,
        rna.toff,
        rna.is_repressive,
        rna.communication,
    )


logger = logging.getLogger(__name__)

SCENARIO_NAMES = (
    "linear",
    "bilinear",
    "radial",
    "multiradial",
)

class SimPipe:
    """Generate synthetic AnnData for linear, bilinear, radial, or multiradial directions."""

    def __init__(
        self,
        sim_dir: str | Path,
        *,
        seed: int = 42,
        ground_truth_config: SimulationConfig | None = None,
    ) -> None:
        self.sim_dir = Path(sim_dir)
        self.seed = seed
        self.ground_truth_config = ground_truth_config or SimulationConfig()
        self._angle = np.pi / 4
        self._n_gene = 500
        self._n_repress_gene = 100
        self._sparsity = (
            self.ground_truth_config.unspliced_dropout,
            self.ground_truth_config.spliced_dropout,
        )

    def run(self, scenarios: Optional[Iterable[str]] = None) -> List[Path]:
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        names = list(scenarios) if scenarios is not None else list(SCENARIO_NAMES)
        outputs: List[Path] = []
        for name in names:
            if name not in SCENARIO_NAMES:
                raise ValueError(
                    f"Unknown scenario {name!r}; expected one of {SCENARIO_NAMES}"
                )
            logger.info("Generating developmental-direction simulation: %s", name)
            adata = self._generate(name)
            path = save_h5ad(adata, self.sim_dir / f"{name}.h5ad")
            outputs.append(path)
            logger.info("Wrote %s", path)
        return outputs

    def _base_kwargs(self, *, n_cell: int, n_init_cell: int) -> dict:
        return {
            "n_cell": n_cell,
            "n_gene": self._n_gene,
            "n_init_cell": n_init_cell,
            "n_repress_gene": self._n_repress_gene,
            "sparsity": self._sparsity,
            "angle": self._angle,
            "noise_level": 0.1,
            "velocity_model": "linear",
            "seed": self.seed,
            "ground_truth_config": self.ground_truth_config,
        }

    def _generate(self, name: str) -> ad.AnnData:
        if name == "linear":
            kwargs = self._base_kwargs(n_cell=3000, n_init_cell=60)
            kwargs.update(growth_type="1d", rho_spatial_pattern="Gaussian")
            return simulation(**kwargs)
        if name == "bilinear":
            kwargs = self._base_kwargs(n_cell=3000, n_init_cell=60)
            kwargs.update(
                growth_type="1d (bidirection)",
                v0=(0.01, 0.01),
                rho_spatial_pattern="Gaussian (Bidirection)",
            )
            return simulation(**kwargs)
        if name == "radial":
            kwargs = self._base_kwargs(n_cell=3000, n_init_cell=100)
            kwargs.update(growth_type="radial", rho_spatial_pattern="Polar Gaussian")
            return simulation(**kwargs)
        if name == "multiradial":
            diagonal = np.sqrt(2) / 2
            centers = diagonal * np.array(
                [[1, 1], [-1, 1], [-1, -1], [1, -1]],
                dtype=float,
            )
            kwargs = self._base_kwargs(n_cell=12000, n_init_cell=400)
            kwargs.update(
                growth_type="multi-radial",
                rho_spatial_pattern="Polar Gaussian",
                n_part=4,
                centers=centers,
            )
            return simulation(**kwargs)

        raise ValueError(
            f"Unknown scenario {name!r}; expected one of {SCENARIO_NAMES}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SimPipe(sim_dir=ROOT / "paper" / "data" / "direction" / "raw").run()
