"""Three training objectives for CellVOCAL: CCC, RNA velocity, and latent time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from model.train.group_ops import group_index, group_mean, group_sum


@dataclass
class LossWeights:
    """Weights for the three independent biological constraints."""

    velocity: float = 1.0
    time: float = 0.3
    ccc: float = 0.3


def _gene_scale(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return x.detach().float().std(dim=0, unbiased=False).clamp_min(eps)


def _local_pairs(
    edge_index: Optional[torch.Tensor],
    *,
    n_spots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index is None or edge_index.numel() == 0:
        device = edge_index.device if edge_index is not None else torch.device("cpu")
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    src, dst = edge_index[0].long(), edge_index[1].long()
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return src, dst
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pair_key = torch.unique(lo * int(n_spots) + hi, sorted=False)
    return pair_key // int(n_spots), pair_key % int(n_spots)


def _local_linear_slope(
    state: torch.Tensor,
    early: torch.Tensor,
    late: torch.Tensor,
    dt: torch.Tensor,
    inverse_early: torch.Tensor,
    n_early: int,
) -> torch.Tensor:
    """Least-squares local derivative from all sampled forward neighbours."""
    delta = state[late] - state[early]
    numerator = state.new_zeros((n_early, state.shape[1]))
    numerator.index_add_(0, inverse_early, dt.unsqueeze(-1) * delta)
    denominator = state.new_zeros(n_early)
    denominator.index_add_(0, inverse_early, dt.square())
    return numerator / denominator.clamp_min(1e-6).unsqueeze(-1)


def _velocity_channel_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative vector fit plus an explicit forward-direction penalty."""
    eps = 1e-6
    target_norm = target.detach().norm(dim=-1)
    keep = target_norm > eps
    if not keep.any():
        zero = predicted.new_zeros(())
        return zero, zero

    predicted = predicted[keep]
    target = target[keep]
    scale = target_norm[keep].unsqueeze(-1)
    fit = F.smooth_l1_loss(predicted / scale, target / scale)
    predicted_norm = predicted.norm(dim=-1)
    target_kept_norm = target.norm(dim=-1)
    cosine = (predicted * target).sum(dim=-1) / (
        predicted_norm.clamp_min(0.1) * target_kept_norm.clamp_min(eps)
    )
    cosine = cosine.clamp(-1.0, 1.0)
    direction = (1.0 - cosine).mean()
    return fit + max(0.0, direction_weight) * direction, cosine.detach().mean()


def loss_rna_velocity(
    outputs: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match predicted RNA derivatives to observed local RNA changes.

    This single constraint supervises both ``du/dt`` and ``ds/dt``. Optional
    Mu/Ms-like target layers change only the observation space, not the loss.
    """
    u = outputs.get("velocity_target_unspliced")
    s = outputs.get("velocity_target_spliced")
    if u is None:
        u = outputs["unspliced"]
    if s is None:
        s = outputs["spliced"]

    velocity = outputs["velocity"]
    du = outputs["du"]
    prior = outputs.get("time_prior")
    if prior is None:
        raise ValueError("time_prior is required for the RNA-velocity objective")
    prior = prior.detach().flatten()
    order_signal = prior
    cell_a, cell_b = _local_pairs(
        outputs.get("edge_index"),
        n_spots=int(prior.shape[0]),
    )
    if cell_a.numel() == 0:
        zero = velocity.new_zeros(())
        return zero, zero

    a_first = order_signal[cell_a] <= order_signal[cell_b]
    early = torch.where(a_first, cell_a, cell_b)
    late = torch.where(a_first, cell_b, cell_a)
    dt = order_signal[late] - order_signal[early]
    keep = dt >= float(outputs.get("velocity_order_min_delta", 0.02))
    if not keep.any():
        zero = velocity.new_zeros(())
        return zero, zero
    early, late, dt = early[keep], late[keep], dt[keep]
    max_pairs = int(outputs.get("loss_event_sample_size", 2048))
    if max_pairs > 0 and early.numel() > max_pairs:
        sample = torch.randperm(early.numel(), device=early.device)[:max_pairs]
        early, late, dt = early[sample], late[sample], dt[sample]
    dt = dt.clamp_min(1e-3)

    u_scale = _gene_scale(u)
    s_scale = _gene_scale(s)
    early_cells, inverse_early = torch.unique(early, return_inverse=True)
    n_early = int(early_cells.numel())
    target_du = _local_linear_slope(
        u / u_scale,
        early,
        late,
        dt,
        inverse_early,
        n_early,
    )
    target_velocity = _local_linear_slope(
        s / s_scale,
        early,
        late,
        dt,
        inverse_early,
        n_early,
    )
    direction_weight = float(outputs.get("velocity_direction_weight", 1.0))
    du_loss, _ = _velocity_channel_loss(
        du[early_cells] / u_scale,
        target_du,
        direction_weight=direction_weight,
    )
    velocity_loss, velocity_cosine = _velocity_channel_loss(
        velocity[early_cells] / s_scale,
        target_velocity,
        direction_weight=direction_weight,
    )
    return 0.5 * (du_loss + velocity_loss), velocity_cosine


def loss_latent_time(
    outputs: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank latent time monotonically along one dataset-provided time prior."""
    time_score = outputs.get("latent_time_score", outputs["latent_time"])
    prior = outputs.get("time_prior")
    if prior is None:
        raise ValueError("time_prior is required for the latent-time objective")
    prior = prior.detach().flatten()
    n_spots = int(time_score.shape[0])
    if n_spots < 2:
        zero = time_score.new_zeros(())
        return zero, zero

    n_pairs = max(1, int(outputs.get("loss_event_sample_size", 2048)))
    n_draw = max(n_pairs, 4 * n_pairs)
    cell_a = torch.randint(n_spots, (n_draw,), device=time_score.device)
    cell_b = torch.randint(n_spots, (n_draw,), device=time_score.device)
    prior_delta = prior[cell_b] - prior[cell_a]
    keep = (cell_a != cell_b) & (
        prior_delta.abs() >= float(outputs.get("time_prior_min_delta", 0.25))
    )
    if not keep.any():
        zero = time_score.new_zeros(())
        return zero, zero
    cell_a = cell_a[keep][:n_pairs]
    cell_b = cell_b[keep][:n_pairs]
    prior_delta = prior_delta[keep][:n_pairs]

    signed_dtau = torch.sign(prior_delta) * (
        time_score[cell_b] - time_score[cell_a]
    )
    temperature = max(float(outputs.get("time_prior_temperature", 10.0)), 1e-3)
    rank_loss = (F.softplus(-temperature * signed_dtau) / temperature).mean()
    score_mean = time_score.mean()
    score_std = time_score.float().std(unbiased=False)
    score_regularization = score_mean.square() + (score_std - 1.0).square()
    regularization_weight = max(
        float(outputs.get("time_score_regularization_weight", 0.01)),
        0.0,
    )
    loss = rank_loss + regularization_weight * score_regularization
    accuracy = (signed_dtau.detach() > 0.0).float().mean()
    return loss, accuracy


def _ccc_prior(evidence: torch.Tensor) -> torch.Tensor:
    evidence = torch.log1p(torch.relu(evidence.float()))
    if evidence.numel() < 2:
        return torch.ones_like(evidence)
    lo = torch.quantile(evidence, 0.05)
    hi = torch.quantile(evidence, 0.95)
    if torch.isclose(lo, hi):
        return torch.ones_like(evidence)
    return ((evidence - lo) / (hi - lo)).clamp(0.0, 1.0)


def _distribution_kl(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    group_key: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, inv, n_groups = group_index(group_key.long())
    target_raw = target_raw.to(dtype=pred_raw.dtype).clamp_min(eps)
    pred_raw = pred_raw.clamp_min(eps)
    target = target_raw / group_sum(target_raw, inv, n_groups)[inv].clamp_min(eps)
    pred = pred_raw / group_sum(pred_raw, inv, n_groups)[inv].clamp_min(eps)
    target_detached = target.detach()
    per_item = target_detached * (
        target_detached.clamp_min(eps).log() - pred.clamp_min(eps).log()
    )
    return group_sum(per_item, inv, n_groups).mean(), pred, target_detached


def _receiver_module_key(
    event_receiver: torch.Tensor,
    event_module: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    module_base = int(event_module.detach().max().item()) + 1
    module_base = max(module_base, 1)
    return event_receiver.long() * module_base + event_module.long(), module_base


def loss_ccc(
    event_influence: torch.Tensor,
    event_receiver: torch.Tensor,
    lr_compatibility: torch.Tensor,
    candidate_score: Optional[torch.Tensor] = None,
    *,
    module_activation_prob: Optional[torch.Tensor] = None,
    sender_share: Optional[torch.Tensor] = None,
    event_module: Optional[torch.Tensor] = None,
    ligand_expr: Optional[torch.Tensor] = None,
    receptor_expr: Optional[torch.Tensor] = None,
    joint_prior_weight: float = 0.25,
    sender_expr_weight: float = 0.5,
    receptor_expr_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Anchor inferred signalling influence to expression-aware LR structure."""
    if event_influence.numel() == 0:
        zero = event_influence.new_zeros(())
        return zero, zero, {
            "joint": zero,
            "sender_expr": zero,
            "receptor_expr": zero,
        }

    eps = 1e-6
    evidence = lr_compatibility if candidate_score is None else candidate_score
    joint_loss, pred, target = _distribution_kl(
        event_influence,
        _ccc_prior(evidence).to(event_influence.dtype) + eps,
        event_receiver,
        eps=eps,
    )
    cosine = F.cosine_similarity(
        pred.detach().unsqueeze(0),
        target.detach().unsqueeze(0),
        dim=-1,
        eps=eps,
    ).squeeze(0)
    total = max(0.0, float(joint_prior_weight)) * joint_loss

    sender_loss = event_influence.new_zeros(())
    receptor_loss = event_influence.new_zeros(())
    if event_module is not None:
        rm_key, _ = _receiver_module_key(event_receiver, event_module)
        if sender_share is not None:
            sender_signal = ligand_expr if ligand_expr is not None else evidence
            sender_loss, _, _ = _distribution_kl(
                sender_share,
                _ccc_prior(sender_signal).to(event_influence.dtype) + eps,
                rm_key,
                eps=eps,
            )
            total = total + max(0.0, float(sender_expr_weight)) * sender_loss

        if module_activation_prob is not None:
            _, inv_rm, n_rm = group_index(rm_key)
            module_pred = group_mean(module_activation_prob, inv_rm, n_rm)
            module_signal_source = receptor_expr if receptor_expr is not None else evidence
            module_signal = group_mean(module_signal_source.float(), inv_rm, n_rm)
            receptor_target = _ccc_prior(module_signal).to(module_pred.dtype)
            receptor_loss = F.binary_cross_entropy(
                module_pred.clamp(eps, 1.0 - eps),
                receptor_target.detach().clamp(0.0, 1.0),
            )
            total = total + max(0.0, float(receptor_expr_weight)) * receptor_loss

    parts = {
        "joint": joint_loss.detach(),
        "sender_expr": sender_loss.detach(),
        "receptor_expr": receptor_loss.detach(),
    }
    return total, cosine, parts


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Aggregate exactly three top-level objectives."""
    velocity_loss, velocity_cosine = loss_rna_velocity(outputs)
    time_loss, time_accuracy = loss_latent_time(outputs)
    ccc_loss, ccc_prior_cosine, ccc_parts = loss_ccc(
        outputs["ccc_event_influence"],
        outputs["event_receiver"],
        outputs["lr_compatibility"],
        outputs.get("candidate_score"),
        module_activation_prob=outputs.get("ccc_module_activation_prob"),
        sender_share=outputs.get("ccc_sender_share"),
        event_module=outputs.get("event_module"),
        ligand_expr=outputs.get("ligand_expr"),
        receptor_expr=outputs.get("receptor_expr"),
        joint_prior_weight=float(outputs.get("ccc_joint_prior_weight", 0.25)),
        sender_expr_weight=float(outputs.get("ccc_sender_expr_weight", 0.5)),
        receptor_expr_weight=float(outputs.get("ccc_receptor_expr_weight", 0.5)),
    )
    total = (
        weights.velocity * velocity_loss
        + weights.time * time_loss
        + weights.ccc * ccc_loss
    )
    metrics = {
        "loss": float(total.item()),
        "loss_velocity": float(velocity_loss.item()),
        "loss_time": float(time_loss.item()),
        "loss_ccc": float(ccc_loss.item()),
        "velocity_cosine": float(velocity_cosine.item()),
        "time_order_accuracy": float(time_accuracy.item()),
        "ccc_prior_cosine": float(ccc_prior_cosine.item()),
        "loss_ccc_joint": float(ccc_parts["joint"].item()),
        "loss_ccc_sender_expr": float(ccc_parts["sender_expr"].item()),
        "loss_ccc_receptor_expr": float(ccc_parts["receptor_expr"].item()),
    }
    return total, metrics


MAIN_LOSS_KEYS = (
    "loss",
    "loss_velocity",
    "loss_time",
    "loss_ccc",
    "loss_ccc_joint",
    "loss_ccc_sender_expr",
    "loss_ccc_receptor_expr",
    "velocity_cosine",
    "time_order_accuracy",
    "ccc_prior_cosine",
)


def format_loss_metrics(
    metrics: dict[str, float],
    keys: tuple[str, ...] = MAIN_LOSS_KEYS,
) -> str:
    return ", ".join(f"{key}={metrics.get(key, 0.0):.4f}" for key in keys)
