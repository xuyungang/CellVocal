"""CellVOCAL: RNA velocity + latent time + velocity-consistent CCC network."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from model.train.config import CellVOCALConfig, CellVOCALModelConfig
from model.train.group_ops import (
    group_index,
    group_mean,
    group_sum,
)
from model.train.types import BatchTensors
from model.translator import CCCEventTranslator


def _softplus_pos(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return F.softplus(x) + eps


def _ccc_prior_signal(evidence: torch.Tensor) -> torch.Tensor:
    evidence = torch.log1p(torch.relu(evidence.float()))
    if evidence.numel() < 2:
        return torch.ones_like(evidence)
    lo = torch.quantile(evidence, 0.05)
    hi = torch.quantile(evidence, 0.95)
    if torch.isclose(lo, hi):
        return torch.ones_like(evidence)
    return ((evidence - lo) / (hi - lo)).clamp(0.0, 1.0)


class FourierSpatialEncoder(nn.Module):
    """Scale-adaptive Fourier features for spatial offsets."""

    def __init__(self, spatial_dim: int, n_bands: int = 4) -> None:
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.n_bands = max(0, int(n_bands))
        if self.n_bands > 0:
            freq = torch.pow(2.0, torch.arange(self.n_bands, dtype=torch.float32))
        else:
            freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("freq_bands", freq, persistent=False)
        self.out_dim = self.spatial_dim * (1 + 2 * self.n_bands)

    def forward(self, delta_spatial: torch.Tensor) -> torch.Tensor:
        if self.n_bands == 0 or delta_spatial.numel() == 0:
            return delta_spatial
        scale = delta_spatial.detach().abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        x = delta_spatial / scale
        angles = math.pi * x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
        return torch.cat(
            [
                x,
                torch.sin(angles).flatten(start_dim=1),
                torch.cos(angles).flatten(start_dim=1),
            ],
            dim=-1,
        )


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dropout: float,
    final_bias: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim, bias=final_bias),
    )


class SenderCategoricalFlow(nn.Module):
    """
    Permutation-equivariant categorical flow over sender candidates.

    The generator is represented implicitly with non-negative transition rates
    from each sender to a learned group-conditioned target distribution. Each
    Euler step preserves total probability mass, and the final simplex repair is
    a positive renormalization rather than a softmax projection.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int,
        n_steps: int,
        max_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_steps = max(1, int(n_steps))
        self.max_rate = max(float(max_rate), 1e-3)
        self.item_encoder = _mlp(
            int(feature_dim),
            int(hidden_dim),
            int(hidden_dim),
            dropout=dropout,
        )
        self.rate_net = _mlp(
            2 * int(hidden_dim) + 1,
            int(hidden_dim),
            2,
            dropout=dropout,
        )

    def _renormalize(
        self,
        probs: torch.Tensor,
        inv: torch.Tensor,
        n_groups: int,
        eps: float,
    ) -> torch.Tensor:
        probs = probs.clamp_min(eps)
        denom = group_sum(probs, inv, n_groups)[inv].clamp_min(eps)
        return probs / denom

    def forward(
        self,
        features: torch.Tensor,
        inv: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        if features.shape[0] == 0:
            return features.new_empty((0,))

        h = self.item_encoder(features)
        group_context = group_mean(h, inv, n_groups)
        counts = group_sum(torch.ones_like(inv, dtype=h.dtype), inv, n_groups)
        probs = 1.0 / counts[inv].clamp_min(1.0)
        dt = 1.0 / float(self.n_steps)
        eps = 1e-8

        for step in range(self.n_steps):
            t = features.new_full((features.shape[0], 1), (step + 0.5) * dt)
            rate_in = torch.cat([h, group_context[inv], t], dim=-1)
            raw = self.rate_net(rate_in)
            attractor = F.softplus(raw[:, 0]) + eps
            target = attractor / group_sum(attractor, inv, n_groups)[inv].clamp_min(eps)
            exit_rate = self.max_rate * torch.sigmoid(raw[:, 1])
            probability_mass = group_sum(probs * exit_rate, inv, n_groups)
            probs = probs + dt * (target * probability_mass[inv] - probs * exit_rate)
            probs = self._renormalize(probs, inv, n_groups, eps)
        return probs


class BinaryCategoricalFlow(nn.Module):
    """Conditional categorical flow on the inactive/active module simplex."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int,
        n_steps: int,
        max_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_steps = max(1, int(n_steps))
        self.max_rate = max(float(max_rate), 1e-3)
        self.context_encoder = _mlp(
            int(feature_dim),
            int(hidden_dim),
            int(hidden_dim),
            dropout=dropout,
        )
        self.rate_net = _mlp(
            int(hidden_dim) + 1,
            int(hidden_dim),
            2,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] == 0:
            return features.new_empty((0, 2))

        h = self.context_encoder(features)
        probs = features.new_full((features.shape[0], 2), 0.5)
        dt = 1.0 / float(self.n_steps)
        eps = 1e-8
        for step in range(self.n_steps):
            t = features.new_full((features.shape[0], 1), (step + 0.5) * dt)
            rates = self.max_rate * torch.sigmoid(
                self.rate_net(torch.cat([h, t], dim=-1))
            )
            inactive_to_active = rates[:, 0]
            active_to_inactive = rates[:, 1]
            delta_active = (
                probs[:, 0] * inactive_to_active
                - probs[:, 1] * active_to_inactive
            )
            probs = torch.stack(
                [
                    probs[:, 0] - dt * delta_active,
                    probs[:, 1] + dt * delta_active,
                ],
                dim=-1,
            )
            probs = probs.clamp_min(eps)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        return probs


class DualSimplexCategoricalFlowDecomposer(nn.Module):
    """
    Decompose inferred signalling influence as a = q * p.

    Conditional sender share is transported on each receiver-module sender
    simplex; receiver activation is transported on an independent binary
    inactive/active simplex.
    """

    def __init__(
        self,
        *,
        sender_feature_dim: int,
        receiver_feature_dim: int,
        hidden_dim: int,
        n_steps: int,
        max_rate: float,
        n_modules: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_modules = int(n_modules)
        self.sender_flow = SenderCategoricalFlow(
            sender_feature_dim,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            max_rate=max_rate,
            dropout=dropout,
        )
        self.receiver_flow = BinaryCategoricalFlow(
            receiver_feature_dim,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            max_rate=max_rate,
            dropout=dropout,
        )

    def forward(
        self,
        sender_features: torch.Tensor,
        receiver_features: torch.Tensor,
        receiver_id: torch.Tensor,
        module_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if sender_features.shape[0] == 0:
            z = sender_features.new_empty((0,))
            return {"event_influence": z, "module_activation_prob": z, "sender_share": z}

        group_key = receiver_id.long() * self.n_modules + module_id.long()
        _, inv, n_groups = group_index(group_key)
        sender_share = self.sender_flow(sender_features, inv, n_groups)
        group_receiver_features = group_mean(receiver_features, inv, n_groups)
        receiver_state = self.receiver_flow(group_receiver_features)
        module_activation_prob = receiver_state[:, 1][inv]
        return {
            "event_influence": module_activation_prob * sender_share,
            "module_activation_prob": module_activation_prob,
            "sender_share": sender_share,
        }


class RNAKineticEncoder(nn.Module):
    """z_i = GCN(phi_in(u_i, s_i, c_i); G_sp) via torch_geometric GCNConv."""

    def __init__(self, n_genes: int, config: CellVOCALModelConfig) -> None:
        super().__init__()
        self.config = config
        self._ann_dim = 0
        self._ann_embed: Optional[nn.Embedding] = None
        self._input_proj = nn.Linear(2 * n_genes, config.hidden_dim)
        n_layers = max(1, int(config.gcn_num_layers))
        self._gcn_layers = nn.ModuleList(
            [GCNConv(config.hidden_dim, config.hidden_dim) for _ in range(n_layers)]
        )
        self._out_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def set_annotation_vocab(self, n_categories: int, embed_dim: int = 8) -> None:
        if n_categories < 1:
            return
        device = next(self.parameters()).device
        self._ann_dim = embed_dim
        self._ann_embed = nn.Embedding(n_categories, embed_dim).to(device)
        old_in = self._input_proj.in_features
        self._input_proj = nn.Linear(old_in + embed_dim, self.config.hidden_dim).to(device)

    def forward(
        self,
        unspliced: torch.Tensor,
        spliced: torch.Tensor,
        edge_index: torch.Tensor,
        annotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            raise ValueError("edge_index must be non-empty for GCN encoder")
        parts = [
            torch.log1p(torch.relu(unspliced)),
            torch.log1p(torch.relu(spliced)),
        ]
        if self._ann_embed is not None and annotation is not None:
            ann = annotation.long().clamp(min=0)
            parts.append(self._ann_embed(ann))
        h = F.relu(self._input_proj(torch.cat(parts, dim=-1)))
        h = F.dropout(h, p=self.config.dropout, training=self.training)
        for conv in self._gcn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)
        return self._out_proj(h)


class CCCInfluenceHead(nn.Module):
    """Estimate nonnegative event influence using two categorical flows."""

    def __init__(self, config: CellVOCALModelConfig, n_modules: int) -> None:
        super().__init__()
        self.config = config
        self.n_modules = int(n_modules)
        d_sp = config.spatial_dim
        self._spatial_enc = FourierSpatialEncoder(
            d_sp,
            n_bands=config.ccc_spatial_fourier_bands,
        )
        sender_in_dim = (
            config.latent_dim
            + config.module_embed_dim
            + self._spatial_enc.out_dim
        )
        receiver_in_dim = config.latent_dim + config.module_embed_dim
        self.influence_decomposer = DualSimplexCategoricalFlowDecomposer(
            sender_feature_dim=sender_in_dim,
            receiver_feature_dim=receiver_in_dim,
            hidden_dim=config.ccc_flow_hidden_dim,
            n_steps=config.ccc_flow_steps,
            max_rate=config.ccc_flow_rate_max,
            n_modules=self.n_modules,
            dropout=config.dropout,
        )
        self._sender_module_embed = nn.Embedding(
            self.n_modules,
            config.module_embed_dim,
        )
        self._receiver_module_embed = nn.Embedding(
            self.n_modules,
            config.module_embed_dim,
        )
        self._sender_semantic_proj: Optional[nn.Sequential] = None
        self._receiver_semantic_proj: Optional[nn.Sequential] = None
        self.event_semantic_dim: Optional[int] = None

    def _module_activation_prior(
        self,
        event_signal: torch.Tensor,
        receiver_id: torch.Tensor,
        module_id: torch.Tensor,
    ) -> torch.Tensor:
        group_key = receiver_id.long() * self.n_modules + module_id.long()
        _, inv_g, n_groups = group_index(group_key)
        out_dtype = event_signal.dtype
        signal = _ccc_prior_signal(event_signal).to(dtype=out_dtype).clamp(0.0, 1.0)
        return group_mean(signal, inv_g, n_groups)[inv_g].clamp(0.0, 1.0)

    def _sender_share_prior(
        self,
        event_signal: torch.Tensor,
        receiver_id: torch.Tensor,
        module_id: torch.Tensor,
    ) -> torch.Tensor:
        group_key = receiver_id.long() * self.n_modules + module_id.long()
        _, inv_g, n_groups = group_index(group_key)
        signal = _ccc_prior_signal(event_signal).to(
            dtype=event_signal.dtype
        ).clamp_min(1e-6)
        prior = signal / group_sum(signal, inv_g, n_groups)[inv_g].clamp_min(1e-6)
        return prior.to(dtype=event_signal.dtype)

    def set_event_semantic_dim(self, semantic_dim: int) -> None:
        """Enable translator event semantics through side-specific projections."""
        if semantic_dim <= 0:
            raise ValueError(f"semantic_dim must be positive, got {semantic_dim}")
        if (
            self.event_semantic_dim == semantic_dim
            and self._sender_semantic_proj is not None
            and self._receiver_semantic_proj is not None
        ):
            return
        device = next(self.parameters()).device
        self.event_semantic_dim = int(semantic_dim)
        self._sender_semantic_proj = _mlp(
            semantic_dim,
            self.config.module_embed_dim,
            self.config.module_embed_dim,
            dropout=self.config.dropout,
        ).to(device)
        self._receiver_semantic_proj = _mlp(
            semantic_dim,
            self.config.module_embed_dim,
            self.config.module_embed_dim,
            dropout=self.config.dropout,
        ).to(device)

    def _module_contexts(
        self,
        module_id: torch.Tensor,
        event_semantics: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module_id = module_id.long().clamp(min=0, max=self.n_modules - 1)
        sender_ctx = self._sender_module_embed(module_id)
        receiver_ctx = self._receiver_module_embed(module_id)
        if event_semantics is not None:
            if (
                self._sender_semantic_proj is None
                or self._receiver_semantic_proj is None
            ):
                raise RuntimeError(
                    "Call set_event_semantic_dim before using event_semantics"
                )
            sender_ctx = sender_ctx + self._sender_semantic_proj(event_semantics)
            receiver_ctx = receiver_ctx + self._receiver_semantic_proj(event_semantics)
        return sender_ctx, receiver_ctx

    def forward(
        self,
        z_sender: torch.Tensor,
        z_receiver: torch.Tensor,
        receiver_id: torch.Tensor,
        module_id: torch.Tensor,
        delta_spatial: torch.Tensor,
        lr_compatibility: torch.Tensor,
        ligand_expr: Optional[torch.Tensor] = None,
        receptor_expr: Optional[torch.Tensor] = None,
        candidate_score: Optional[torch.Tensor] = None,
        event_semantics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_sender.shape[0] == 0:
            z = z_sender.new_empty((0,))
            return {"event_influence": z, "module_activation_prob": z, "sender_share": z}

        phi_x = self._spatial_enc(delta_spatial)
        sender_ctx, receiver_ctx = self._module_contexts(module_id, event_semantics)
        sender_x = torch.cat([z_sender.detach(), sender_ctx, phi_x], dim=-1)
        receiver_x = torch.cat([z_receiver.detach(), receiver_ctx], dim=-1)
        out = self.influence_decomposer(
            sender_x,
            receiver_x,
            receiver_id,
            module_id,
        )
        evidence = lr_compatibility if candidate_score is None else candidate_score
        activation_prior_source = receptor_expr if receptor_expr is not None else evidence
        sender_prior_source = ligand_expr if ligand_expr is not None else evidence
        activation_prior = self._module_activation_prior(
            activation_prior_source,
            receiver_id,
            module_id,
        )
        blend = min(max(float(self.config.ccc_prior_blend), 0.0), 1.0)
        learned = 1.0 - blend
        module_activation_prob = blend * activation_prior + learned * out["module_activation_prob"]
        sender_share = blend * self._sender_share_prior(
            sender_prior_source, receiver_id, module_id
        ) + learned * out["sender_share"]
        return {
            "event_influence": module_activation_prob * sender_share,
            "module_activation_prob": module_activation_prob,
            "sender_share": sender_share,
        }


class RNAVelocityHead(nn.Module):
    """One closed-form splicing trajectory with additive CCC velocity."""

    def __init__(self, n_genes: int, config: CellVOCALModelConfig) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.config = config
        self._alpha_raw = nn.Parameter(torch.zeros(n_genes))
        self._beta_raw = nn.Parameter(torch.zeros(n_genes))
        self._gamma_raw = nn.Parameter(torch.zeros(n_genes))
        self._u0_raw = nn.Parameter(torch.zeros(n_genes))
        self._s0_raw = nn.Parameter(torch.zeros(n_genes))
        rho_out = nn.Linear(config.hidden_dim, n_genes)
        nn.init.normal_(rho_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(rho_out.bias)
        self._rho_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            rho_out,
        )
        self._mixing = nn.Sequential(
            nn.Linear(2 * n_genes + config.latent_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def intrinsic_dynamics(
        self,
        latent: torch.Tensor,
        latent_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one denoised ODE state and its derivatives at latent time."""
        t = latent_time.unsqueeze(-1).clamp(0.0, 1.0)
        rho_raw = self._rho_net(latent)
        rho_amp = max(0.0, float(self.config.rho_modulation_strength))
        rho_state_amp = min(
            rho_amp,
            max(0.0, float(self.config.rho_state_strength)),
        )
        rho = (1.0 + rho_amp * torch.tanh(rho_raw)).clamp_min(1e-3)
        rho_state = (
            1.0 + rho_state_amp * torch.tanh(rho_raw)
        ).clamp_min(1e-3)

        alpha_base = _softplus_pos(self._alpha_raw).unsqueeze(0)
        alpha_state = alpha_base * rho_state
        alpha_velocity = alpha_base * rho
        beta = _softplus_pos(self._beta_raw).unsqueeze(0)
        gamma = _softplus_pos(self._gamma_raw).unsqueeze(0)
        u0 = _softplus_pos(self._u0_raw).unsqueeze(0)
        s0 = _softplus_pos(self._s0_raw).unsqueeze(0)

        expb = torch.exp(-beta * t)
        expg = torch.exp(-gamma * t)
        u = u0 * expb + alpha_state / beta * (1.0 - expb)

        denom = gamma - beta
        forcing = beta * u0 - alpha_state
        near = denom.abs() < 1e-3
        denom_sign = torch.where(
            denom >= 0.0,
            torch.ones_like(denom),
            -torch.ones_like(denom),
        )
        safe_denom = torch.where(near, denom_sign * 1e-3, denom)
        transient_regular = forcing * (expb - expg) / safe_denom
        transient_limit = forcing * t * expb
        transient = torch.where(near, transient_limit, transient_regular)
        s = (
            s0 * expg
            + alpha_state / gamma * (1.0 - expg)
            + transient
        )
        u = torch.relu(u)
        s = torch.relu(s)

        velocity_u_gain = max(
            0.0,
            float(self.config.rho_velocity_unspliced_gain),
        )
        u_velocity = u + (
            velocity_u_gain
            * (alpha_velocity - alpha_state)
            / beta
            * (1.0 - expb)
        )
        u_velocity = torch.relu(u_velocity)
        du = alpha_velocity - beta * u
        velocity = beta * u_velocity - gamma * s
        return u, s, du, velocity

    def mixing_lambda(
        self,
        unspliced: torch.Tensor,
        spliced: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.cat([unspliced, spliced, latent], dim=-1)
        return self._mixing(state).squeeze(-1)

    def forward(
        self,
        latent: torch.Tensor,
        latent_time: torch.Tensor,
        ccc_delta: Optional[Dict[str, torch.Tensor]] = None,
        mixing_lambda: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kinetic_u, kinetic_s, du_int, v_int = self.intrinsic_dynamics(
            latent,
            latent_time,
        )
        lam = (
            self.mixing_lambda(kinetic_u, kinetic_s, latent)
            if mixing_lambda is None
            else mixing_lambda
        )
        if ccc_delta is None:
            du_ccc = torch.zeros_like(du_int)
            v_ccc_raw = torch.zeros_like(v_int)
        else:
            du_ccc = ccc_delta["du"]
            v_ccc_raw = ccc_delta["ds"]
        lam_e = lam.unsqueeze(-1)
        ccc_scale = max(0.0, float(self.config.ccc_velocity_scale))
        du = du_int + ccc_scale * lam_e * du_ccc
        v_ccc = ccc_scale * lam_e * v_ccc_raw
        v = v_int + v_ccc
        return {
            "velocity": v,
            "velocity_intrinsic": v_int,
            "velocity_ccc_pred": v_ccc,
            "du": du,
        }


def _scatter_ccc_to_spots(
    n_spots: int,
    n_genes: int,
    event_influence: torch.Tensor,
    receiver: torch.Tensor,
    module_id: torch.Tensor,
    basis: torch.Tensor,
    *,
    channel: Optional[int] = None,
    chunk_size: int = 32_768,
) -> torch.Tensor:
    """Sum signed event-level gene contributions into per-spot ``(n_spots, n_genes)`` without full ``(E, G)``."""
    device = event_influence.device
    dtype = basis.dtype
    out = torch.zeros(n_spots, n_genes, device=device, dtype=dtype)
    n_events = event_influence.numel()
    if n_events == 0:
        return out
    recv = receiver.long()
    mod = module_id.long()
    step = max(1, int(chunk_size))
    for start in range(0, n_events, step):
        sl = slice(start, min(start + step, n_events))
        if channel is None:
            gene_contribution = event_influence[sl].unsqueeze(-1) * basis[mod[sl]]
        else:
            gene_contribution = event_influence[sl].unsqueeze(-1) * basis[mod[sl], :, channel]
        out.scatter_add_(0, recv[sl].unsqueeze(-1).expand_as(gene_contribution), gene_contribution)
    return out


def aggregate_ccc_effects(
    n_spots: int,
    n_genes: int,
    event_influence: torch.Tensor,
    receiver: torch.Tensor,
    module_id: torch.Tensor,
    basis: torch.Tensor,
    *,
    chunk_size: int = 32_768,
) -> Dict[str, torch.Tensor]:
    """
    F_ccc_RNA(x_i, N(i), M) = sum_{j,m} a_ji^(m) B_m.

    ``basis`` has two channels: 0 = extrinsic du/dtau, 1 = extrinsic ds/dtau.
    """
    if event_influence.numel() == 0:
        device = event_influence.device
        z = torch.zeros(n_spots, n_genes, device=device, dtype=basis.dtype)
        return {"du": z, "ds": z.clone()}
    return {
        "du": _scatter_ccc_to_spots(
            n_spots,
            n_genes,
            event_influence,
            receiver,
            module_id,
            basis,
            channel=0,
            chunk_size=chunk_size,
        ),
        "ds": _scatter_ccc_to_spots(
            n_spots,
            n_genes,
            event_influence,
            receiver,
            module_id,
            basis,
            channel=1,
            chunk_size=chunk_size,
        ),
    }


class CellVOCAL(nn.Module):
    """Joint RNA kinetics, latent time, and velocity-consistent CCC on one slice."""

    def __init__(
        self,
        n_genes: int,
        n_modules: int,
        cfg: CellVOCALConfig,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_modules = n_modules
        self.cfg = cfg
        self.config = cfg.model
        self.encoder = RNAKineticEncoder(n_genes, self.config)
        self.ccc_head = CCCInfluenceHead(self.config, n_modules)
        self.kinetic_head = RNAVelocityHead(n_genes, self.config)
        self._latent_time_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        # Channels: F_ccc_u (du/dtau), F_ccc_s (ds/dtau).
        self.ccc_param_basis = nn.Parameter(torch.randn(n_modules, n_genes, 2) * 1e-3)
        self.ccc_translator: Optional[CCCEventTranslator] = None

    def init_translator(
        self,
        semantic_dim: Optional[int] = None,
        *,
        z: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
        annotation: Optional[torch.Tensor] = None,
    ) -> CCCEventTranslator:
        if self.cfg.translator.mode == "off":
            raise RuntimeError("translator is disabled in config")
        if self.ccc_translator is None:
            stack = CCCEventTranslator(self.cfg, self.n_modules)
            self.ccc_translator = stack
            self.add_module("ccc_translator", stack)
        stack = self.ccc_translator
        if z is not None:
            if tau is None or spatial is None:
                raise ValueError("init_translator requires tau and spatial when z is set")
            stack.setup_feat_builder(z=z, tau=tau, spatial=spatial, annotation=annotation)
        return stack

    @property
    def translator_semantic_dim(self) -> Optional[int]:
        return None if self.ccc_translator is None else self.ccc_translator.semantic_dim

    def configure_event_semantics(self, semantic_dim: int) -> None:
        self.ccc_head.set_event_semantic_dim(semantic_dim)

    def encode(
        self,
        unspliced: torch.Tensor,
        spliced: torch.Tensor,
        edge_index: torch.Tensor,
        annotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(unspliced, spliced, edge_index, annotation)

    def predict_latent_time(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self._latent_time_head(latent).squeeze(-1))

    def predict_latent_time_with_score(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self._latent_time_head(latent).squeeze(-1)
        return torch.sigmoid(score), score

    def score_ccc_events(self, batch: BatchTensors, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        send = batch.event_sender
        recv = batch.event_receiver
        mod = batch.event_module
        if send.numel() == 0:
            z = latent.new_zeros((0,))
            return {
                "event_influence": z,
                "module_activation_prob": z,
                "sender_share": z,
            }
        rho = batch.lr_compatibility
        if rho is None:
            raise ValueError("batch.lr_compatibility is required for CCC event scoring")
        # Directed displacement for the communication event sender -> receiver.
        delta = batch.spatial[recv] - batch.spatial[send]
        ccc_latent = latent.detach()
        return self.ccc_head(
            ccc_latent[send],
            ccc_latent[recv],
            recv,
            mod,
            delta,
            rho,
            batch.ligand_expr,
            batch.receptor_expr,
            batch.candidate_score,
            batch.event_semantics,
        )

    def predict_velocity(
        self,
        latent: torch.Tensor,
        latent_time: torch.Tensor,
        event_influence: torch.Tensor,
        receiver: torch.Tensor,
        module_id: torch.Tensor,
        n_spots: int,
    ) -> Dict[str, torch.Tensor]:
        ccc_delta = aggregate_ccc_effects(
            n_spots,
            self.n_genes,
            event_influence,
            receiver,
            module_id,
            self.ccc_param_basis,
            chunk_size=self.cfg.train.ccc_event_chunk_size,
        )
        return self.kinetic_head(
            latent, latent_time, ccc_delta=ccc_delta
        )

    def forward(self, batch: BatchTensors) -> Dict[str, torch.Tensor]:
        u, s = batch.unspliced, batch.spliced
        if batch.edge_index is None:
            raise ValueError("batch.edge_index is required for GCN encoder")
        z = self.encode(
            u,
            s,
            batch.edge_index,
            batch.annotation,
        )
        tau, time_score = self.predict_latent_time_with_score(z)
        ccc = self.score_ccc_events(batch, z)
        a = ccc["event_influence"]
        vel = self.predict_velocity(
            z,
            tau,
            a,
            batch.event_receiver,
            batch.event_module,
            batch.n_spots,
        )
        return {
            "unspliced": u,
            "spliced": s,
            "spatial": batch.spatial,
            "latent": z,
            "latent_time": tau,
            "latent_time_score": time_score,
            "ccc_event_influence": a,
            "ccc_module_activation_prob": ccc["module_activation_prob"],
            "ccc_sender_share": ccc["sender_share"],
            "event_receiver": batch.event_receiver,
            "event_module": batch.event_module,
            "lr_compatibility": batch.lr_compatibility,
            "candidate_score": batch.candidate_score,
            "ligand_expr": batch.ligand_expr,
            "receptor_expr": batch.receptor_expr,
            "edge_index": batch.edge_index,
            "velocity_target_unspliced": batch.velocity_target_unspliced,
            "velocity_target_spliced": batch.velocity_target_spliced,
            "time_prior": batch.time_prior,
            "loss_event_sample_size": self.cfg.train.loss_event_sample_size,
            "velocity_order_min_delta": self.cfg.train.velocity_order_min_delta,
            "velocity_direction_weight": self.cfg.train.velocity_direction_weight,
            "time_prior_min_delta": self.cfg.train.time_prior_min_delta,
            "time_prior_temperature": self.cfg.train.time_prior_temperature,
            "time_score_regularization_weight": (
                self.cfg.train.time_score_regularization_weight
            ),
            "ccc_joint_prior_weight": self.cfg.train.ccc_joint_prior_weight,
            "ccc_sender_expr_weight": self.cfg.train.ccc_sender_expr_weight,
            "ccc_receptor_expr_weight": self.cfg.train.ccc_receptor_expr_weight,
            **vel,
        }

    @classmethod
    def from_config(cls, n_genes: int, n_modules: int, cfg: CellVOCALConfig) -> CellVOCAL:
        return cls(n_genes=n_genes, n_modules=n_modules, cfg=cfg)
