"""LLM-free soft event translator and pretraining utilities."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from model.train.config import CellVOCALConfig
from model.train.types import CCCEventIndex


def slice_events(events: CCCEventIndex, idx: np.ndarray) -> CCCEventIndex:
    """Return a numpy-indexed subset of CCC events."""
    return CCCEventIndex(
        sender=events.sender[idx],
        receiver=events.receiver[idx],
        module=events.module[idx],
        lr_compatibility=events.lr_compatibility[idx],
        candidate_score=events.candidate_score[idx],
        ligand_expr=None if events.ligand_expr is None else events.ligand_expr[idx],
        receptor_expr=None if events.receptor_expr is None else events.receptor_expr[idx],
    )


def _as_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    value = value.detach()
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=False)


def _event_scalar(
    arr: Optional[np.ndarray],
    *,
    fallback: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if arr is None:
        return fallback.to(device=device, dtype=torch.float32)
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def _match_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    return F.pad(x, (0, dim - x.shape[-1]))


def _mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    hidden = int(hidden_dim or output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
        nn.LayerNorm(output_dim),
    )


class ComponentTokenEncoder(nn.Module):
    """Encode each CCC event as the five evidence tokens T_e."""

    num_components = 5
    semantic_component_count = 5

    def __init__(
        self,
        config: CellVOCALConfig,
        *,
        num_modules: int,
        num_cell_types: int | None = None,
    ) -> None:
        super().__init__()
        tcfg = config.translator
        mcfg = config.model
        hidden_dim = int(tcfg.token_hidden_dim)
        dropout = float(tcfg.translator_dropout)
        cell_types = int(num_cell_types or tcfg.cell_type_vocab_size)

        self.module_emb = nn.Embedding(max(1, int(num_modules)), hidden_dim)
        self.cell_type_emb = nn.Embedding(max(1, cell_types), hidden_dim)
        state_dim = int(mcfg.latent_dim) + 1 + hidden_dim
        self.sender_state_proj = _mlp(
            state_dim,
            hidden_dim,
            hidden_dim=mcfg.hidden_dim,
            dropout=dropout,
        )
        self.receiver_state_proj = _mlp(
            state_dim,
            hidden_dim,
            hidden_dim=mcfg.hidden_dim,
            dropout=dropout,
        )
        self.expr_proj = _mlp(3, hidden_dim, hidden_dim=mcfg.hidden_dim, dropout=dropout)
        self.spatial_proj = _mlp(
            int(mcfg.spatial_dim) + 2,
            hidden_dim,
            hidden_dim=mcfg.hidden_dim,
            dropout=dropout,
        )
        self.component_type_emb = nn.Parameter(torch.empty(self.num_components, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.component_type_emb, std=0.02)

    @staticmethod
    def _ids(ids: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        return ids.long().clamp(min=0, max=emb.num_embeddings - 1)

    def forward(self, event_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        module_id = event_inputs["module_id"]
        if module_id.ndim != 2:
            raise ValueError("module_id must have shape [batch_size, top_b]")

        mol_token = self.module_emb(self._ids(module_id, self.module_emb))
        sender_type = self.cell_type_emb(
            self._ids(event_inputs["sender_type_id"], self.cell_type_emb)
        )
        receiver_type = self.cell_type_emb(
            self._ids(event_inputs["receiver_type_id"], self.cell_type_emb)
        )

        sender_state = event_inputs["sender_state"]
        if sender_state.ndim == 2:
            sender_state = sender_state.unsqueeze(1)
        sender_time = event_inputs["sender_latent_time"]
        if sender_time.ndim == 2:
            sender_time = sender_time.unsqueeze(1)
        receiver_state = event_inputs["receiver_state"]
        if receiver_state.ndim == 2:
            receiver_state = receiver_state.unsqueeze(1)
        receiver_time = event_inputs["receiver_latent_time"]
        if receiver_time.ndim == 2:
            receiver_time = receiver_time.unsqueeze(1)

        sender_token = self.sender_state_proj(
            torch.cat([sender_state, sender_time, sender_type], dim=-1)
        )
        receiver_token = self.receiver_state_proj(
            torch.cat([receiver_state, receiver_time, receiver_type], dim=-1)
        )
        expr_token = self.expr_proj(event_inputs["expr_features"])
        spatial_token = self.spatial_proj(event_inputs["spatial_features"])

        tokens = torch.stack(
            [
                mol_token,
                sender_token,
                receiver_token,
                expr_token,
                spatial_token,
            ],
            dim=2,
        )
        tokens = tokens + self.component_type_emb.view(1, 1, self.num_components, -1)
        return self.output_norm(tokens)


class _CrossAttentionTranslator(nn.Module):
    """Cross-attend event-conditioned soft queries to component tokens."""

    def __init__(self, hidden_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        m = self.memory_norm(memory)
        h, _ = self.attn(q, m, m, need_weights=False)
        return query + self.dropout(h)


class SoftEventTranslatorEncoder(nn.Module):
    """Convert component tokens into continuous biological soft event tokens."""

    def __init__(self, config: CellVOCALConfig) -> None:
        super().__init__()
        tcfg = config.translator
        hidden_dim = int(tcfg.token_hidden_dim)
        nhead = int(tcfg.translator_nhead)
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"token_hidden_dim ({hidden_dim}) must be divisible by "
                f"translator_nhead ({nhead})"
            )
        self.num_soft_tokens = int(tcfg.num_soft_tokens)
        self.soft_queries = nn.Parameter(torch.empty(self.num_soft_tokens, hidden_dim))
        self.query_modulator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(tcfg.translator_dropout)),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.cross_attention = _CrossAttentionTranslator(
            hidden_dim,
            nhead,
            float(tcfg.translator_dropout),
        )
        self.self_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=int(tcfg.translator_ff_dim),
                    dropout=float(tcfg.translator_dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(tcfg.translator_num_layers)))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.soft_queries, std=0.02)

    def forward(
        self,
        component_tokens: torch.Tensor,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if component_tokens.ndim != 4:
            raise ValueError(
                "component_tokens must have shape "
                "[batch_size, top_b, num_components, hidden_dim]"
            )
        batch_size, top_b, num_components, hidden_dim = component_tokens.shape
        event_summary = component_tokens.mean(dim=2)
        gamma, beta = self.query_modulator(event_summary).chunk(2, dim=-1)
        gamma = 1.0 + gamma
        queries = (
            gamma.unsqueeze(2)
            * self.soft_queries.view(1, 1, self.num_soft_tokens, hidden_dim)
            + beta.unsqueeze(2)
        )

        flat_q = queries.reshape(batch_size * top_b, self.num_soft_tokens, hidden_dim)
        flat_memory = component_tokens.reshape(batch_size * top_b, num_components, hidden_dim)
        soft_tokens = self.cross_attention(flat_q, flat_memory)
        for block in self.self_blocks:
            soft_tokens = block(soft_tokens)
        soft_tokens = self.output_norm(soft_tokens)
        soft_tokens = soft_tokens.reshape(batch_size, top_b, self.num_soft_tokens, hidden_dim)

        if event_mask is not None:
            soft_tokens = soft_tokens * event_mask.to(soft_tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        return soft_tokens


class CCCEventTranslator(nn.Module):
    """
    LLM-free event encoder that maps each CCC event to biological soft tokens.

    The encoder is pretrained, frozen, and feeds event semantics back to the
    CCC influence head as ``event_semantics``.
    """

    def __init__(self, cfg: CellVOCALConfig, n_modules: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_modules = max(1, int(n_modules))
        self.semantic_dim = int(cfg.translator.token_hidden_dim)
        self.component_encoder = ComponentTokenEncoder(
            cfg,
            num_modules=self.n_modules,
        )
        self.soft_translator = SoftEventTranslatorEncoder(cfg)
        self.event_norm = nn.LayerNorm(self.semantic_dim)
        self.module_head = nn.Linear(self.semantic_dim, self.n_modules)
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(self.semantic_dim),
            nn.Linear(self.semantic_dim, self.semantic_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.translator.translator_dropout)),
            nn.Linear(self.semantic_dim, 4),
        )
        self.component_reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.semantic_dim),
            nn.Linear(self.semantic_dim, self.semantic_dim),
            nn.GELU(),
            nn.Linear(self.semantic_dim, self.semantic_dim),
        )

    def setup_feat_builder(
        self,
        *,
        z: torch.Tensor,
        tau: torch.Tensor,
        spatial: torch.Tensor,
        annotation: Optional[torch.Tensor] = None,
    ) -> None:
        """Register frozen cell state used to materialize event component inputs."""
        tau = tau.reshape(-1, 1)
        _as_buffer(self, "z", z.float())
        _as_buffer(self, "tau", tau.float())
        _as_buffer(self, "spatial", spatial.float())
        if annotation is None:
            annotation = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        _as_buffer(self, "annotation", annotation.long())

    def _require_context(self) -> None:
        missing = [name for name in ("z", "tau", "spatial", "annotation") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Call setup_feat_builder before encoding events; missing "
                + ", ".join(missing)
            )

    @staticmethod
    def _spatial_weight(
        ligand_expr: torch.Tensor,
        receptor_expr: torch.Tensor,
        candidate_score: torch.Tensor,
    ) -> torch.Tensor:
        product = torch.relu(ligand_expr) * torch.relu(receptor_expr)
        eps = 1e-6
        weight = torch.where(
            product > eps,
            torch.relu(candidate_score) / product.clamp_min(eps),
            torch.ones_like(candidate_score),
        )
        weight = torch.where(torch.isfinite(weight), weight, torch.ones_like(weight))
        return weight.clamp(0.0, 1.0)

    def event_inputs(self, events: CCCEventIndex) -> Dict[str, torch.Tensor]:
        self._require_context()
        device = self.z.device
        sender = torch.as_tensor(events.sender, device=device, dtype=torch.long)
        receiver = torch.as_tensor(events.receiver, device=device, dtype=torch.long)
        module = torch.as_tensor(events.module, device=device, dtype=torch.long).clamp(
            min=0,
            max=self.n_modules - 1,
        )
        lr = torch.as_tensor(events.lr_compatibility, device=device, dtype=torch.float32)
        score = torch.as_tensor(events.candidate_score, device=device, dtype=torch.float32)
        ligand = _event_scalar(events.ligand_expr, fallback=torch.sqrt(torch.relu(lr)), device=device)
        receptor = _event_scalar(
            events.receptor_expr,
            fallback=torch.sqrt(torch.relu(lr)),
            device=device,
        )

        delta_spatial = self.spatial[receiver] - self.spatial[sender]
        delta_spatial = _match_last_dim(delta_spatial, int(self.cfg.model.spatial_dim))
        spatial_dist = delta_spatial.norm(dim=-1, keepdim=True)
        spatial_weight = self._spatial_weight(ligand, receptor, score).unsqueeze(-1)
        expr_features = torch.stack(
            [
                torch.log1p(torch.relu(ligand)),
                torch.log1p(torch.relu(receptor)),
                torch.log1p(torch.relu(ligand) * torch.relu(receptor)),
            ],
            dim=-1,
        )
        spatial_features = torch.cat(
            [delta_spatial, spatial_dist, spatial_weight],
            dim=-1,
        )
        return {
            "module_id": module[:, None],
            "sender_type_id": self.annotation[sender].clamp(min=0)[:, None],
            "receiver_type_id": self.annotation[receiver].clamp(min=0)[:, None],
            "sender_state": self.z[sender][:, None, :],
            "sender_latent_time": self.tau[sender][:, None, :],
            "receiver_state": self.z[receiver][:, None, :],
            "receiver_latent_time": self.tau[receiver][:, None, :],
            "expr_features": expr_features[:, None, :],
            "spatial_features": spatial_features[:, None, :],
        }

    def forward_event_batch(
        self,
        events: CCCEventIndex,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        event_inputs = self.event_inputs(events)
        component_tokens = self.component_encoder(event_inputs)
        soft_tokens = self.soft_translator(component_tokens)
        event_emb = self.event_norm(soft_tokens.mean(dim=2).squeeze(1))
        return event_emb, soft_tokens.squeeze(1), component_tokens.squeeze(1)

    def forward_semantic_event_batch(self, events: CCCEventIndex) -> torch.Tensor:
        """Return q_e from the full five-token event translator."""
        event_emb, _, _ = self.forward_event_batch(events)
        return event_emb

    def pretraining_loss(
        self,
        events: CCCEventIndex,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        event_emb, soft_tokens, component_tokens = self.forward_event_batch(events)
        device = event_emb.device
        module = torch.as_tensor(events.module, device=device, dtype=torch.long).clamp(
            min=0,
            max=self.n_modules - 1,
        )
        lr = torch.as_tensor(events.lr_compatibility, device=device, dtype=torch.float32)
        score = torch.as_tensor(events.candidate_score, device=device, dtype=torch.float32)
        ligand = _event_scalar(events.ligand_expr, fallback=torch.sqrt(torch.relu(lr)), device=device)
        receptor = _event_scalar(
            events.receptor_expr,
            fallback=torch.sqrt(torch.relu(lr)),
            device=device,
        )

        module_loss = F.cross_entropy(self.module_head(event_emb), module)
        scalar_target = torch.stack(
            [
                torch.log1p(torch.relu(lr)),
                torch.log1p(torch.relu(score)),
                torch.log1p(torch.relu(ligand)),
                torch.log1p(torch.relu(receptor)),
            ],
            dim=-1,
        )
        scalar_loss = F.smooth_l1_loss(self.scalar_head(event_emb), scalar_target)
        component_target = component_tokens.mean(dim=1).detach()
        reconstruction_loss = F.mse_loss(
            self.component_reconstruction_head(event_emb),
            component_target,
        )

        tcfg = self.cfg.translator
        loss = (
            float(tcfg.lambda_mod) * module_loss
            + float(tcfg.lambda_rec) * reconstruction_loss
            + float(tcfg.lambda_scalar) * scalar_loss
        )
        metrics = {
            "translator_module": module_loss.detach(),
            "translator_reconstruction": reconstruction_loss.detach(),
            "translator_scalar": scalar_loss.detach(),
        }
        return loss, metrics, event_emb, soft_tokens


@torch.no_grad()
def infer_event_semantics(
    tr: CCCEventTranslator,
    events: CCCEventIndex,
    config: CellVOCALConfig,
) -> np.ndarray:
    n = len(events.sender)
    out = np.zeros((n, tr.semantic_dim), dtype=np.float32)
    was_training = tr.training
    tr.eval()
    for start in tqdm(
        range(0, n, config.translator.batch_size),
        desc="Event encoder infer",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        idx = np.arange(start, min(start + config.translator.batch_size, n))
        batch = slice_events(events, idx)
        event_emb = tr.forward_semantic_event_batch(batch)
        out[idx] = event_emb.detach().cpu().numpy()
    tr.train(was_training)
    return out


__all__ = [
    "CCCEventTranslator",
    "SoftEventTranslatorEncoder",
    "infer_event_semantics",
    "slice_events",
]
