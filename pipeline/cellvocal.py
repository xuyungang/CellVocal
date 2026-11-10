"""CellVOCAL end-to-end pipeline."""
from __future__ import annotations

import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from model import (
    CellVOCAL,
    ann_to_slice_input,
    attach_results_to_anndata,
    build_candidate_ccc_events,
    build_spatial_graph,
    prune_lr_modules_by_events,
)
from model.graph.event_features import frozen_kinetic_states
from model.graph.lr_prior import prepare_lr_pairs_for_cellvocal
from model.train.batch import (
    annotation_codes,
    index_events_by_receiver,
    make_local_batch,
)
from model.train.callbacks import EarlyStopping
from model.train.checkpoint import (
    DEFAULT_TRANSLATOR_CHECKPOINT_NAME,
    apply_translator_checkpoint,
    load_translator_checkpoint,
    save_translator_checkpoint,
    save_cellvocal_checkpoint,
    cellvocal_checkpoint_path,
)
from model.train.config import CellVOCALConfig
from model.train.data import (
    layer_to_dense_f32,
    lr_pair_labels,
    lr_pair_module_ids,
    validate_slice_anndata,
)
from model.train.qc import apply_rna_velocity_qc
from model.train.losses import LossWeights, MAIN_LOSS_KEYS, compute_total_loss, format_loss_metrics
from model.train.prior import smooth_graph_signal
from model.train.types import CCCEventIndex, CellVOCALResult, SliceInput
from model.translator import infer_event_semantics, slice_events

logger = logging.getLogger(__name__)

_TRANSLATOR_MODES = {"off", "load", "train"}
_TQDM_KWARGS = {"dynamic_ncols": True, "mininterval": 1.0}
_DEFAULT_VELOCITY_TARGET_LAYERS = ("Mu", "Ms")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("global random seed set to %d", seed)


def _balanced_event_sample_indices(
    modules: np.ndarray,
    *,
    max_events: int,
    seed: int,
) -> np.ndarray:
    """Return a fixed module-balanced event subset for Translator pretraining."""
    modules = np.asarray(modules)
    n_events = int(modules.size)
    if max_events <= 0 or n_events <= max_events:
        return np.arange(n_events, dtype=np.int64)

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(modules == module) for module in np.unique(modules)]
    rng.shuffle(groups)

    selected: list[np.ndarray] = []
    remaining_by_group: list[np.ndarray] = []
    per_group = max(1, max_events // len(groups))
    for group in groups:
        shuffled = rng.permutation(group)
        take = min(per_group, shuffled.size)
        selected.append(shuffled[:take])
        remaining_by_group.append(shuffled[take:])

    selected_idx = np.concatenate(selected)
    if selected_idx.size > max_events:
        selected_idx = rng.choice(selected_idx, size=max_events, replace=False)
    elif selected_idx.size < max_events:
        remaining = [idx for idx in remaining_by_group if idx.size]
        if remaining:
            pool = np.concatenate(remaining)
            extra = rng.choice(
                pool,
                size=min(max_events - selected_idx.size, pool.size),
                replace=False,
            )
            selected_idx = np.concatenate([selected_idx, extra])
    return np.sort(selected_idx.astype(np.int64, copy=False))


def _resolve_velocity_target_layers(
    adata: ad.AnnData,
    config: CellVOCALConfig,
) -> tuple[str | None, str | None]:
    target_layers = (
        config.train.velocity_target_unspliced_layer,
        config.train.velocity_target_spliced_layer,
    )
    if (target_layers[0] is None) != (target_layers[1] is None):
        raise ValueError(
            "velocity target layers must be configured together so the "
            "velocity objective uses one consistent U/S observation space"
        )
    if target_layers == (None, None):
        return None, None
    if all(layer in adata.layers for layer in target_layers):
        return target_layers
    if target_layers == _DEFAULT_VELOCITY_TARGET_LAYERS:
        missing = [layer for layer in target_layers if layer not in adata.layers]
        logger.info(
            "default velocity target layers %s are unavailable; using base "
            "unspliced/spliced layers",
            missing,
        )
        return None, None
    missing = [layer for layer in target_layers if layer not in adata.layers]
    raise KeyError(
        "Missing velocity target layer(s) for RNA velocity target: "
        + ", ".join(f"layers[{layer!r}]" for layer in missing)
    )


class CellVOCALPipeline:
    def __init__(
        self,
        adata: ad.AnnData,
        lr_pairs: List[Dict[str, Any]],
        config: CellVOCALConfig,
    ) -> None:
        self.adata = adata
        self.lr_pairs = list(lr_pairs)
        self.config = config
        self.n_modules = len(self.lr_pairs)
        self._model: Optional[CellVOCAL] = None
        self._graph = None
        self._events: Optional[CCCEventIndex] = None
        self._event_semantics: Optional[np.ndarray] = None
        self._weights: Optional[LossWeights] = None
        self._n_genes: Optional[int] = None
        self._slice_in: Optional[SliceInput] = None
        self._annotation: Optional[np.ndarray] = None
        self._device = torch.device(config.train.device)
        self._main_opt: Optional[torch.optim.Optimizer] = None
        self._events_by_receiver: Optional[List[np.ndarray]] = None

    def _setup(self) -> None:
        self._graph = build_spatial_graph(self.adata, self.config.graph)
        self._slice_in = ann_to_slice_input(self.adata)
        self._n_genes = self._slice_in.unspliced.shape[1]

        target_layers = _resolve_velocity_target_layers(self.adata, self.config)
        for layer, target_name in (
            (target_layers[0], "velocity_target_unspliced"),
            (target_layers[1], "velocity_target_spliced"),
        ):
            if layer is None:
                continue
            target = layer_to_dense_f32(self.adata.layers[layer])
            if target.shape != self._slice_in.spliced.shape:
                raise ValueError(
                    f"Velocity target {layer!r} has shape {target.shape}, "
                    f"expected {self._slice_in.spliced.shape}"
                )
            setattr(self._slice_in, target_name, target)

        prior_source = self.config.train.time_prior_source
        if prior_source == "rna_maturity":
            u = np.log1p(np.maximum(self._slice_in.unspliced, 0.0)).mean(axis=1)
            s = np.log1p(np.maximum(self._slice_in.spliced, 0.0)).mean(axis=1)
            time_prior = s - u
        elif prior_source == "spatial":
            axes = tuple(int(axis) for axis in self.config.train.time_prior_spatial_axes)
            if not axes:
                raise ValueError("time_prior_spatial_axes must contain at least one axis")
            if min(axes) < 0 or max(axes) >= self._slice_in.spatial.shape[1]:
                raise ValueError(
                    "time_prior_spatial_axes contains an invalid axis for coordinates "
                    f"with shape {self._slice_in.spatial.shape}: {axes}"
                )
            coordinates = np.asarray(self._slice_in.spatial[:, axes], dtype=np.float64)
            if coordinates.ndim == 1:
                coordinates = coordinates[:, None]
            configured_signs = tuple(
                float(sign) for sign in self.config.train.time_prior_spatial_signs
            )
            if configured_signs:
                if len(configured_signs) != len(axes):
                    raise ValueError(
                        "time_prior_spatial_signs must match time_prior_spatial_axes"
                    )
                coordinates *= np.asarray(configured_signs, dtype=np.float64)[None, :]
            time_prior = coordinates.mean(axis=1)
        elif prior_source == "obs":
            key = self.config.train.time_prior_obs_key
            if key is None or key not in self.adata.obs:
                raise KeyError(
                    f"time_prior_obs_key={key!r} is not available in adata.obs"
                )
            time_prior = np.asarray(self.adata.obs[key], dtype=np.float64)
        else:
            raise ValueError(
                "time_prior_source must be 'rna_maturity', 'spatial', or 'obs'"
            )
        smoothing_steps = int(self.config.train.time_prior_smoothing_steps)
        if smoothing_steps > 0:
            time_prior = smooth_graph_signal(
                time_prior,
                self._graph,
                steps=smoothing_steps,
                alpha=self.config.train.time_prior_smoothing_alpha,
            )
            logger.info(
                "smoothed time prior over spatial graph: steps=%d, alpha=%.3f",
                smoothing_steps,
                self.config.train.time_prior_smoothing_alpha,
            )
        if not np.isfinite(time_prior).all():
            raise ValueError("time prior contains non-finite values")
        prior_std = float(np.std(time_prior))
        if prior_std < 1e-6:
            raise ValueError(f"time prior from {prior_source!r} is constant")
        self._slice_in.time_prior = (
            (time_prior - float(np.mean(time_prior))) / prior_std
        ).astype(np.float32)
        logger.info(
            "unified objectives use time prior=%s and velocity U/S=(%s, %s)",
            prior_source,
            target_layers[0] or "unspliced",
            target_layers[1] or "spliced",
        )
        n_lr_before = len(self.lr_pairs)
        self._events = build_candidate_ccc_events(
            self._graph,
            self.n_modules,
            lr_pairs=self.lr_pairs,
            spliced=self._slice_in.spliced,
            gene_names=self._slice_in.gene_names,
            config=self.config.graph,
        )
        self.lr_pairs, self._events = prune_lr_modules_by_events(
            self.lr_pairs,
            self._events,
            min_events=self.config.graph.ccc_min_events_per_module,
        )
        self.n_modules = len(self.lr_pairs)
        if self.n_modules == 0 or len(self._events.sender) == 0:
            raise ValueError("No candidate CCC events survived LR/spatial filtering")
        logger.info(
            "pruned LR modules by surviving events: %d -> %d modules, %d events",
            n_lr_before,
            self.n_modules,
            len(self._events.sender),
        )
        self._annotation = annotation_codes(self.adata, self.config)
        assert self._events is not None
        self._events_by_receiver = index_events_by_receiver(
            self._events, int(self._slice_in.unspliced.shape[0])
        )

    def _build_translator_context(self):
        assert self._model is not None and self._slice_in is not None
        z, tau = frozen_kinetic_states(
            self._slice_in.unspliced,
            self._slice_in.spliced,
            latent_dim=self.config.model.latent_dim,
            device=self._device,
        )
        spatial = torch.as_tensor(
            self._slice_in.spatial,
            device=self._device,
            dtype=torch.float32,
        )
        annotation = (
            None
            if self._annotation is None
            else torch.as_tensor(self._annotation, device=self._device, dtype=torch.long)
        )
        tr = self._model.init_translator(
            z=z,
            tau=tau,
            spatial=spatial,
            annotation=annotation,
        ).to(self._device)
        self._model.configure_event_semantics(tr.semantic_dim)
        return tr

    def _load_translator(self) -> None:
        ckpt = cellvocal_checkpoint_path(self.config).with_name(DEFAULT_TRANSLATOR_CHECKPOINT_NAME)
        payload = load_translator_checkpoint(ckpt, map_location="cpu")
        tr = self._build_translator_context()
        apply_translator_checkpoint(tr, payload, strict=True)
        assert self._events is not None
        self._event_semantics = infer_event_semantics(tr, self._events, self.config)
        logger.info("loaded pretrained translator from %s", ckpt)

    def _train_translator(self) -> None:
        tcfg = self.config.translator
        assert self._events is not None

        tr = self._build_translator_context()
        opt = torch.optim.AdamW(
            tr.parameters(),
            lr=tcfg.lr,
            weight_decay=self.config.train.weight_decay,
        )

        n_total_events = len(self._events.sender)
        max_train_events = int(tcfg.max_train_events)
        train_indices = _balanced_event_sample_indices(
            self._events.module,
            max_events=max_train_events,
            seed=self.config.eval.seed,
        )
        logger.info(
            "Translator pretraining uses %d/%d candidate events across %d modules",
            train_indices.size,
            n_total_events,
            np.unique(self._events.module[train_indices]).size,
        )
        last_epoch = 0
        with logging_redirect_tqdm():
            for epoch in range(tcfg.max_epochs):
                last_epoch = epoch
                order = torch.randperm(train_indices.size).cpu().numpy()
                perm = train_indices[order]
                loss_sum = 0.0
                n_batches = 0
                batch_bar = tqdm(
                    range(0, len(perm), tcfg.batch_size),
                    desc=f"Translator {epoch + 1}/{tcfg.max_epochs}",
                    unit="batch",
                    leave=(epoch + 1 == tcfg.max_epochs),
                    **_TQDM_KWARGS,
                )
                for start in batch_bar:
                    idx = perm[start : start + tcfg.batch_size]
                    batch = slice_events(self._events, idx)
                    loss, metrics, _, _ = tr.pretraining_loss(batch)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tr.parameters(), self.config.train.grad_clip)
                    opt.step()
                    loss_sum += float(loss.item())
                    n_batches += 1
                    metric_text = f"loss={loss_sum / n_batches:.4f}"
                    if metrics:
                        metric_text += ", " + ", ".join(
                            f"{k}={float(v.item()):.4f}" for k, v in metrics.items()
                        )
                    batch_bar.set_postfix_str(metric_text, refresh=False)

        save_translator_checkpoint(
            tr,
            cellvocal_checkpoint_path(self.config).with_name(DEFAULT_TRANSLATOR_CHECKPOINT_NAME),
            config=self.config,
            semantic_dim=tr.semantic_dim,
            epoch=last_epoch,
        )
        self._event_semantics = infer_event_semantics(tr, self._events, self.config)

    def _prepare_translator(self) -> None:
        mode = self.config.translator.mode
        if mode == "off":
            return
        if mode == "load":
            self._load_translator()
        elif mode == "train":
            self._train_translator()
        else:
            raise ValueError("translator mode must be 'off', 'load', or 'train'")

    def _freeze_translator(self) -> None:
        if self._model is None or self._model.ccc_translator is None:
            return
        for p in self._model.ccc_translator.parameters():
            p.requires_grad = False

    def train(self) -> None:
        self._setup()
        self.config.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model = CellVOCAL.from_config(self._n_genes, self.n_modules, self.config)
        ann = self._annotation
        if ann is not None:
            self._model.encoder.set_annotation_vocab(int(ann.max()) + 1)
        self._model.to(self._device)
        self._prepare_translator()
        self._freeze_translator()

        main_params = [p for p in self._model.parameters() if p.requires_grad]
        self._main_opt = torch.optim.AdamW(
            main_params,
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        self._weights = LossWeights(
            velocity=self.config.train.w_velocity,
            time=self.config.train.w_time,
            ccc=self.config.train.w_ccc,
        )
        early_stop = EarlyStopping.from_train_config(self.config.train)
        early_stop_start = max(0, int(self.config.train.early_stop_warmup_epochs))
        last_epoch = 0
        best_epoch = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        epoch_bar = tqdm(range(self.config.train.max_epochs), desc="CellVOCAL train", unit="epoch", **_TQDM_KWARGS)
        with logging_redirect_tqdm():
            for epoch in epoch_bar:
                metrics = self._train_epoch()
                last_epoch = epoch
                epoch_bar.set_postfix_str(format_loss_metrics(metrics), refresh=False)
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                if epoch < early_stop_start:
                    continue
                previous_best_epoch = early_stop.state.best_epoch
                if early_stop.step(metrics, epoch):
                    break
                if early_stop.state.best_epoch != previous_best_epoch:
                    best_epoch = epoch
                    best_state_dict = deepcopy(
                        {
                            key: value.detach().cpu()
                            for key, value in self._model.state_dict().items()
                        }
                    )
        if best_state_dict is not None and best_epoch != last_epoch:
            self._model.load_state_dict(
                {key: value.to(self._device) for key, value in best_state_dict.items()}
            )
            last_epoch = best_epoch
            logger.info("Restored best CellVOCAL epoch %d before export/checkpoint", best_epoch)
        save_cellvocal_checkpoint(
            self._model,
            cellvocal_checkpoint_path(self.config),
            config=self.config,
            n_genes=self._n_genes,
            n_modules=self.n_modules,
            epoch=last_epoch,
            event_semantics=self._event_semantics,
        )

    def _train_epoch(self) -> Dict[str, float]:
        assert self._model is not None and self._slice_in is not None and self._events is not None
        assert self._main_opt is not None and self._weights is not None
        assert self._events_by_receiver is not None and self._graph is not None

        self._model.train()
        n_spots = int(self._slice_in.unspliced.shape[0])
        cell_batch_size = max(1, int(self.config.train.batch_size))
        perm = torch.randperm(n_spots).cpu().numpy().astype(np.int64, copy=False)
        totals: Dict[str, float] = {}
        total_weight = 0.0
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]

        for start in range(0, n_spots, cell_batch_size):
            receiver_batch = perm[start : start + cell_batch_size]
            batch, _, receiver_ids, _ = make_local_batch(
                receiver_batch,
                slice_in=self._slice_in,
                events=self._events,
                events_by_receiver=self._events_by_receiver,
                graph=self._graph,
                config=self.config,
                device=self._device,
                annotation=self._annotation,
                event_semantics=self._event_semantics,
            )
            self._main_opt.zero_grad(set_to_none=True)
            outputs = self._model(batch)
            loss, metrics = compute_total_loss(outputs, self._weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.train.grad_clip)
            self._main_opt.step()

            weight = float(max(1, receiver_ids.shape[0]))
            total_weight += weight
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
        if total_weight == 0.0:
            return {key: 0.0 for key in MAIN_LOSS_KEYS}
        return {key: value / total_weight for key, value in totals.items()}

    def _run_inference(self, device: str) -> CellVOCALResult:
        assert self._model is not None and self._slice_in is not None and self._events is not None
        assert self._events_by_receiver is not None and self._graph is not None
        dev = torch.device(device)
        n_spots, n_genes = self._slice_in.unspliced.shape
        n_events = int(self._events.sender.shape[0])
        latent_dim = int(self.config.model.latent_dim)
        velocity_intrinsic = np.zeros((n_spots, n_genes), dtype=np.float32)
        velocity_ccc = np.zeros((n_spots, n_genes), dtype=np.float32)
        latent_time = np.zeros(n_spots, dtype=np.float32)
        latent_state = np.zeros((n_spots, latent_dim), dtype=np.float32)
        ccc_event_influence = np.zeros(n_events, dtype=np.float32)
        ccc_module_activation_prob = np.zeros(n_events, dtype=np.float32)
        ccc_sender_share = np.zeros(n_events, dtype=np.float32)

        cell_batch_size = max(1, int(self.config.train.batch_size))
        all_receivers = np.arange(n_spots, dtype=np.int64)
        for start in range(0, n_spots, cell_batch_size):
            receiver_batch = all_receivers[start : start + cell_batch_size]
            batch, event_idx, receiver_ids, target_local = make_local_batch(
                receiver_batch,
                slice_in=self._slice_in,
                events=self._events,
                events_by_receiver=self._events_by_receiver,
                graph=self._graph,
                config=self.config,
                device=dev,
                annotation=self._annotation,
                event_semantics=self._event_semantics,
            )
            out = self._model(batch)
            target_t = torch.as_tensor(target_local, device=dev, dtype=torch.long)
            vel_int = out["velocity_intrinsic"].index_select(0, target_t).detach().cpu().numpy()
            vel_ccc = out["velocity_ccc_pred"].index_select(0, target_t).detach().cpu().numpy()
            velocity_intrinsic[receiver_ids] = vel_int
            velocity_ccc[receiver_ids] = vel_ccc
            latent_time[receiver_ids] = (
                out["latent_time"].index_select(0, target_t).detach().cpu().numpy()
            )
            latent_state[receiver_ids] = (
                out["latent"].index_select(0, target_t).detach().cpu().numpy()
            )
            if event_idx.size:
                ccc_event_influence[event_idx] = (
                    out["ccc_event_influence"].detach().cpu().numpy()
                )
                ccc_module_activation_prob[event_idx] = out["ccc_module_activation_prob"].detach().cpu().numpy()
                ccc_sender_share[event_idx] = (
                    out["ccc_sender_share"].detach().cpu().numpy()
                )

        return CellVOCALResult(
            velocity=velocity_intrinsic + velocity_ccc,
            latent_time=latent_time,
            latent_state=latent_state,
            velocity_intrinsic=velocity_intrinsic,
            velocity_ccc=velocity_ccc,
            ccc_event_influence=ccc_event_influence,
            ccc_module_activation_prob=ccc_module_activation_prob,
            ccc_sender_share=ccc_sender_share,
            event_index=self._events,
            event_semantics=self._event_semantics,
            lr_pair_labels=lr_pair_labels(self.lr_pairs),
            lr_pair_module_ids=lr_pair_module_ids(self.lr_pairs),
        )

    def predict_and_export(self) -> ad.AnnData:
        if self._model is None:
            raise RuntimeError("Model not trained")
        if self._slice_in is None or self._events is None:
            self._setup()
        self._model.eval()
        assert self._events_by_receiver is not None and self._graph is not None
        with torch.no_grad():
            result = self._run_inference(self.config.train.device)
        prefix = self.config.eval.velocity_key
        adata_out = attach_results_to_anndata(self.adata, result, prefix=prefix)
        for key in ("count_matrix_file", "spatial_metadata_file", "velocity_matrix_dir"):
            adata_out.uns.pop(key, None)
        adata_out.uns.get("human_liver_spot_annotations", {}).pop("source", None)
        adata_out.uns.get("spatial", {}).pop("image_paths", None)
        out_dir = Path(self.config.eval.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "out.h5ad"
        adata_out.write_h5ad(output_path)
        print(
            f"[{self.config.eval.velocity_key}] CellVOCAL done → "
            f"{self.config.train.checkpoint_dir}, results: {output_path}"
        )
        return adata_out


def run_cellvocal(
    adata: ad.AnnData,
    config: Optional[CellVOCALConfig] = None,
    *,
    translator_mode: Optional[str] = None,
    lr_pairs: Optional[List[Dict[str, Any]]] = None,
) -> ad.AnnData:
    """Train CellVOCAL on ``adata``, infer, and save h5ad."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",)
    cfg = config or CellVOCALConfig()
    if translator_mode is not None:
        cfg.translator.mode = translator_mode
    if cfg.translator.mode not in _TRANSLATOR_MODES:
        raise ValueError("translator_mode must be 'off', 'load', or 'train'")
    validate_slice_anndata(adata)
    _set_global_seed(cfg.eval.seed)
    adata = apply_rna_velocity_qc(adata, copy=True, **vars(cfg.qc))
    prepared_lr_pairs = prepare_lr_pairs_for_cellvocal(adata, cfg, lr_pairs=lr_pairs)
    logger.info("using %d ligand-receptor modules after prior filtering", len(prepared_lr_pairs))
    pipeline = CellVOCALPipeline(adata, prepared_lr_pairs, cfg)
    pipeline.train()
    return pipeline.predict_and_export()

__all__ = ["run_cellvocal", "CellVOCALPipeline"]
