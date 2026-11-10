"""Configuration dataclasses for CellVOCAL."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CellVOCALQCConfig:
    """RNA-velocity/spatial transcriptomics quality-control thresholds."""

    min_spliced_counts_per_spot: float = 1.0
    min_unspliced_counts_per_spot: float = 1.0
    min_genes_per_spot: int = 1
    min_shared_counts_per_gene: float = 20.0
    n_top_genes: int = 2000
    normalize_per_spot: bool = True


@dataclass
class CellVOCALModelConfig:
    """Architecture hyper-parameters (encoder, CCC flow, kinetic head)."""

    latent_dim: int = 32
    hidden_dim: int = 128
    spatial_dim: int = 2
    module_embed_dim: int = 16
    dropout: float = 0.1
    use_cell_annotation: bool = True
    gcn_num_layers: int = 2
    ccc_velocity_scale: float = 0.2
    ccc_prior_blend: float = 0.5
    ccc_flow_hidden_dim: int = 128
    ccc_flow_steps: int = 6
    ccc_flow_rate_max: float = 4.0
    ccc_spatial_fourier_bands: int = 4
    rho_modulation_strength: float = 1.0
    rho_state_strength: float = 0.6
    rho_velocity_unspliced_gain: float = 1.0


@dataclass
class CellVOCALGraphConfig:
    """Spatial neighbourhood and candidate CCC event construction."""

    n_spatial_neighbors: int = 30
    spatial_graph_key: str = "spatial_connectivities"
    spatial_coord_key: str = "spatial"
    self_loops: bool = False
    ccc_delta_ligand: float = 0.0
    ccc_delta_receptor: float = 0.0
    ccc_ligand_expr_quantile: float = 0.7
    ccc_receptor_expr_quantile: float = 0.7
    ccc_event_budget_per_receiver: int = 5  # top sender events per receiver-module group
    ccc_min_events_per_module: int = 10
    ccc_expr_workers: int = 12


@dataclass
class CellVOCALTranslatorConfig:
    """LLM-free soft event translator config."""

    mode: str = "train"  # "off" | "load" | "train"; all modes are LLM-free
    max_epochs: int = 20
    batch_size: int = 128
    max_train_events: int = 0  # 0 = use every candidate event
    lr: float = 3e-4
    token_hidden_dim: int = 128
    num_soft_tokens: int = 4
    translator_num_layers: int = 2
    translator_nhead: int = 8
    translator_ff_dim: int = 512
    translator_dropout: float = 0.05
    cell_type_vocab_size: int = 1024
    cellchatdb_dir: Path = Path("resource/cellchatdb")
    cellchat_species: str = "mouse"
    lambda_mod: float = 0.2
    lambda_rec: float = 0.1
    lambda_scalar: float = 0.2


@dataclass
class CellVOCALTrainConfig:
    """Training loop, optimizer, and loss weights."""

    max_epochs: int = 1000
    early_stop_patience: int = 120  # 0 = disabled
    early_stop_warmup_epochs: int = 100
    early_stop_min_delta: float = 0.0
    early_stop_metric: str = "loss"
    early_stop_mode: str = "min"  # "min" | "max"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1000  # receiver-cell batch size for main CellVOCAL train/inference
    ccc_event_chunk_size: int = 32_768  # CCC event scatter chunk size inside model forward
    loss_event_sample_size: int = 2048  # stochastic pairs for latent-time loss
    velocity_order_min_delta: float = 0.02
    velocity_direction_weight: float = 1.0
    # Optional paired layers defining the local finite-difference
    # supervision space for the shared kinetic trajectory. Defaults prefer
    # TopoVelo Mu/Ms; the pipeline falls back to unspliced/spliced if absent.
    velocity_target_unspliced_layer: str | None = "Mu"
    velocity_target_spliced_layer: str | None = "Ms"
    time_prior_source: str = "spatial"  # "rna_maturity" | "spatial" | "obs"
    time_prior_obs_key: str | None = None
    time_prior_spatial_axes: tuple[int, ...] = (0,)
    time_prior_spatial_signs: tuple[float, ...] = ()
    time_prior_smoothing_steps: int = 0
    time_prior_smoothing_alpha: float = 0.8
    time_prior_min_delta: float = 0.25
    time_prior_temperature: float = 10.0
    time_score_regularization_weight: float = 0.01
    device: str = "cuda:0"
    checkpoint_dir: Path = Path("checkpoints")
    # L = w_velocity * L_velocity + w_time * L_time + w_ccc * L_ccc.
    w_velocity: float = 1.0
    w_time: float = 0.3
    w_ccc: float = 0.3
    grad_clip: float = 1.0
    ccc_joint_prior_weight: float = 0.25
    ccc_sender_expr_weight: float = 0.5
    ccc_receptor_expr_weight: float = 0.5


@dataclass
class CellVOCALEvalConfig:
    """Post-training evaluation and export."""

    output_dir: Path = Path("outputs/cellvocal")
    velocity_key: str = "cv"
    seed: int = 2022
    umap_n_neighbors: int = 40
    umap_n_pcs: int = 30
    translator_plot_max_events: int = 3000
    translator_metric_max_events: int = 3000
    translator_plot_dpi: int = 300
    benchmark_n_clusters: int = 5
    benchmark_n_jobs: int = 8
    lr_spatial_top_pairs: int = 3
    lr_spatial_top_fraction: float = 0.05


@dataclass
class CellVOCALConfig:
    """Configuration for CellVOCAL training, inference, and analysis."""

    qc: CellVOCALQCConfig = field(default_factory=CellVOCALQCConfig)
    model: CellVOCALModelConfig = field(default_factory=CellVOCALModelConfig)
    graph: CellVOCALGraphConfig = field(default_factory=CellVOCALGraphConfig)
    translator: CellVOCALTranslatorConfig = field(default_factory=CellVOCALTranslatorConfig)
    train: CellVOCALTrainConfig = field(default_factory=CellVOCALTrainConfig)
    eval: CellVOCALEvalConfig = field(default_factory=CellVOCALEvalConfig)
