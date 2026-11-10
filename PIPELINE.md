# CellVOCAL pipeline and I/O contract

This is the operational guide for the reusable CellVOCAL workflow. Dataset
READMEs under `paper/data/` document provenance, input preparation, and saved
outputs; the [model card](MODEL_CARD.md) covers intended use and limitations.

## Pipeline at a glance

```text
Raw spatial RNA data
    -> dataset-specific preparation
AnnData with U, S, spatial coordinates, and gene names
    -> run_cellvocal: QC, LR filtering, spatial graph and event candidates
Model-ready cells, genes, graph, and events
    -> optional translator pretraining/loading; main-model training
Dataset-specific checkpoints
    -> batched inference and AnnData export
out.h5ad
    -> CellVOCALAnalysis and utils plotting/evaluation tools
```

The public end-to-end entry point is `pipeline.cellvocal.run_cellvocal`. Run
commands from the project root so the local `resource/cellchatdb/` resource and
relative output paths resolve correctly.

## 1. Prepare an AnnData slice

Let `n` be the number of cells or spots and `g` the number of genes before QC.

| Location | Shape | Requirement |
| --- | --- | --- |
| `adata.layers["unspliced"]` | `(n, g)` | Unspliced counts, dense or sparse |
| `adata.layers["spliced"]` | `(n, g)` | Spliced counts aligned to unspliced counts |
| `adata.obsm["spatial"]` | `(n, c)` | Finite numeric spatial coordinates; normally `c = 2` |
| `adata.var_names` | `g` | Gene identifiers matching the LR prior |
| `adata.layers["Mu"]`, `adata.layers["Ms"]` | `(n, g)` | Optional paired velocity targets used by the default setting when both exist |
| `adata.obs["clusters"]`, `["cell_type"]`, or `["domain"]` | `n` | Optional annotations when `model.use_cell_annotation=True` |
| User-provided LR pair list | list of mappings | Optional alternative to local CellChatDB |
| Observed time column in `adata.obs` | `n` | Required only for `train.time_prior_source="obs"` |

The two velocity layers must match `adata.shape`. For a custom LR list, pass
`lr_pairs=...` to `run_cellvocal`; otherwise, the pipeline reads the local
CellChatDB species selected by `config.translator.cellchat_species` (default:
`"mouse"`). The matched ligand and receptor genes must be expressed after QC.
Multi-gene complexes are retained only when all required ligand and receptor
subunits survive QC and have expression above the prior's threshold. A missing
subunit excludes the whole interaction; partial complexes are not used.
For CellChatDB single-gene records, gene matching uses `ligand` and `receptor`
directly. An unmatched or unexpressed name excludes the interaction instead of
substituting a different gene from `ligand.symbol` or `receptor.symbol`.
Multi-gene records use the symbol columns to expand complex subunits.

## 2. Configure quality control and training

```python
from pathlib import Path

import anndata as ad
from model.train.config import CellVOCALConfig
from pipeline.cellvocal import run_cellvocal

adata = ad.read_h5ad("your_data.h5ad")
cfg = CellVOCALConfig()
cfg.train.device = "cuda:0"  # use "cpu" if needed
cfg.train.checkpoint_dir = Path("outputs/cellvocal/checkpoints")
cfg.eval.output_dir = Path("outputs/cellvocal/results")
cfg.translator.mode = "off"  # "off", "load", or "train"

result = run_cellvocal(adata, config=cfg)
```

`run_cellvocal` accepts `translator_mode=...` as a per-call override and
`lr_pairs=...` for an explicit LR prior. Use `Path` values for checkpoint and
output directories.

The `CellVOCALConfig` sections are `qc`, `graph`, `model`, `train`, `eval`,
and `translator`. The default QC thresholds retain spots with at least one
count in each velocity layer and one detected velocity gene. A gene needs at
least 20 shared spliced/unspliced counts; at most 2,000 genes are retained by
total counts. Per-spot normalization is enabled by default. The pipeline copies
the input, applies QC, and records the retained spot and gene counts in
`adata.uns["cellvocal_qc"]`. Review this summary, coordinate orientation, and
LR coverage before interpreting a fit.

The default velocity targets prefer paired `Mu`/`Ms` layers when both are
present. If they are absent, the objective uses `unspliced`/`spliced`. Custom
target layer names must both exist and have the post-QC shape. Set both target
layer options to `None` to use the base layers explicitly.

## 3. Construct graph, time prior, and candidate events

After QC, `build_spatial_graph` creates a symmetric k-nearest-neighbor distance
graph at `adata.obsp[cfg.graph.spatial_graph_key]`. The default key is
`"spatial_connectivities"` and the default neighbor count is 30. Candidate
sender–receiver–LR events are filtered using this graph, matched genes, and
expression evidence. Modules without enough surviving events are removed; a
run fails if no eligible module or event remains.

The time objective uses `cfg.train.time_prior_source`:

| Value | Source |
| --- | --- |
| `"spatial"` (default) | Selected spatial axes; the first axis by default |
| `"rna_maturity"` | Mean log spliced abundance minus mean log unspliced abundance |
| `"obs"` | The column named by `cfg.train.time_prior_obs_key` |

The prior may be smoothed on the spatial graph, then is standardized. A
constant or non-finite prior raises an error. Check coordinate orientation
before using the default spatial prior.

## 4. Train and infer

The translator modes are:

| Mode | Behavior |
| --- | --- |
| `"off"` | No event translator |
| `"load"` | Load compatible local `translator.pt` from the checkpoint directory |
| `"train"` (default) | Pretrain on candidate events and save `translator.pt` |

When used, translator parameters are frozen for main-model training. The main
model uses receiver-cell mini-batches, AdamW, and an objective combining
velocity, time, and CCC terms. Defaults are 1,000 maximum epochs, early
stopping, and a batch size of 1,000 receiver cells. Set `cfg.eval.seed` for a
repeatable random seed; keep the full configuration and software environment
with the output for reproducibility.

After training, inference iterates over receiver batches and computes total,
intrinsic, and CCC-associated velocity, latent time, latent state, and event
scores. The main checkpoint is saved before inference.

## 5. Export contract

`run_cellvocal` returns a QC-filtered `AnnData` result and writes the same
result to `<eval.output_dir>/out.h5ad`. It writes
`<train.checkpoint_dir>/cellvocal.pt`, and writes `translator.pt` there when
translator training is selected. In `load` mode, an existing translator
checkpoint is required.

For the default prefix `cfg.eval.velocity_key = "cv"`:

| Result location | Shape | Meaning |
| --- | --- | --- |
| `.layers["cv_velocity"]` | `(n_qc, g_qc)` | Total predicted RNA velocity |
| `.layers["cv_intrinsic_velocity"]` | `(n_qc, g_qc)` | Intrinsic component |
| `.layers["cv_ccc_velocity"]` | `(n_qc, g_qc)` | CCC-associated component |
| `.obs["cv_time"]` | `n_qc` | Normalized latent-time estimate |
| `.obsm["cv_latent"]` | `(n_qc, cfg.model.latent_dim)` | Learned latent state |
| `.uns["cv_ccc"]` | mapping | Candidate events, scores, LR metadata, and optional translator semantics |
| `.uns["cellvocal_qc"]` | mapping | Filtering and normalization summary |
| `.obsp[cfg.graph.spatial_graph_key]` | `(n_qc, n_qc)` | Symmetric spatial neighbor graph |

In `cv_ccc`, `sender`, `receiver`, and `module` are integer indices into the
QC-filtered cells and retained LR modules. Event-aligned fields include
`event_influence`, `module_activation_prob`, `sender_share`,
`candidate_score`, and `lr_compatibility`. The `lr_pairs` and
`lr_pair_module_ids` arrays describe retained modules. Use the saved metadata
to interpret indices after QC.

The three event-aligned arrays have distinct meanings:

| Field | Definition |
| --- | --- |
| `module_activation_prob` | q_i^m, repeated for events in the same receiver/module group |
| `sender_share` | p_ji^m, the conditional relative sender share; sums to one within that group |
| `event_influence` | a_ji^m = q_i^m p_ji^m, nonnegative inferred signalling influence |

`CellVOCALResult` and model forward outputs use the corresponding
`ccc_module_activation_prob`, `ccc_sender_share`, and `ccc_event_influence`
fields. `incoming_influence` sums event influences over the selected senders
within each receiver/module. With every eligible sender retained it equals q;
source filtering does not change the denominator or renormalize p.
Gene effects and velocity contributions are signed and use the model's
configured mixing and scaling factors.

The analysis API exposes `analysis.incoming_influence()` and
`utils.tools.data.incoming_influence`. Simulation uses
`simulate_event_attribution_dataset`, `EventAttributionSimulationPipe`, and
`simulated_event_influence` for hidden simulator-defined event influences.
The differential scenario `sender_shift` (experiments R01–R09) keeps q fixed
and changes p. Public simulation outputs omit hidden q/p/a truth.

## 6. Analyze results

```python
from pathlib import Path
from pipeline.analysis import CellVOCALAnalysis

analysis = CellVOCALAnalysis.from_h5ad(
    Path("outputs/cellvocal/results/out.h5ad"),
    config=cfg,
)
# analysis.plot_velocity(...)
# analysis.evaluate_metrics(...)
```

`CellVOCALAnalysis` provides plotting and evaluation methods. Some metrics
require dataset-specific cluster labels or simulation truth and create files
only when called. The manuscript reference notebooks in `paper/notebook/`
provide dataset-specific context; consult each dataset README for available
inputs and saved CellVOCAL outputs under `paper/data/<dataset>/`. The
standalone [example notebook](example/example.ipynb) demonstrates result
loading and plotting with the shared `utils` tools.

The `human_liver`, `mouse_embryo`, and `mouse_melanoma` reference notebooks
load LR-screening tables that are not bundled. Follow their notes to skip
those initialization lines when viewing saved results; for a new fit, provide
or rebuild the tables, or configure an alternative LR prior.

For controlled data generation, run:

```bash
python paper/simulation/direction_pipe.py
python paper/simulation/differential_pipe.py
python paper/simulation/event_attribution_pipe.py
```

These generators write their default raw outputs under
`paper/data/{direction,differential,event_attribution}/raw/`. Dataset README
files document their panels, saved outputs, and synthetic LR priors. To fit a
generated AnnData file, follow that panel's README or notebook to pass its
ground-truth LR pairs explicitly to `run_cellvocal`; synthetic gene names do
not match the default CellChatDB prior.

## Documentation map

- [README.md](README.md): overview, installation, and repository layout.
- [MODEL_CARD.md](MODEL_CARD.md): scientific scope, evaluation context, and
  limitations.
- `paper/data/<dataset>/README.md`: dataset acquisition, preparation, and
  provenance.
- `paper/notebook/`: manuscript reference notebooks.
