# Direction-recovery benchmark

Synthetic spatial RNA data for evaluating developmental direction and cell-cell
communication. Each input has 500 genes and four communication modules.

| Geometry | Description | Cells |
|----------|-------------|------:|
| `linear` | One linear trajectory | 3,000 |
| `bilinear` | Two directional branches | 3,000 |
| `radial` | One radial trajectory | 3,000 |
| `multiradial` | Four radial centers | 12,000 |

## Files and loading

- `raw/<geometry>.h5ad`: simulated input.
- `cellvocal/<geometry>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<geometry>/checkpoints/`: saved model.

Inputs include `spliced`/`unspliced` layers, spatial coordinates and velocity
truth in `obsm`, `obs["true_time"]`, and communication truth in `uns["ground_truth"]`.

Run from the repository root:

```python
import anndata as ad

adata = ad.read_h5ad("paper/data/direction/raw/linear.h5ad")
fitted = ad.read_h5ad("paper/data/direction/cellvocal/linear/results/out.h5ad")
```

## Generate or fit

Generate a separate set of inputs from the repository root (requires `topovelo`):

```python
from paper.simulation.direction_pipe import SimPipe

SimPipe(sim_dir="outputs/direction/raw", seed=42).run()
```

For training, see the [notebook](../../notebook/direction_simulation.ipynb) and
[pipeline guide](../../../PIPELINE.md). Use the synthetic LR pairs in
`adata.uns["ground_truth"]["lr_pairs"]` instead of CellChatDB; the notebook shows
the conversion. Existing fits use `cfg.train.time_prior_source = "rna_maturity"`.
Save new models and results separately.
