# Event-attribution benchmark

Synthetic spatial RNA data for evaluating which sender cells contribute to a
receiver response. Each input has 3,000 cells, 500 genes and four communication modules.

| Scenario | Description |
|----------|-------------|
| `organizer_dominant` | One dominant sender |
| `collective_redundant` | Several redundant nearby senders |
| `contact_dependent` | Short-range contacts |
| `graded_paracrine` | Contributions vary with ligand distance |
| `developmental_switch` | Sender program changes with receiver stage |

## Files and loading

- `raw/<scenario>.h5ad`: simulated input.
- `cellvocal/<scenario>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<scenario>/checkpoints/`: saved model.

Inputs include `spliced`/`unspliced` layers, spatial coordinates and velocity
truth in `obsm`, `obs["true_time"]`, and event-attribution labels in `uns["ground_truth"]`.
Hidden simulator-defined q/p/a values are not exported.

Run from the repository root:

```python
import anndata as ad

adata = ad.read_h5ad("paper/data/event_attribution/raw/organizer_dominant.h5ad")
fitted = ad.read_h5ad("paper/data/event_attribution/cellvocal/organizer_dominant/results/out.h5ad")
```

## Generate or fit

Generate separate inputs from the repository root; add `--mode <scenario>` to select a case:

```bash
python paper/simulation/event_attribution_pipe.py --output-dir outputs/event_attribution/raw
```

For training, see the [notebook](../../notebook/event_attribution_simulation.ipynb)
and [pipeline guide](../../../PIPELINE.md). Use the synthetic LR pairs in
`adata.uns["ground_truth"]["lr_pairs"]` instead of CellChatDB; the notebook shows
the conversion. Existing fits use `cfg.train.time_prior_source = "rna_maturity"`.
Save new models and results separately.

Event-attribution inputs and fitted outputs are stored under
`paper/data/event_attribution/`.
