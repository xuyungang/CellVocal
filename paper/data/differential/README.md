# Differential communication benchmark

Synthetic data for distinguishing changes in receiver activity from changes in
conditional sender shares. Each input has four control and four perturbed replicates,
with 240 cells per replicate: 1,920 cells × 180 genes.

| Experiment | Change |
|------------|--------|
| `N00` | No change (null control) |
| `A01`–`A09` | Receiver module activation: q changes while p stays fixed |
| `R01`–`R09` | Sender shift: p changes while q stays fixed |
| `M01`–`M09` | Both |

Within each non-null series, IDs 01–03, 04–06 and 07–09 use affected fractions
of 5%, 20% and 40%, respectively. Each group uses effect strengths 0.25, 0.50 and 1.00.

## Files and loading

- `raw/<experiment>.h5ad`: simulated input (28 experiments total).
- `cellvocal/<experiment>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<experiment>/checkpoints/`: saved model.

Inputs include sample/condition labels in `obs`, `spliced`/`unspliced` layers,
spatial coordinates and velocity truth in `obsm`, and experimental design,
LR pairs and differential interaction labels in `uns["ground_truth"]`.

Run from the repository root:

```python
import anndata as ad

adata = ad.read_h5ad("paper/data/differential/raw/N00.h5ad")
fitted = ad.read_h5ad("paper/data/differential/cellvocal/N00/results/out.h5ad")
```

## Generate or fit

Generate separate inputs from the repository root; add `--experiment A01` to select a case:

```bash
python paper/simulation/differential_pipe.py --output-dir outputs/differential/raw
```

For training, see the [notebook](../../notebook/differential_simulation.ipynb) and
[pipeline guide](../../../PIPELINE.md). Use the synthetic LR pairs in
`adata.uns["ground_truth"]["lr_pairs"]` instead of CellChatDB; the notebook shows
the conversion. Existing fits use `cfg.train.time_prior_source = "rna_maturity"`.
Save new models and results separately.
