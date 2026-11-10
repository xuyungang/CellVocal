# Mouse gut — Stereo-seq

Preprocessed spatial transcriptomics from the TopoVelo Figure 2 benchmark
(**2,381 bins × 2,000 genes**). Embryonic gut, described as E13.5 in the paper, binned at 20 μm resolution.

Source: Gu et al., *Topological velocity inference from spatial transcriptomic
data* ([paper](https://doi.org/10.1038/s41587-025-02688-8),
[processed data](https://doi.org/10.6084/m9.figshare.28516139.v2)).

## Files

- `raw/stereo-seq-e14-gut_pp.h5ad`: prepared AnnData input.
- `cellvocal/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/checkpoints/`: saved model.

The input includes `spliced`/`unspliced` layers and spatial coordinates in
`obsm["X_spatial"]`.

## Load

Run from the repository root:

```python
import anndata as ad

adata = ad.read_h5ad("paper/data/mouse_gut/raw/stereo-seq-e14-gut_pp.h5ad")
fitted = ad.read_h5ad("paper/data/mouse_gut/cellvocal/results/out.h5ad")
```

For a new fit, set `adata.obsm["spatial"] = adata.obsm["X_spatial"].copy()`
and follow the [pipeline guide](../../../PIPELINE.md). Save new outputs separately;
default settings do not reproduce all settings of the saved fit.
