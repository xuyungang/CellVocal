# Mouse brain cortex — Slide-seq

Preprocessed spatial transcriptomics from the TopoVelo Figure 2 benchmark
(**2,525 spots × 200 genes**). The filename says `adult`, while inherited documentation describes E15 cortex;
the developmental stage is unresolved.

Source: Gu et al., *Topological velocity inference from spatial transcriptomic
data* ([paper](https://doi.org/10.1038/s41587-025-02688-8),
[processed data](https://doi.org/10.6084/m9.figshare.28516139.v2)).

## Files

- `raw/slide-seq-adult-cortex_pp.h5ad`: prepared AnnData input.
- `cellvocal/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/checkpoints/`: saved model.

The input includes `spliced`/`unspliced` layers and spatial coordinates in
`obsm["X_spatial"]`.

## Load

Run from the repository root:

```python
import anndata as ad

adata = ad.read_h5ad("paper/data/mouse_brain_cortex/raw/slide-seq-adult-cortex_pp.h5ad")
fitted = ad.read_h5ad("paper/data/mouse_brain_cortex/cellvocal/results/out.h5ad")
```

For a new fit, set `adata.obsm["spatial"] = adata.obsm["X_spatial"].copy()`
and follow the [pipeline guide](../../../PIPELINE.md). Save new outputs separately;
default settings do not reproduce all settings of the saved fit.
