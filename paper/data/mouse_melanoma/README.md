# Mouse melanoma — Visium

Three mouse melanoma sections with 10x Visium expression, velocyto RNA counts
and transferred cell annotations. Each input contains 31,053 genes (mm10).

Source: Karras et al., *A cellular hierarchy in melanoma uncouples growth and
metastasis* ([Nature, 2022](https://doi.org/10.1038/s41586-022-05242-7));
[GEO GSE207592](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207592).

## Samples and files

| Sample ID | GEO sample | Spots |
|-----------|------------|------:|
| `Visium_melanoma_2` | GSM6319696 | 3,239 |
| `Visium_melanoma_3` | GSM6319697 | 2,683 |
| `Visium_melanoma_4` | GSM6319698 | 2,641 |

- `raw/<sample_id>.h5ad`: prepared AnnData input, including `spliced`/`unspliced`
  layers and `obsm["spatial"]` coordinates.
- `cellvocal/<sample_id>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<sample_id>/checkpoints/`: saved model.

`obs["malignant_state_prior"]` orders states X0 → X1 → X3 → X4 → X5 → X6,
with spatial imputation for missing states. It is an annotation-derived prior,
not measured developmental time.

## Load

Run from the repository root:

```python
import anndata as ad

sample = "Visium_melanoma_3"
adata = ad.read_h5ad(f"paper/data/mouse_melanoma/raw/{sample}.h5ad")
fitted = ad.read_h5ad(f"paper/data/mouse_melanoma/cellvocal/{sample}/results/out.h5ad")
```

For a new fit, follow the [pipeline guide](../../../PIPELINE.md). To use the
annotation prior, set `cfg.train.time_prior_source = "obs"` and
`cfg.train.time_prior_obs_key = "malignant_state_prior"`. The LR CSV used for the
saved fits is not included; use mouse CellChatDB or supply an LR prior.
Save new outputs separately.
