# Mouse embryo — SPTEdU-seq

Three mouse embryo sections with corrected linker-aware RNA counts and existing
CellVOCAL fits. `reprocessed_linker_v2` identifies the input version in metadata.

Source: Niu et al., *SPTEdU-seq enables parallel optics-free newborn cell tracking
and spatial total transcriptional dynamics in intact microenvironments*
([Cell Stem Cell, 2026](https://doi.org/10.1016/j.stem.2026.03.001));
[GEO GSE283159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283159);
[processed data](https://doi.org/10.6084/m9.figshare.27923973).

## Samples and files

| Stage | Sample ID | Input (sites × genes) | Fitted result |
|-------|-----------|----------------------|---------------|
| E12.5 | `GSM9421073` | 114,692 × 31,821 | 49,908 × 1,960 |
| E14.5 | `GSM9421074` | 147,345 × 30,815 | 49,883 × 1,954 |
| E15.5 | `GSM9421075` | 136,419 × 29,839 | 49,756 × 1,949 |

- `raw/<sample_id>_SPTEdU.h5ad`: prepared AnnData input.
- `cellvocal/<sample_id>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<sample_id>/checkpoints/`: saved model.

Inputs contain the author expression matrix in `X`, corrected `spliced`,
`unspliced` and `ambiguous` layers, coordinates in `obsm["spatial"]`, and
annotations/UTI metadata in `obs`. Rows are capture sites, not segmented cells.

UTI measures retained EdU DNA tags; UTI/UMI is a tag-to-RNA ratio, not nascent
or future RNA. All retained sites have UTI ≥ 1, with no unlabelled control.
There is one library per stage; spatial blocks are not biological replicates.

## Load

Run from the repository root (these inputs require substantial RAM):

```python
import anndata as ad

sample = "GSM9421073"
adata = ad.read_h5ad(f"paper/data/mouse_embryo/raw/{sample}_SPTEdU.h5ad")
fitted = ad.read_h5ad(f"paper/data/mouse_embryo/cellvocal/{sample}/results/out.h5ad")
```

For a new fit, follow the [pipeline guide](../../../PIPELINE.md) with mouse
CellChatDB and `cfg.train.time_prior_source = "rna_maturity"`. The LR CSV used
for the saved fits is not included. Save new outputs separately; use each saved
fit's stored normalization and source metadata for quantitative comparisons.
