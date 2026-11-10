# Human liver regeneration — Visium

Eight human liver sections: three healthy, three APAP-induced acute liver failure,
and two non-APAP acute liver failure (NonAE). Inputs combine 10x Visium expression,
velocyto RNA counts and spot annotations, with 33,538 genes before CellVOCAL QC.

Source: Matchett et al., *Multimodal decoding of human liver regeneration*
([Nature, 2024](https://doi.org/10.1038/s41586-024-07376-2));
[GEO GSE223559](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223559).

## Samples and files

| Condition | Sample ID |
|-----------|-----------|
| APAP | `APAP_liver_spatial_01_A` |
| APAP | `APAP_liver_spatial_01_B` |
| APAP | `APAP_liver_spatial_03` |
| Healthy | `Healthy_liver_spatial_01` |
| Healthy | `Healthy_liver_spatial_02` |
| Healthy | `Healthy_liver_spatial_10` |
| NonAE | `NonAE_liver_spatial_03` |
| NonAE | `NonAE_liver_spatial_05` |

APAP01 A/B are sections from the same donor: eight sections represent seven donors.

- `raw/<sample_id>.h5ad`: prepared AnnData input, including `spliced`/`unspliced`
  layers and `obsm["spatial"]` coordinates.
- `cellvocal/<sample_id>/results/out.h5ad`: fitted CellVOCAL result.
- `cellvocal/<sample_id>/checkpoints/`: saved model.

## Load

Run from the repository root:

```python
import anndata as ad

sample = "APAP_liver_spatial_01_A"
adata = ad.read_h5ad(f"paper/data/human_liver/raw/{sample}.h5ad")
fitted = ad.read_h5ad(f"paper/data/human_liver/cellvocal/{sample}/results/out.h5ad")
```

For a new fit, follow the [pipeline guide](../../../PIPELINE.md) with
`cfg.translator.cellchat_species = "human"`. The LR CSV used for the saved fits
is not included; use CellChatDB or supply an LR prior, and save new outputs separately.
