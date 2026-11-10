# External example data: HuBMAP human kidney Slide-seq

This directory contains the complete public processed RNA count file for human
right kidney cortex, Slide-seq puck `Puck_200501_25`, from HuBMAP dataset
**HBM986.KFWG.239**. The [official processed-dataset metadata](https://entity.api.hubmapconsortium.org/entities/HBM986.KFWG.239)
identifies this dataset as public. Its parent experimental record is
[HBM554.DHLV.735](https://portal.hubmapconsortium.org/browse/dataset/21cfe0f6c3368b06237716a6b48f5825).

`human_kidney_slideseq.h5ad` is the unmodified download of the official
[`expr.h5ad`](https://assets.hubmapconsortium.org/cdf9127efdfc96189845c35244d4d8ab/expr.h5ad),
saved under a descriptive local filename. It contains **35,052 spatial beads
and 60,286 genes**, and is **131,875,528 bytes** (125.77 MiB, approximately
126 MiB). Every original bead, feature, and count is retained in this file;
no sampling, filtering, or numerical transformation has been applied to it.

The original AnnData fields are:

- `layers["spliced"]`, `layers["unspliced"]`, and
  `layers["spliced_unspliced_sum"]`: aligned CSR sparse count matrices.
- `obsm["X_spatial"]`: the provider's measured bead coordinates, with shape
  `(35052, 2)`. Coordinates retain the provider's units; a physical length
  calibration is not asserted.
- `var_names`: versioned Ensembl gene identifiers. `var["hugo_symbol"]`
  contains categorical gene symbols, including missing entries.
- `obs_names`: bead barcodes. The original file has no cell-type annotations.

The [HuBMAP RNA-seq processing documentation](https://docs.hubmapconsortium.org/assays/rnaseq.html)
defines `spliced` and `unspliced` as exonic and intronic RNA counts quantified
with Salmon against a reference containing both sequence classes. Counts can
be fractional when reads map to multiple sequences.

The dimensions above describe the downloaded input. The example applies the
model's standard quality control before fitting; its retained bead and gene
counts are reported by the workflow. Gene-symbol preparation may aggregate
duplicate mapped symbols and use Ensembl identifiers for missing symbols.
These operations apply to the working AnnData object; the downloaded file
remains unchanged.

The complete download and configuration steps are in
[example.ipynb](../example.ipynb). The file was retrieved on **2026-10-08**;
its SHA-256 is
`19bfaeb96e900dc76f39d3d6063e8594c07eae05f89b6340572fbd843ad36d8f`.
Provenance is recorded directly in the notebook; there are no separate JSON files.

HuBMAP licenses these data under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/),
as stated on the [official dataset page](https://portal.hubmapconsortium.org/browse/dataset/21cfe0f6c3368b06237716a6b48f5825).
Please credit HuBMAP and both dataset identifiers, and cite the experimental
record [doi:10.35079/HBM554.DHLV.735](https://doi.org/10.35079/HBM554.DHLV.735)
when reusing the file. The original Slide-seq technology is described by
Rodriques et al., *Science* 363, 1463–1467 (2019),
[doi:10.1126/science.aaw1219](https://doi.org/10.1126/science.aaw1219).
