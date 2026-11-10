# Model Card: CellVOCAL

## 1. Model details

CellVOCAL is a graph-based research framework for estimating RNA velocity,
latent developmental time, and spatially resolved candidate cell–cell
communication (CCC) event influence. It combines spliced and unspliced RNA,
spatial proximity, ligand–receptor (LR) evidence, and optional cell annotations.
The model is fitted for each input dataset; the project does not provide a
universal pretrained CellVOCAL checkpoint.

The optional event translator is local and LLM-free. It may be disabled,
trained on candidate events for the dataset, or loaded from a compatible local
checkpoint. It does not send data to a remote language model.

| Field | Value |
| --- | --- |
| Model | CellVOCAL |
| Domain | Spatial transcriptomics and RNA velocity |
| Framework | PyTorch and PyTorch Geometric |
| Original upstream source | <https://github.com/shuai-wang-edu/CellVOCAL-dup> |
| License | GNU General Public License v3.0 |

## 2. Intended use

CellVOCAL supports exploratory research on velocity direction, latent-time
ordering, intrinsic and CCC-associated velocity components, and candidate
sender–receiver–LR events. It may be used to compare methods on controlled
simulations and appropriately prepared tissue sections. Its outputs are
computational hypotheses that require independent biological validation.

It is outside scope for clinical diagnosis, prognosis, treatment selection,
absolute chronological dating without an independently justified prior, or
claims of physical interaction or causality from an event influence estimate alone.
Checkpoints should not be transferred to datasets with incompatible genes,
preprocessing, LR modules, annotations, or configuration.

## 3. Input scope

The intended input is a spatial `AnnData` slice with reliable spliced and
unspliced measurements, finite coordinates, and gene identifiers that can be
matched to an LR prior. Optional cell annotations and observed-time priors
depend on the chosen configuration. The exact fields, shapes, filtering rules,
and public API are specified in [PIPELINE.md](PIPELINE.md).

The project includes direction, event-attribution, and differential synthetic
panels, plus mouse embryo, mouse melanoma, human liver, mouse brain cortex,
mouse lung, and mouse gut data documentation and reference notebooks under
`paper/`. These are examples of the supported analysis context, not evidence
of universal performance.

## 4. Output interpretation

CellVOCAL returns total RNA velocity, intrinsic and CCC-associated components,
normalized latent time, a learned latent state, and candidate-event scores.
For receiver i and module m, `module_activation_prob` is q. Conditional on
that receiver and module, `sender_share` is p and sums to one over eligible
senders. `event_influence = q * p` is inferred signalling influence; it is
nonnegative and is not a calibrated probability that a biological interaction
occurred. Multiplying a by a signed module gene-effect coefficient produces
a gene contribution to RNA velocity. This describes transcriptional modulation
of receiver dynamics. Screened ligand–receptor modules are candidate regulatory
signals, rather than established causal regulators. The exact
`AnnData` keys and generated files are documented in [PIPELINE.md](PIPELINE.md).

## 5. Learning and evaluation

The model trains on each supplied dataset with an objective combining velocity,
time-prior, and CCC terms. A configurable prior can come from spatial
coordinates, RNA maturity, or an observation column. The optional translator
is pretrained before the main model and then frozen. Configuration and
algorithmic hand-offs are in [PIPELINE.md](PIPELINE.md).

The project provides simulation panels, dataset documentation, reference
notebooks, and saved CellVOCAL results. It does not claim one global
performance value or a validated generalization bound. Evaluation should
include appropriate direction or time checks, simulated event recovery where
ground truth exists,
sensitivity to seeds and priors, biological replicates, negative controls, and
orthogonal assays where available.

For reproducibility, retain the input-data version, preprocessing record, full
configuration, LR database and species, random seed, environment, source
snapshot or available upstream revision, and checkpoints together.

## 6. Limitations and risks

- Spatial proximity and LR expression do not establish secretion, receptor
  activation, or a causal effect on the receiver.
- Time orientation depends on the selected prior. The default spatial prior
  uses the first coordinate axis and may be inappropriate for a dataset.
- Low unspliced coverage, ambient RNA, batch effects, and departures from RNA
  kinetic assumptions can bias velocity and event estimates.
- LR databases have incomplete species and gene coverage, and optional cell
  annotations can contain transfer errors.
- Quality-control filtering changes the cells, genes, graph, and eligible LR
  events. Inspect the retained data before interpreting a result.
- Scores do not provide calibrated uncertainty intervals; checkpoint transfer
  and generalization are not guaranteed.
- Dense preprocessing paths and many candidate events can require substantial
  memory. Hardware, CUDA kernels, dependencies, and data order can affect exact
  reproducibility.

## 7. Software and governance

The supplied [environment.yml](environment.yml) targets Python 3.11 and lists
the core scientific Python dependencies. GPU training is the default;
`cfg.train.device = "cpu"` is available when a GPU is unavailable.

The core pipeline uses local data and a local LR resource. Researchers remain
responsible for access controls, consent, de-identification, retention, and
sharing rules for human-derived data and generated artifacts.

## 8. Availability and citation

Source code is distributed under the [GNU GPL v3.0](LICENSE). For source
provenance and upstream issues, see the
[original upstream repository](https://github.com/shuai-wang-edu/CellVOCAL-dup)
and its [issue tracker](https://github.com/shuai-wang-edu/CellVOCAL-dup/issues).
Until a formal publication citation is available, cite that source and identify
the exact source snapshot or available release, tag, or commit used, including
any local changes.

This card describes the implementation distributed with this file. Update it
when intended use, data requirements, evaluation claims, or model behavior
change.
