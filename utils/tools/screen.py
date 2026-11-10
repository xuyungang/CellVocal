"""Screen ligand–receptor modules as candidate regulatory signals for a supplied outcome.

These utilities use receiver-level incoming influences, rather than
co-expression scores. They deliberately do not attach nominal site-wise
p-values or designate a winning module. The caller chooses the RNA readout,
its direction and its scaling before inspecting the screen.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _aligned_vector(values, index: pd.Index, name: str) -> np.ndarray:
    """Align labelled vectors without silently pairing different receivers."""
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            if not index.is_unique or not values.index.is_unique:
                raise ValueError(f"{name} must have the same ordered receiver index")
            if len(values) != len(index) or not index.isin(values.index).all():
                raise ValueError(f"{name} must contain exactly the incoming receiver index")
            values = values.reindex(index)
        values = values.to_numpy()
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError(f"{name} must have one value per incoming row")
    return values


def screen_communication_modules(
    incoming: pd.DataFrame,
    outcome: Sequence[float] | pd.Series,
    *,
    sections: Sequence | pd.Series | None = None,
    receiver_mask: Sequence[bool] | pd.Series | None = None,
    high_quantile: float = 0.75,
    min_active: int = 4,
    standardize_outcome: bool = False,
    missing: Literal["raise", "omit"] = "raise",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return section-level statistics and an equally weighted module screen.

    Parameters
    ----------
    incoming
        Rows are capture sites or cells; columns are uniquely named modules.
        Entries are summed, non-negative incoming influences.
    outcome
        One explicitly defined signed readout per row. For example, the
        negative of a cycling-programme velocity represents cycling decline.
        This function never changes the sign or chooses a velocity component.
        A Series is aligned by receiver identifiers.
    sections
        Section labels. Each section contributes one correlation and one
        high-input contrast to the summary, irrespective of its size.
    receiver_mask
        Optional Boolean selector applied before computing any statistics.
    high_quantile
        Threshold quantile among strictly positive inputs, within each
        section and module. High input includes ties at the threshold.
        Background contains every remaining eligible receiver, including
        positive-input receivers below the threshold.
    min_active
        Minimum positive-input observations needed to form the high group.
    standardize_outcome
        If True, centre and divide the outcome by its population standard
        deviation separately within each selected section. A constant
        outcome becomes zero but still has an undefined correlation.
        Leave False for an already input-standardised RNA-velocity programme.
    missing
        ``"raise"`` rejects non-finite inputs and outcomes in selected rows.
        ``"omit"`` uses finite pairs per module and records omitted counts.
        Outcome standardisation uses the section's finite outcomes, before
        module-specific pair omission. Inferred signalling influence totals include all
        finite inputs in the selected cohort, even if an outcome is missing.

    Returns
    -------
    per_section, summary
        ``per_section`` retains sample sizes, thresholds, group means and
        separate status fields for undefined statistics. ``summary`` includes
        ``mean_rho``, ``mean_high_delta`` and ``total_influence``. A mean
        is NaN if any section lacks that statistic; valid-section counts
        distinguish missing evidence from a measured zero. No significance
        tests or candidate-selection thresholds are imposed.
    """
    if not isinstance(incoming, pd.DataFrame):
        raise TypeError("incoming must be a receiver-by-module DataFrame")
    if incoming.empty or not incoming.columns.is_unique:
        raise ValueError("incoming must have rows and uniquely named module columns")
    if incoming.columns.isna().any():
        raise ValueError("Module labels cannot be missing")
    if not 0 <= high_quantile <= 1:
        raise ValueError("high_quantile must lie between 0 and 1")
    if isinstance(min_active, bool) or int(min_active) != min_active or min_active < 1:
        raise ValueError("min_active must be a positive integer")
    if missing not in ("raise", "omit"):
        raise ValueError("missing must be 'raise' or 'omit'")

    index = incoming.index
    q = incoming.to_numpy(dtype=float)
    y = _aligned_vector(outcome, index, "outcome").astype(float)
    section = (
        np.full(len(index), "all", dtype=object)
        if sections is None
        else _aligned_vector(sections, index, "sections")
    )
    mask = np.ones(len(index), dtype=bool)
    if receiver_mask is not None:
        mask_values = _aligned_vector(receiver_mask, index, "receiver_mask")
        if pd.isna(mask_values).any() or not all(
            isinstance(value, (bool, np.bool_)) for value in mask_values
        ):
            raise ValueError("receiver_mask must contain only Boolean values")
        mask = mask_values.astype(bool)
    if not mask.any():
        raise ValueError("receiver_mask selects no receivers")
    q, y, section = q[mask], y[mask], section[mask]
    if pd.isna(section).any():
        raise ValueError("Every selected receiver must have a section label")
    if (q[np.isfinite(q)] < 0).any():
        raise ValueError("Incoming inferred signalling influences must be non-negative")
    if missing == "raise" and (
        not np.isfinite(q).all() or not np.isfinite(y).all()
    ):
        raise ValueError("Selected inputs and outcomes must be finite; use missing='omit' explicitly")

    rows = []
    for section_name in pd.unique(section):
        use = section == section_name
        section_q, section_y = q[use], y[use].copy()
        outcome_ok = np.isfinite(section_y)
        if standardize_outcome and outcome_ok.any():
            observed = section_y[outcome_ok]
            sd = float(observed.std(ddof=0))
            section_y[outcome_ok] = (
                (observed - observed.mean()) / sd if sd > 0 else 0.0
            )
        for j, module in enumerate(incoming.columns):
            input_ok = np.isfinite(section_q[:, j])
            paired = input_ok & outcome_ok
            x, z = section_q[paired, j], section_y[paired]
            rho = np.nan
            rho_status = "ok"
            if len(x) < 2:
                rho_status = "insufficient_observations"
            elif np.ptp(x) == 0:
                rho_status = "constant_input"
            elif np.ptp(z) == 0:
                rho_status = "constant_outcome"
            else:
                rank_x, rank_y = rankdata(x), rankdata(z)
                rank_x -= rank_x.mean()
                rank_y -= rank_y.mean()
                rho = float(
                    np.dot(rank_x, rank_y)
                    / (np.linalg.norm(rank_x) * np.linalg.norm(rank_y))
                )

            active = x > 0
            threshold = np.nan
            high = np.zeros(len(x), dtype=bool)
            contrast_status = "ok"
            if active.sum() < min_active:
                contrast_status = "insufficient_active"
            else:
                threshold = float(np.quantile(x[active], high_quantile))
                high = active & (x >= threshold)
                if high.all():
                    contrast_status = "empty_background"
            background = ~high
            high_mean = float(z[high].mean()) if high.any() else np.nan
            background_mean = float(z[background].mean()) if background.any() else np.nan
            rows.append(
                {
                    "section": section_name,
                    "module": module,
                    "n_receivers": int(len(section_y)),
                    "n_observations": int(paired.sum()),
                    "n_omitted": int((~paired).sum()),
                    "n_active_receivers": int(active.sum()),
                    "total_influence": (
                        float(section_q[input_ok, j].sum()) if input_ok.any() else np.nan
                    ),
                    "rho": rho,
                    "rho_status": rho_status,
                    "high_quantile": float(high_quantile),
                    "high_threshold": threshold,
                    "n_high": int(high.sum()),
                    "n_background": int(background.sum()),
                    "high_mean": high_mean,
                    "background_mean": background_mean,
                    "high_delta": high_mean - background_mean,
                    "contrast_status": contrast_status,
                }
            )
    per_section = pd.DataFrame(rows)
    summary_rows = []
    for module, group in per_section.groupby("module", sort=False):
        rho_ok = np.isfinite(group["rho"])
        contrast_ok = np.isfinite(group["high_delta"])
        summary_rows.append(
            {
                "module": module,
                "n_sections": len(group),
                "n_valid_rho_sections": int(rho_ok.sum()),
                "n_valid_contrast_sections": int(contrast_ok.sum()),
                "mean_rho": float(group.rho.mean()) if rho_ok.all() else np.nan,
                "mean_high_delta": (
                    float(group.high_delta.mean()) if contrast_ok.all() else np.nan
                ),
                "total_influence": group.total_influence.sum(min_count=len(group)),
                "n_receivers": int(group.n_receivers.sum()),
                "n_observations": int(group.n_observations.sum()),
                "min_active_receivers": int(group.n_active_receivers.min()),
                "positive_rho_sections": int((group.rho > 0).sum()),
                "positive_contrast_sections": int((group.high_delta > 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    definitions = {
        "analysis": "descriptive communication-module screen; not a significance test",
        "section_weighting": "equal; incomplete section statistics remain undefined",
        "background": "all eligible receivers outside the high-input group",
        "outcome_standardized_within_section": bool(standardize_outcome),
        "influence_cohort": "all selected receivers with finite input",
    }
    per_section.attrs.update(definitions)
    summary.attrs.update(definitions)
    return per_section, summary
