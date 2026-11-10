"""Count-based condition comparisons of inferred communication events.

The outcome is an *event count*, not a sum of inferred signalling influences.  The default
offset measures each sample's total retained event count, so coefficients are
relative event-rate contrasts.  Biological donors, rather than capture sites
or technical sections, supply covariance clusters.  These observational
comparisons do not estimate causal signalling effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import warnings

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    absent = [column for column in columns if column not in frame]
    if absent:
        raise ValueError(f"{name} is missing columns: {absent}")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} has duplicate column names.")


def _labels(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    groups = frame[list(group_cols)].drop_duplicates().copy()
    if groups.isna().any().any():
        raise ValueError("Communication group identifiers cannot be missing.")
    if groups.empty:
        groups["group_label"] = pd.Series(dtype=str)
        return groups
    groups["group_label"] = groups.astype(str).agg(" | ".join, axis=1)
    if groups.group_label.duplicated().any():
        raise ValueError("Group identifiers produce ambiguous display labels.")
    return groups


def prepare_event_counts(
    events: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("sender", "module"),
    sample_col: str = "sample",
    condition_col: str = "condition",
    donor_col: str = "donor",
    event_influence_col: str = "event_influence",
    require_all_conditions: bool = True,
) -> pd.DataFrame:
    """Count admitted events, retaining every metadata sample including zeros.

    ``events`` contains one row per candidate event and must already be
    restricted to the receiver population of interest.  Positive
    inferred signalling influences define admitted events; zero-influence rows are
    ignored.  ``group_cols`` can identify a module alone or sender class plus
    module.  The default retains groups with at least one event in *each*
    condition.  No receiver state, velocity or contrast direction enters that
    filter.

    The offset denominator is recomputed after group retention.  Samples with
    no retained events remain in the table with zero counts and an undefined
    log offset; :func:`fit_communication_regression` records their exclusion.
    The supplied tables are not modified.
    """
    group_cols = tuple(group_cols)
    if not group_cols or len(set(group_cols)) != len(group_cols):
        raise ValueError("group_cols must contain distinct group column names.")
    id_cols = (sample_col, condition_col, donor_col)
    if len(set(id_cols)) != 3 or set(group_cols).intersection(id_cols):
        raise ValueError("Sample, condition, donor and group columns must be distinct.")
    _require_columns(events, (sample_col, *group_cols, event_influence_col), "events")
    _require_columns(sample_metadata, id_cols, "sample_metadata")
    meta = sample_metadata.copy()
    if meta.empty or meta[list(id_cols)].isna().any().any():
        raise ValueError("sample_metadata requires nonmissing sample, condition and donor values.")
    if meta[sample_col].duplicated().any():
        raise ValueError("sample_metadata must contain exactly one row per sample.")
    reserved = {
        *group_cols, "group_label", "active_event_count", "influence_sum",
        "sample_total_events", "events_per_1000", "offset_log_total_events",
    }
    if reserved.intersection(meta.columns):
        raise ValueError("sample_metadata contains reserved output/group columns.")
    if events[sample_col].isna().any():
        raise ValueError("Event sample identifiers cannot be missing.")
    unknown = ~events[sample_col].isin(meta[sample_col])
    if unknown.any():
        raise ValueError("Every event sample must occur in sample_metadata.")
    event_influence = pd.to_numeric(events[event_influence_col], errors="raise").to_numpy(float)
    if not np.isfinite(event_influence).all() or (event_influence < 0).any():
        raise ValueError("Event influences must be finite and nonnegative.")
    active = events.loc[event_influence > 0, [sample_col, *group_cols]].copy()
    active["_event_influence"] = event_influence[event_influence > 0]
    if active[list(group_cols)].isna().any().any():
        raise ValueError("Communication group identifiers cannot be missing.")
    groups = _labels(active, group_cols)
    if require_all_conditions and not active.empty:
        observed = active.merge(meta[[sample_col, condition_col]], on=sample_col, validate="many_to_one")
        coverage = observed.groupby(list(group_cols), observed=True, dropna=False)[condition_col].nunique()
        retained = coverage[coverage == meta[condition_col].nunique()].reset_index()[list(group_cols)]
        groups = groups.merge(retained, on=list(group_cols), how="inner", validate="one_to_one")
    if groups.empty:
        columns = [
            *meta.columns, *group_cols, "group_label", "active_event_count",
            "influence_sum", "sample_total_events", "events_per_1000",
            "offset_log_total_events",
        ]
        result = pd.DataFrame(columns=columns)
    else:
        observed = active.groupby([sample_col, *group_cols], as_index=False, observed=True).agg(
            active_event_count=("_event_influence", "size"),
            influence_sum=("_event_influence", "sum"),
        )
        result = meta.merge(groups, how="cross").merge(
            observed, on=[sample_col, *group_cols], how="left", validate="one_to_one",
        )
        result["active_event_count"] = result.active_event_count.fillna(0).astype(np.int64)
        result["influence_sum"] = result.influence_sum.fillna(0.0)
        result["sample_total_events"] = result.groupby(sample_col, observed=True).active_event_count.transform("sum")
        exposure = result.sample_total_events.where(result.sample_total_events > 0).astype(float)
        result["events_per_1000"] = 1000 * result.active_event_count / exposure
        result["offset_log_total_events"] = np.log(exposure)
    result.attrs.update(
        group_columns=list(group_cols),
        offset_definition="log(total positive-influence events in retained groups per sample)",
        retention_rule="present in every condition" if require_all_conditions else "all observed groups",
        receiver_filter="caller-supplied event subset",
        n_input_events=len(events),
        n_admitted_events=int((event_influence > 0).sum()),
    )
    return result


def _bh(values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind="stable")]
    if len(order):
        raw = values[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0, 1)
    return adjusted


def fit_communication_regression(
    counts: pd.DataFrame,
    *,
    contrasts: Mapping[str, tuple[str, str]] | Sequence[tuple[str, str]] | None = None,
    reference: str | None = None,
    alpha: float = 1.0,
    group_cols: Sequence[str] = ("sender", "module"),
    sample_col: str = "sample",
    condition_col: str = "condition",
    donor_col: str = "donor",
    count_col: str = "active_event_count",
    offset_col: str = "offset_log_total_events",
    fdr_scope: str = "global",
) -> pd.DataFrame:
    """Fit negative-binomial event rates with donor-clustered uncertainty.

    ``contrasts`` maps a label to ``(numerator, denominator)`` conditions, or
    supplies a sequence of such pairs.  By default each non-reference
    condition is compared with preceding conditions.  ``alpha`` is fixed
    negative-binomial dispersion (1 by default), not an estimated parameter.

    Zero-exposure samples (NaN offsets) are excluded and counted explicitly.
    Missing donor covariance never falls back to unclustered inference:
    point estimates may be retained, but intervals, P values and FDR are NaN.
    A normal approximation and finite-sample cluster correction are used,
    matching statsmodels' GLM cluster covariance.  Interpret very small donor
    cohorts as exploratory even when covariance estimation is available.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    group_cols = tuple(group_cols)
    if not group_cols or len(set(group_cols)) != len(group_cols):
        raise ValueError("group_cols must contain distinct group column names.")
    required = (sample_col, condition_col, donor_col, *group_cols, count_col, offset_col)
    _require_columns(counts, required, "counts")
    if counts.empty:
        raise ValueError("No communication groups remain for regression.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if fdr_scope not in {"global", "contrast"}:
        raise ValueError("fdr_scope must be 'global' or 'contrast'.")
    frame = counts.copy()
    if frame[[sample_col, condition_col, donor_col, *group_cols]].isna().any().any():
        raise ValueError("Sample, donor, condition and group identifiers cannot be missing.")
    if frame.duplicated([sample_col, *group_cols]).any():
        raise ValueError("Each sample must occur once per communication group.")
    consistency = frame.groupby(sample_col, observed=True)[[condition_col, donor_col]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("Sample metadata must be consistent across communication groups.")
    y_all = pd.to_numeric(frame[count_col], errors="raise").to_numpy(float)
    if not np.isfinite(y_all).all() or (y_all < 0).any() or not np.equal(y_all, np.floor(y_all)).all():
        raise ValueError("Event counts must be finite nonnegative integers.")
    offset_all = pd.to_numeric(frame[offset_col], errors="raise").to_numpy(float)
    if np.isinf(offset_all).any():
        raise ValueError("Offsets must be finite or NaN for unavailable exposure.")
    if np.any(np.isnan(offset_all) & (y_all > 0)):
        raise ValueError("Positive event counts require a defined exposure offset.")
    frame[count_col], frame[offset_col] = y_all, offset_all
    labels = _labels(frame, group_cols)
    if "group_label" in frame:
        frame = frame.drop(columns="group_label")
    frame = frame.merge(labels, on=list(group_cols), how="left", validate="many_to_one")
    conditions = list(pd.unique(frame[condition_col]))
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required.")
    reference = conditions[0] if reference is None else reference
    if reference not in conditions:
        raise ValueError(f"Reference condition {reference!r} is absent.")
    conditions = [reference, *(value for value in conditions if value != reference)]
    if contrasts is None:
        pairs = {f"{num}_vs_{den}": (num, den) for den, num in combinations(conditions, 2)}
    elif isinstance(contrasts, Mapping):
        pairs = dict(contrasts)
    else:
        if any(isinstance(pair, (str, bytes)) or len(pair) != 2 for pair in contrasts):
            raise ValueError("Contrasts must contain pairs of condition identifiers.")
        pairs = {f"{pair[0]}_vs_{pair[1]}": tuple(pair) for pair in contrasts}
        if len(pairs) != len(contrasts):
            raise ValueError("Contrasts must contain unique pairs of conditions.")
    if not pairs:
        raise ValueError("At least one contrast is required.")
    for label, pair in pairs.items():
        if isinstance(pair, (str, bytes)) or len(pair) != 2 or pair[0] == pair[1] or any(value not in conditions for value in pair):
            raise ValueError(f"Invalid condition pair for contrast {label!r}: {pair!r}")
    param_index = {condition: i for i, condition in enumerate(conditions)}
    rows = []
    all_samples = set(frame[sample_col])
    for label, group in frame.groupby("group_label", sort=False, observed=True):
        if set(group[sample_col]) != all_samples:
            raise ValueError("Counts must be zero-filled across all samples for every group.")
        used = group.loc[np.isfinite(group[offset_col])].copy()
        n_params, n_donors = len(conditions), used[donor_col].nunique()
        x = np.column_stack((
            np.ones(len(used)),
            *(used[condition_col].eq(condition).to_numpy(float) for condition in conditions[1:]),
        ))
        parameters = np.full(n_params, np.nan)
        covariance = np.full((n_params, n_params), np.nan)
        status, reason = "unavailable", ""
        notes = []
        if len(used) <= n_params or np.linalg.matrix_rank(x) < n_params:
            reason = "insufficient independent rows or missing condition after exposure filtering"
        elif used[count_col].sum() <= 0:
            reason = "all event counts are zero"
        else:
            model = sm.GLM(
                used[count_col].to_numpy(float), x,
                family=sm.families.NegativeBinomial(alpha=alpha),
                offset=used[offset_col].to_numpy(float),
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    point_fit = model.fit()
                notes.extend(str(w.message) for w in caught)
                if not point_fit.converged or not np.isfinite(point_fit.params).all():
                    raise FloatingPointError("Negative-binomial fitting did not converge.")
                parameters = np.asarray(point_fit.params, float)
                status = "point_estimate_only"
                if n_donors < 2:
                    reason = "at least two biological donor clusters are required"
                else:
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            fit = model.fit(
                                cov_type="cluster",
                                cov_kwds={"groups": used[donor_col].to_numpy(), "use_correction": True},
                            )
                        notes.extend(str(w.message) for w in caught)
                        covariance = np.asarray(fit.cov_params(), float)
                        if not fit.converged or not np.isfinite(covariance).all():
                            raise FloatingPointError("Nonfinite or unconverged donor-cluster covariance.")
                        parameters = np.asarray(fit.params, float)
                        status = "donor_clustered"
                    except Exception as error:
                        covariance[:] = np.nan
                        reason = f"donor-cluster covariance unavailable: {type(error).__name__}: {error}"
            except Exception as error:
                reason = f"model fit unavailable: {type(error).__name__}: {error}"
        for contrast, (numerator, denominator) in pairs.items():
            vector = np.zeros(n_params)
            if numerator != reference:
                vector[param_index[numerator]] += 1
            if denominator != reference:
                vector[param_index[denominator]] -= 1
            beta = float(vector @ parameters)
            variance = float(vector @ covariance @ vector)
            se = np.sqrt(variance) if np.isfinite(variance) and variance > 0 else np.nan
            pvalue = float(2 * norm.sf(abs(beta / se))) if np.isfinite(se) else np.nan
            rows.append({
                **{column: group[column].iloc[0] for column in group_cols},
                "group_label": label, "contrast": contrast,
                "numerator": numerator, "denominator": denominator,
                "contrast_label": f"{numerator} vs {denominator}",
                "log2_rate_ratio": beta / np.log(2), "se_log2": se / np.log(2),
                "ci_low": (beta - norm.ppf(.975) * se) / np.log(2),
                "ci_high": (beta + norm.ppf(.975) * se) / np.log(2),
                "pvalue": pvalue, "alpha": float(alpha),
                "total_events": int(used[count_col].sum()),
                "n_samples": len(used), "n_donors": n_donors,
                "n_excluded_samples": len(group) - len(used),
                "inference_status": status, "inference_note": reason,
                "fit_warnings": " | ".join(dict.fromkeys(notes)),
                "offset_column": offset_col,
                "model": "negative binomial; donor-clustered covariance",
            })
    result = pd.DataFrame(rows)
    result["fdr"] = np.nan
    if fdr_scope == "global":
        result["fdr"] = _bh(result.pvalue.to_numpy(float))
    else:
        for indices in result.groupby("contrast", sort=False).groups.values():
            result.loc[indices, "fdr"] = _bh(result.loc[indices, "pvalue"].to_numpy(float))
    result["fdr_scope"] = fdr_scope
    result.attrs.update(
        interpretation="exploratory relative event-rate association, not a causal signalling effect",
        covariance="donor-clustered sandwich covariance with finite-sample correction",
        dispersion="fixed alpha", reference_condition=reference,
    )
    return result
