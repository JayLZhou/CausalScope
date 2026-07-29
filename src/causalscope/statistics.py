"""Randomization statistics and subtree envelopes."""

from __future__ import annotations


def center_outcomes_within_treatment(
    outcomes: tuple[float, ...],
    observed_assignment: tuple[int, ...],
    focal_units: tuple[int, ...],
) -> tuple[float, ...]:
    strata: dict[int, list[float]] = {0: [], 1: []}
    for unit in focal_units:
        strata[observed_assignment[unit]].append(outcomes[unit])
    means = {
        treatment: (sum(values) / len(values) if values else 0.0)
        for treatment, values in strata.items()
    }
    return tuple(
        outcomes[unit] - means[observed_assignment[unit]]
        for unit in focal_units
    )


def linear_statistic(
    residuals: tuple[float, ...],
    exposure: tuple[bool, ...],
) -> float:
    if len(residuals) != len(exposure):
        raise ValueError("residual and exposure vectors must have equal length")
    return abs(sum(value for value, is_exposed in zip(residuals, exposure) if is_exposed))


def subtree_envelope(
    residuals: tuple[float, ...],
    parent_exposure: tuple[bool, ...],
) -> float:
    """Bound every binary descendant whose exposed set is a subset of the parent."""

    if len(residuals) != len(parent_exposure):
        raise ValueError("residual and exposure vectors must have equal length")
    positive = sum(
        value
        for value, is_exposed in zip(residuals, parent_exposure)
        if is_exposed and value > 0.0
    )
    negative = -sum(
        value
        for value, is_exposed in zip(residuals, parent_exposure)
        if is_exposed and value < 0.0
    )
    return max(positive, negative)


def max_t_adjusted_p(
    observed_statistic: float,
    randomization_maxima: tuple[float, ...],
) -> float:
    exceedances = sum(
        maximum >= observed_statistic
        for maximum in randomization_maxima
    )
    return (1 + exceedances) / (len(randomization_maxima) + 1)

