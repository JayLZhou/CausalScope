"""Reference randomization-aware pattern-tree search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from causalscope.pattern import RootedPattern
from causalscope.statistics import linear_statistic, max_t_adjusted_p, subtree_envelope

ExposureProvider = Callable[[RootedPattern, tuple[int, ...]], tuple[bool, ...]]


@dataclass(frozen=True)
class PatternFamily:
    patterns: dict[str, RootedPattern]
    children: dict[str, tuple[str, ...]]
    roots: tuple[str, ...]

    def __post_init__(self) -> None:
        missing_roots = set(self.roots) - self.patterns.keys()
        if missing_roots:
            raise ValueError(f"unknown roots: {sorted(missing_roots)}")
        for parent, child_names in self.children.items():
            if parent not in self.patterns:
                raise ValueError(f"unknown parent: {parent}")
            missing = set(child_names) - self.patterns.keys()
            if missing:
                raise ValueError(f"unknown children of {parent}: {sorted(missing)}")
            for child in child_names:
                if not self.patterns[child].is_monotone_extension_of(self.patterns[parent]):
                    raise ValueError(
                        f"{child} is not a monotone extension of {parent}"
                    )

        parent_counts = {name: 0 for name in self.patterns}
        for child_names in self.children.values():
            for child in child_names:
                parent_counts[child] += 1
        repeated = sorted(name for name, count in parent_counts.items() if count > 1)
        if repeated:
            raise ValueError(f"patterns must have a unique parent: {repeated}")

        reachable: set[str] = set()
        active: set[str] = set()

        def validate_tree(name: str) -> None:
            if name in active:
                raise ValueError("pattern family contains a cycle")
            if name in reachable:
                return
            active.add(name)
            for child in self.children.get(name, ()):
                validate_tree(child)
            active.remove(name)
            reachable.add(name)

        for root in self.roots:
            validate_tree(root)
        unreachable = sorted(self.patterns.keys() - reachable)
        if unreachable:
            raise ValueError(f"patterns are unreachable from roots: {unreachable}")

    def depth_first_names(self) -> tuple[str, ...]:
        names: list[str] = []

        def visit(name: str) -> None:
            names.append(name)
            for child in self.children.get(name, ()):
                visit(child)

        for root in self.roots:
            visit(root)
        return tuple(names)


@dataclass(frozen=True)
class MaxSearchResult:
    maxima: tuple[float, ...]
    visited_patterns: tuple[str, ...]
    pruned_roots: tuple[str, ...]
    statistic_evaluations: int


@dataclass(frozen=True)
class DiscoveryResult:
    adjusted_p_values: dict[str, float]
    significant_patterns: tuple[str, ...]
    visited_patterns: tuple[str, ...]
    pruned_roots: tuple[str, ...]


def brute_force_randomization_maxima(
    family: PatternFamily,
    assignments: tuple[tuple[int, ...], ...],
    residuals: tuple[float, ...],
    exposure_provider: ExposureProvider,
) -> tuple[float, ...]:
    maxima = [0.0] * len(assignments)
    for name in family.depth_first_names():
        pattern = family.patterns[name]
        for index, assignment in enumerate(assignments):
            exposure = exposure_provider(pattern, assignment)
            maxima[index] = max(maxima[index], linear_statistic(residuals, exposure))
    return tuple(maxima)


def randomization_max_search(
    family: PatternFamily,
    assignments: tuple[tuple[int, ...], ...],
    residuals: tuple[float, ...],
    exposure_provider: ExposureProvider,
) -> MaxSearchResult:
    """Compute exact max statistics with assignment-wise envelope pruning."""

    maxima = [0.0] * len(assignments)
    visited: list[str] = []
    pruned: list[str] = []
    evaluations = 0

    def visit(name: str) -> None:
        nonlocal evaluations
        pattern = family.patterns[name]
        exposures = tuple(
            exposure_provider(pattern, assignment)
            for assignment in assignments
        )
        envelopes = tuple(
            subtree_envelope(residuals, exposure)
            for exposure in exposures
        )
        active = tuple(
            index
            for index, envelope in enumerate(envelopes)
            if envelope > maxima[index]
        )
        if not active:
            pruned.append(name)
            return

        visited.append(name)
        for index in active:
            statistic = linear_statistic(residuals, exposures[index])
            evaluations += 1
            maxima[index] = max(maxima[index], statistic)

        for child in family.children.get(name, ()):
            visit(child)

    for root in family.roots:
        visit(root)

    return MaxSearchResult(tuple(maxima), tuple(visited), tuple(pruned), evaluations)


def brute_force_adjusted_p_values(
    family: PatternFamily,
    observed_assignment: tuple[int, ...],
    residuals: tuple[float, ...],
    exposure_provider: ExposureProvider,
    randomization_maxima: tuple[float, ...],
) -> dict[str, float]:
    return {
        name: max_t_adjusted_p(
            linear_statistic(
                residuals,
                exposure_provider(family.patterns[name], observed_assignment),
            ),
            randomization_maxima,
        )
        for name in family.depth_first_names()
    }


def discover_significant_patterns(
    family: PatternFamily,
    observed_assignment: tuple[int, ...],
    residuals: tuple[float, ...],
    exposure_provider: ExposureProvider,
    randomization_maxima: tuple[float, ...],
    *,
    alpha: float,
) -> DiscoveryResult:
    """Find adjusted-significant patterns using a subtree p-value lower bound."""

    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    adjusted: dict[str, float] = {}
    significant: list[str] = []
    visited: list[str] = []
    pruned: list[str] = []

    def visit(name: str) -> None:
        pattern = family.patterns[name]
        exposure = exposure_provider(pattern, observed_assignment)
        envelope = subtree_envelope(residuals, exposure)
        subtree_p_lower = max_t_adjusted_p(envelope, randomization_maxima)
        if subtree_p_lower > alpha:
            pruned.append(name)
            return

        visited.append(name)
        statistic = linear_statistic(residuals, exposure)
        p_value = max_t_adjusted_p(statistic, randomization_maxima)
        adjusted[name] = p_value
        if p_value <= alpha:
            significant.append(name)

        for child in family.children.get(name, ()):
            visit(child)

    for root in family.roots:
        visit(root)

    return DiscoveryResult(
        adjusted_p_values=adjusted,
        significant_patterns=tuple(significant),
        visited_patterns=tuple(visited),
        pruned_roots=tuple(pruned),
    )
