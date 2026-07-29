from __future__ import annotations

from causalscope.graph import PropertyGraph
from causalscope.matching import exposure_vector
from causalscope.pattern import RootedPattern
from causalscope.randomization import BernoulliDesign
from causalscope.search import (
    PatternFamily,
    brute_force_adjusted_p_values,
    brute_force_randomization_maxima,
    discover_significant_patterns,
    randomization_max_search,
)


def test_pruned_randomization_maxima_equal_brute_force(
    demo_graph: PropertyGraph,
    demo_family: PatternFamily,
) -> None:
    focal = (0, 2, 3, 5, 6, 7)
    observed = (0, 1, 1, 0, 1, 0, 0, 1)
    assignments = BernoulliDesign.constant(len(demo_graph.node_ids), 0.5).conditional_samples(
        observed,
        focal,
        draws=127,
        seed=11,
    )
    residuals = (1.8, -1.1, 0.6, -1.7, 0.9, -0.5)

    def exposures(pattern: RootedPattern, assignment: tuple[int, ...]) -> tuple[bool, ...]:
        return exposure_vector(demo_graph, pattern, focal, assignment)

    brute = brute_force_randomization_maxima(
        demo_family,
        assignments,
        residuals,
        exposures,
    )
    result = randomization_max_search(
        demo_family,
        assignments,
        residuals,
        exposures,
    )

    assert result.maxima == brute
    assert "impossible_purchase" in result.pruned_roots
    assert result.statistic_evaluations < len(assignments) * len(demo_family.patterns)


def test_adjusted_p_pruning_preserves_significant_pattern_set(
    demo_graph: PropertyGraph,
    demo_family: PatternFamily,
) -> None:
    focal = (0, 2, 3, 5, 6, 7)
    observed = (0, 1, 1, 0, 1, 0, 0, 1)
    assignments = BernoulliDesign.constant(len(demo_graph.node_ids), 0.5).conditional_samples(
        observed,
        focal,
        draws=127,
        seed=19,
    )
    residuals = (1.8, -1.1, 0.6, -1.7, 0.9, -0.5)

    def exposures(pattern: RootedPattern, assignment: tuple[int, ...]) -> tuple[bool, ...]:
        return exposure_vector(demo_graph, pattern, focal, assignment)

    maxima = brute_force_randomization_maxima(
        demo_family,
        assignments,
        residuals,
        exposures,
    )
    brute_p = brute_force_adjusted_p_values(
        demo_family,
        observed,
        residuals,
        exposures,
        maxima,
    )
    alpha = 0.25
    brute_significant = {
        name
        for name, p_value in brute_p.items()
        if p_value <= alpha
    }
    result = discover_significant_patterns(
        demo_family,
        observed,
        residuals,
        exposures,
        maxima,
        alpha=alpha,
    )

    assert set(result.significant_patterns) == brute_significant
    assert result.pruned_roots
    assert len(result.visited_patterns) < len(demo_family.patterns)
