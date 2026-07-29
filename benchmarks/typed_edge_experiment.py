"""Representation comparison against fixed and hand-specified motif sets.

The fixed baseline receives the complete one-hot basis for the number of
treated neighbors in an untyped open triad. CausalScope discovers edge-typed
one-hop patterns from the property graph schema, including any causally
irrelevant decoy edge types. The hand-specified control is given the two true
typed motifs. The data-generating process assigns opposite spillover effects
to FRIEND and WORKS_WITH neighbors, making the conditional mean zero for every
untyped treatment-count category.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from causalscope.generation import generate_one_hop_treated_patterns
from causalscope.graph import PropertyGraph
from causalscope.pattern import RootedPattern
from causalscope.randomization import BernoulliDesign
from causalscope.search import (
    brute_force_adjusted_p_values,
    brute_force_randomization_maxima,
    randomization_max_search,
)
from causalscope.statistics import (
    center_outcomes_within_treatment,
    linear_statistic,
    max_t_adjusted_p,
)

FRIEND_PATTERN = "User-[FRIEND]->User:z=1"
WORK_PATTERN = "User-[WORKS_WITH]->User:z=1"
ROOT_PATTERN = "root:User"


@dataclass(frozen=True)
class TrialResult:
    causalscope_any: bool
    causalscope_both: bool
    causalscope_report_any: bool
    causalscope_decoy_any: bool
    fixed_motifs_any: bool
    specified_typed_any: bool
    causalscope_objective: float
    specified_typed_objective: float
    fixed_untyped_objective: float
    causalscope_true_maximizer: bool
    exact_observed_maximum: bool
    pruning_fraction: float


@dataclass(frozen=True)
class MethodSummary:
    rejection_rate: float
    standard_error: float


@dataclass(frozen=True)
class ContinuousSummary:
    mean: float
    standard_error: float


def build_typed_star_transactions(
    focal_count: int,
    decoy_relations: int = 0,
) -> tuple[PropertyGraph, tuple[int, ...], dict[str, tuple[int, ...]]]:
    """Give every focal one alter of each true and decoy relation type."""

    if focal_count <= 0:
        raise ValueError("focal_count must be positive")
    if decoy_relations < 0:
        raise ValueError("decoy_relations must be nonnegative")

    graph = PropertyGraph()
    relation_labels = (
        "FRIEND",
        "WORKS_WITH",
        *(f"DECOY_{index:02d}" for index in range(decoy_relations)),
    )
    nodes_per_focal = 1 + len(relation_labels)
    total_nodes = nodes_per_focal * focal_count
    for node_id in range(total_nodes):
        graph.add_node(node_id, "User")

    focal = tuple(range(focal_count))
    alters_by_relation: dict[str, list[int]] = {
        label: [] for label in relation_labels
    }
    for unit in focal:
        first_alter = focal_count + len(relation_labels) * unit
        for offset, label in enumerate(relation_labels):
            alter = first_alter + offset
            alters_by_relation[label].append(alter)
            graph.add_edge(unit, alter, label)
    return (
        graph,
        focal,
        {
            label: tuple(alters)
            for label, alters in alters_by_relation.items()
        },
    )


def count_feature_vectors(
    assignment: tuple[int, ...],
    alters_by_relation: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[bool, ...], ...]:
    alters_by_focal = zip(*alters_by_relation.values())
    counts = tuple(
        sum(assignment[alter] for alter in alters)
        for alters in alters_by_focal
    )
    neighbor_count = len(alters_by_relation)
    return tuple(
        tuple(count == target for count in counts)
        for target in range(neighbor_count + 1)
    )


def typed_feature_vectors(
    assignment: tuple[int, ...],
    alters_by_relation: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    return (
        tuple(
            bool(assignment[node])
            for node in alters_by_relation["FRIEND"]
        ),
        tuple(
            bool(assignment[node])
            for node in alters_by_relation["WORKS_WITH"]
        ),
    )


def feature_family_maxima(
    assignments: tuple[tuple[int, ...], ...],
    residuals: tuple[float, ...],
    feature_provider: Callable[
        [tuple[int, ...]],
        tuple[tuple[bool, ...], ...],
    ],
) -> tuple[float, ...]:
    return tuple(
        max(
            linear_statistic(residuals, feature)
            for feature in feature_provider(assignment)
        )
        for assignment in assignments
    )


def any_adjusted_rejection(
    observed_features: tuple[tuple[bool, ...], ...],
    residuals: tuple[float, ...],
    maxima: tuple[float, ...],
    alpha: float,
) -> bool:
    return any(
        max_t_adjusted_p(linear_statistic(residuals, feature), maxima) <= alpha
        for feature in observed_features
    )


def run_trial(
    *,
    seed: int,
    focal_count: int,
    randomizations: int,
    alpha: float,
    spillover: float,
    direct_effect: float,
    noise_sd: float,
    decoy_relations: int = 0,
) -> TrialResult:
    graph, focal, alters_by_relation = build_typed_star_transactions(
        focal_count,
        decoy_relations,
    )
    family = generate_one_hop_treated_patterns(graph, root_label="User")
    design = BernoulliDesign.constant(len(graph.node_ids), 0.5)
    rng = random.Random(seed)
    observed = design.sample(rng)

    outcomes = [0.0] * len(graph.node_ids)
    for unit, friend, coworker in zip(
        focal,
        alters_by_relation["FRIEND"],
        alters_by_relation["WORKS_WITH"],
    ):
        typed_spillover = spillover * (
            observed[coworker] - observed[friend]
        )
        outcomes[unit] = (
            direct_effect * observed[unit]
            + typed_spillover
            + rng.gauss(0.0, noise_sd)
        )
    residuals = center_outcomes_within_treatment(
        tuple(outcomes),
        observed,
        focal,
    )
    assignments = design.conditional_samples(
        observed,
        focal,
        draws=randomizations,
        seed=seed + 1_000_003,
    )

    def pattern_exposures(
        pattern: RootedPattern,
        assignment: tuple[int, ...],
    ) -> tuple[bool, ...]:
        if pattern.name == ROOT_PATTERN:
            return (True,) * len(focal)
        edge_label = pattern.edges[0].label
        return tuple(
            bool(assignment[node])
            for node in alters_by_relation[edge_label]
        )

    search = randomization_max_search(
        family,
        assignments,
        residuals,
        pattern_exposures,
    )
    pattern_p = brute_force_adjusted_p_values(
        family,
        observed,
        residuals,
        pattern_exposures,
        search.maxima,
    )
    reportable_names = tuple(
        name for name in family.depth_first_names() if name != ROOT_PATTERN
    )
    decoy_names = tuple(
        name for name in reportable_names if "DECOY_" in name
    )
    signal_rejections = (
        pattern_p[FRIEND_PATTERN] <= alpha,
        pattern_p[WORK_PATTERN] <= alpha,
    )
    observed_search = randomization_max_search(
        family,
        (observed,),
        residuals,
        pattern_exposures,
    )
    observed_brute = brute_force_randomization_maxima(
        family,
        (observed,),
        residuals,
        pattern_exposures,
    )
    exact_observed_maximum = observed_search.maxima == observed_brute
    if not exact_observed_maximum:
        raise AssertionError("pruned observed maximum differs from exhaustive")
    automatic_objective = observed_search.maxima[0] / focal_count
    pattern_scores = {
        name: linear_statistic(
            residuals,
            pattern_exposures(family.patterns[name], observed),
        )
        / focal_count
        for name in reportable_names
    }
    best_score = max(pattern_scores.values())
    true_maximizer = any(
        math.isclose(pattern_scores[name], best_score)
        for name in (FRIEND_PATTERN, WORK_PATTERN)
    )

    def fixed_provider(
        assignment: tuple[int, ...],
    ) -> tuple[tuple[bool, ...], ...]:
        return count_feature_vectors(assignment, alters_by_relation)

    fixed_maxima = feature_family_maxima(
        assignments,
        residuals,
        fixed_provider,
    )
    fixed_rejection = any_adjusted_rejection(
        fixed_provider(observed),
        residuals,
        fixed_maxima,
        alpha,
    )
    fixed_objective = max(
        linear_statistic(residuals, feature)
        for feature in fixed_provider(observed)
    ) / focal_count

    def specified_provider(
        assignment: tuple[int, ...],
    ) -> tuple[tuple[bool, ...], ...]:
        return typed_feature_vectors(assignment, alters_by_relation)

    specified_maxima = feature_family_maxima(
        assignments,
        residuals,
        specified_provider,
    )
    specified_rejection = any_adjusted_rejection(
        specified_provider(observed),
        residuals,
        specified_maxima,
        alpha,
    )
    specified_objective = max(
        linear_statistic(residuals, feature)
        for feature in specified_provider(observed)
    ) / focal_count
    if automatic_objective + 1e-12 < specified_objective:
        raise AssertionError(
            "the automatic candidate superset has a smaller objective "
            "than its hand-specified subset"
        )
    automatic_signal_rejection = any(signal_rejections)
    if automatic_signal_rejection and not specified_rejection:
        raise AssertionError(
            "a superset maxT search cannot reject a true motif that its "
            "specified-motif subset does not reject"
        )

    exhaustive_evaluations = randomizations * len(family.patterns)
    return TrialResult(
        causalscope_any=automatic_signal_rejection,
        causalscope_both=all(signal_rejections),
        causalscope_report_any=any(
            pattern_p[name] <= alpha for name in reportable_names
        ),
        causalscope_decoy_any=any(
            pattern_p[name] <= alpha for name in decoy_names
        ),
        fixed_motifs_any=fixed_rejection,
        specified_typed_any=specified_rejection,
        causalscope_objective=automatic_objective,
        specified_typed_objective=specified_objective,
        fixed_untyped_objective=fixed_objective,
        causalscope_true_maximizer=true_maximizer,
        exact_observed_maximum=exact_observed_maximum,
        pruning_fraction=1.0 - search.statistic_evaluations / exhaustive_evaluations,
    )


def summarize_binary(values: list[bool]) -> MethodSummary:
    rate = sum(values) / len(values)
    standard_error = math.sqrt(rate * (1.0 - rate) / len(values))
    return MethodSummary(rate, standard_error)


def summarize_continuous(values: list[float]) -> ContinuousSummary:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return ContinuousSummary(mean, math.sqrt(variance / len(values)))


def summarize_trials(trials: list[TrialResult]) -> dict[str, object]:
    return {
        "CausalScope-any": asdict(
            summarize_binary([trial.causalscope_any for trial in trials])
        ),
        "CausalScope-both": asdict(
            summarize_binary([trial.causalscope_both for trial in trials])
        ),
        "CausalScope-any-report": asdict(
            summarize_binary(
                [trial.causalscope_report_any for trial in trials]
            )
        ),
        "CausalScope-decoy-any": asdict(
            summarize_binary(
                [trial.causalscope_decoy_any for trial in trials]
            )
        ),
        "Fixed-untyped": asdict(
            summarize_binary([trial.fixed_motifs_any for trial in trials])
        ),
        "Hand-specified-correct": asdict(
            summarize_binary([trial.specified_typed_any for trial in trials])
        ),
        "CausalScope-objective": asdict(
            summarize_continuous(
                [trial.causalscope_objective for trial in trials]
            )
        ),
        "Hand-specified-objective": asdict(
            summarize_continuous(
                [trial.specified_typed_objective for trial in trials]
            )
        ),
        "Fixed-untyped-objective": asdict(
            summarize_continuous(
                [trial.fixed_untyped_objective for trial in trials]
            )
        ),
        "CausalScope-true-maximizer": asdict(
            summarize_binary(
                [trial.causalscope_true_maximizer for trial in trials]
            )
        ),
        "Exact-vs-exhaustive": asdict(
            summarize_binary(
                [trial.exact_observed_maximum for trial in trials]
            )
        ),
        "mean_pruning_fraction": sum(
            trial.pruning_fraction for trial in trials
        )
        / len(trials),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    scenarios = {
        "null": 0.0,
        "hidden_typed_spillover": args.spillover,
    }
    results: dict[str, object] = {
        "config": {
            "repetitions": args.repetitions,
            "focal_count": args.focal_count,
            "randomizations": args.randomizations,
            "alpha": args.alpha,
            "spillover": args.spillover,
            "direct_effect": args.direct_effect,
            "noise_sd": args.noise_sd,
            "decoy_relations": args.decoy_relations,
            "seed": args.seed,
        }
    }
    for scenario_index, (scenario, spillover) in enumerate(scenarios.items()):
        trials = [
            run_trial(
                seed=args.seed + scenario_index * 1_000_000 + repetition,
                focal_count=args.focal_count,
                randomizations=args.randomizations,
                alpha=args.alpha,
                spillover=spillover,
                direct_effect=args.direct_effect,
                noise_sd=args.noise_sd,
                decoy_relations=args.decoy_relations,
            )
            for repetition in range(args.repetitions)
        ]
        results[scenario] = summarize_trials(trials)
    return results


def print_results(results: dict[str, object]) -> None:
    print("CausalScope typed-edge representation benchmark")
    print(json.dumps(results["config"], indent=2, sort_keys=True))
    for scenario in ("null", "hidden_typed_spillover"):
        print(f"\n{scenario}")
        summary = results[scenario]
        assert isinstance(summary, dict)
        for method in (
            "CausalScope-any",
            "CausalScope-both",
            "CausalScope-any-report",
            "CausalScope-decoy-any",
            "Fixed-untyped",
            "Hand-specified-correct",
        ):
            metric = summary[method]
            assert isinstance(metric, dict)
            print(
                f"  {method:20s} "
                f"{metric['rejection_rate']:.3f} "
                f"+/- {metric['standard_error']:.3f}"
            )
        for metric_name in (
            "CausalScope-objective",
            "Hand-specified-objective",
            "Fixed-untyped-objective",
        ):
            metric = summary[metric_name]
            assert isinstance(metric, dict)
            print(
                f"  {metric_name:25s} "
                f"{metric['mean']:.3f} "
                f"+/- {metric['standard_error']:.3f}"
            )
        for metric_name in (
            "CausalScope-true-maximizer",
            "Exact-vs-exhaustive",
        ):
            metric = summary[metric_name]
            assert isinstance(metric, dict)
            print(
                f"  {metric_name:25s} "
                f"{metric['rejection_rate']:.3f}"
            )
        print(f"  mean pruning fraction {summary['mean_pruning_fraction']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--focal-count", type=int, default=80)
    parser.add_argument("--randomizations", type=int, default=199)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--spillover", type=float, default=1.5)
    parser.add_argument("--direct-effect", type=float, default=1.0)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--decoy-relations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args)
    print_results(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
