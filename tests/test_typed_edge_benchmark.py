from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_benchmark_module() -> object:
    path = Path(__file__).parents[1] / "benchmarks" / "typed_edge_experiment.py"
    spec = importlib.util.spec_from_file_location("typed_edge_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_typed_edge_trial_runs_end_to_end() -> None:
    benchmark = load_benchmark_module()
    result = benchmark.run_trial(
        seed=17,
        focal_count=12,
        randomizations=19,
        alpha=0.2,
        spillover=1.5,
        direct_effect=1.0,
        noise_sd=1.0,
    )

    assert isinstance(result.causalscope_any, bool)
    assert isinstance(result.causalscope_report_any, bool)
    assert result.causalscope_decoy_any is False
    assert isinstance(result.fixed_motifs_any, bool)
    assert result.exact_observed_maximum
    assert result.causalscope_objective >= result.specified_typed_objective
    assert 0.0 <= result.pruning_fraction <= 1.0


def test_typed_edge_trial_includes_automatic_decoy_search() -> None:
    benchmark = load_benchmark_module()
    graph, _, alters = benchmark.build_typed_star_transactions(
        focal_count=4,
        decoy_relations=3,
    )
    family = benchmark.generate_one_hop_treated_patterns(
        graph,
        root_label="User",
    )
    result = benchmark.run_trial(
        seed=23,
        focal_count=12,
        randomizations=19,
        alpha=0.2,
        spillover=1.0,
        direct_effect=1.0,
        noise_sd=1.0,
        decoy_relations=3,
    )

    assert set(alters) == {
        "FRIEND",
        "WORKS_WITH",
        "DECOY_00",
        "DECOY_01",
        "DECOY_02",
    }
    assert len(family.patterns) == 6
    assert isinstance(result.causalscope_decoy_any, bool)


def test_automatic_rejection_implies_specified_rejection() -> None:
    benchmark = load_benchmark_module()

    for seed in range(10):
        result = benchmark.run_trial(
            seed=seed,
            focal_count=20,
            randomizations=39,
            alpha=0.1,
            spillover=0.75,
            direct_effect=1.0,
            noise_sd=1.0,
            decoy_relations=5,
        )
        assert not result.causalscope_any or result.specified_typed_any
