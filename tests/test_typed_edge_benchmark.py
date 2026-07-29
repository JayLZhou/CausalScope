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
    assert isinstance(result.fixed_motifs_any, bool)
    assert 0.0 <= result.pruning_fraction <= 1.0
