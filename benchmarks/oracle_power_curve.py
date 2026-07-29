"""Power curve for automatic mining versus specified and misspecified motifs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.typed_edge_experiment import (
    run_trial,
    summarize_binary,
)


def parse_effect_sizes(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(","))
    if not values:
        raise ValueError("at least one effect size is required")
    if any(value < 0.0 for value in values):
        raise ValueError("effect sizes must be nonnegative")
    return values


def run_power_curve(args: argparse.Namespace) -> dict[str, object]:
    effect_sizes = parse_effect_sizes(args.effect_sizes)
    rows: list[dict[str, object]] = []

    for spillover in effect_sizes:
        trials = [
            run_trial(
                seed=args.seed + repetition,
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
        automatic = summarize_binary(
            [trial.causalscope_any for trial in trials]
        )
        automatic_both = summarize_binary(
            [trial.causalscope_both for trial in trials]
        )
        automatic_report = summarize_binary(
            [trial.causalscope_report_any for trial in trials]
        )
        automatic_decoy = summarize_binary(
            [trial.causalscope_decoy_any for trial in trials]
        )
        oracle = summarize_binary(
            [trial.oracle_typed_any for trial in trials]
        )
        misspecified = summarize_binary(
            [trial.fixed_motifs_any for trial in trials]
        )
        rows.append(
            {
                "spillover": spillover,
                "automatic_signal_any": asdict(automatic),
                "automatic_signal_both": asdict(automatic_both),
                "automatic_any_report": asdict(automatic_report),
                "automatic_decoy_any": asdict(automatic_decoy),
                "oracle_specified": asdict(oracle),
                "fixed_untyped": asdict(misspecified),
                "oracle_minus_automatic": (
                    oracle.rejection_rate - automatic.rejection_rate
                ),
                "mean_pruning_fraction": sum(
                    trial.pruning_fraction for trial in trials
                )
                / len(trials),
            }
        )

    return {
        "config": {
            "repetitions": args.repetitions,
            "focal_count": args.focal_count,
            "randomizations": args.randomizations,
            "alpha": args.alpha,
            "effect_sizes": effect_sizes,
            "direct_effect": args.direct_effect,
            "noise_sd": args.noise_sd,
            "decoy_relations": args.decoy_relations,
            "seed": args.seed,
        },
        "rows": rows,
    }


def print_results(results: dict[str, object]) -> None:
    config = results["config"]
    assert isinstance(config, dict)
    print("Oracle-specified versus automatic motif-mining power curve")
    print(json.dumps(config, indent=2, sort_keys=True))
    print(
        "\n beta | automatic | oracle | oracle gap | fixed | "
        "any decoy | any report"
    )
    print("-" * 75)
    rows = results["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        automatic = row["automatic_signal_any"]
        oracle = row["oracle_specified"]
        fixed = row["fixed_untyped"]
        decoy = row["automatic_decoy_any"]
        any_report = row["automatic_any_report"]
        assert isinstance(automatic, dict)
        assert isinstance(oracle, dict)
        assert isinstance(fixed, dict)
        assert isinstance(decoy, dict)
        assert isinstance(any_report, dict)
        print(
            f"{row['spillover']:5.2f} |"
            f" {automatic['rejection_rate']:9.3f} |"
            f" {oracle['rejection_rate']:6.3f} |"
            f" {row['oracle_minus_automatic']:10.3f} |"
            f" {fixed['rejection_rate']:5.3f} |"
            f" {decoy['rejection_rate']:9.3f} |"
            f" {any_report['rejection_rate']:10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--focal-count", type=int, default=80)
    parser.add_argument("--randomizations", type=int, default=199)
    parser.add_argument(
        "--effect-sizes",
        default="0,0.25,0.5,0.75,1.0,1.25,1.5",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--direct-effect", type=float, default=1.0)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--decoy-relations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_power_curve(args)
    print_results(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
