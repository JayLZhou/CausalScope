# Benchmarks

The benchmark suite is designed to compare CausalScope with
representation-matched baselines and, in a later end-to-end adapter, the
official `facebookresearch/CausalMotifs` implementation.

## Executable typed-edge experiment

The first experiment isolates pattern representation. Every focal user has one
`FRIEND` and one `WORKS_WITH` neighbor, and the outcome contains

```text
beta * (Z_work - Z_friend).
```

With no decoys, the fixed CausalMotifs-style baseline receives the complete
one-hot basis for the untyped treated-neighbor count `0, 1, 2`. Consequently,
it can express any rule based on the dyad/open-triad treatment count, but its
conditional expected signal is zero. CausalScope discovers the two edge-typed
patterns from the property graph itself.

Both methods use the same conditional maxT procedure in this experiment. A
power difference therefore isolates representation rather than inference or
tree implementation.

```bash
python benchmarks/typed_edge_experiment.py \
  --repetitions 100 \
  --output benchmark_results/typed_edge.json
```

The experiment reports:

- null familywise rejection rates;
- power to detect any typed spillover pattern;
- probability of recovering both typed patterns;
- a correctly hand-specified typed-feature control;
- normalized maximized causal objectives;
- true-pattern argmax recovery and exact equality with exhaustive search;
- the fraction of statistic evaluations avoided by envelope pruning.

### Reference result

The checked-in reference run uses 100 repeated experiments, 80 focal units,
199 conditional randomizations, `alpha=0.05`, and seed `20260730`.

| Scenario | CausalScope any | CausalScope both | Fixed motifs any | Hand-specified |
|---|---:|---:|---:|---:|
| No spillover | 0.05 | 0.00 | 0.02 | 0.05 |
| Hidden typed spillover | 1.00 | 0.99 | 0.07 | 1.00 |

The complete machine-readable result is in
[`results/typed_edge_default.json`](results/typed_edge_default.json).

This is a representation-isolation experiment, not yet an end-to-end
replication of the WWW tree. The fixed baseline is deliberately given the same
maxT inference as CausalScope and the complete untyped treatment-count basis.
Its low power in the second row therefore reflects missing edge-type semantics,
not weaker inference code.

The equality between CausalScope and the hand-specified control at `beta=1.5`
is a ceiling effect. It must not be interpreted as evidence that automatic
search is as powerful as being told the correct motif.

## Exact objective and recovery curve

`objective_recovery_curve.py` separates three questions that the first
experiment had conflated:

1. Does pruned automatic search return exactly the exhaustive maximum?
2. Does the larger automatic candidate set attain at least the objective of
   its correctly hand-specified subset?
3. Is the maximizer a true causal pattern rather than a high-scoring decoy?

The compared candidate spaces are:

1. `Hand-specified-correct` receives exactly `FRIEND` and `WORKS_WITH`.
2. `CausalScope-automatic` reads the graph schema and searches those two motifs
   together with 20 causally irrelevant typed relations.
3. `Fixed-untyped` receives the complete one-hot basis for the total number of
   treated neighbors, but no relation types.

All three use the same residual statistic and conditional randomization
procedure. The experiment therefore isolates motif specification and
multiplicity, not differences in effect estimators.

```bash
python -m benchmarks.objective_recovery_curve \
  --repetitions 100 \
  --decoy-relations 20 \
  --output benchmarks/results/objective_recovery_curve.json
```

### Objective-recovery reference result

The checked-in run uses 100 repetitions, 80 focal units, 199 conditional
randomizations, `alpha=0.05`, and seed `20260730`.

| `beta` | Automatic objective | Hand-specified objective | Fixed-untyped objective | True argmax | Automatic power | Hand-specified power | Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.119 | 0.061 | 0.065 | 0.07 | 0.01 | 0.05 | 1.00 |
| 0.25 | 0.128 | 0.094 | 0.066 | 0.29 | 0.07 | 0.22 | 1.00 |
| 0.50 | 0.162 | 0.153 | 0.070 | 0.73 | 0.27 | 0.73 | 1.00 |
| 0.75 | 0.215 | 0.213 | 0.075 | 0.94 | 0.76 | 0.95 | 1.00 |
| 1.00 | 0.274 | 0.274 | 0.081 | 0.99 | 0.96 | 1.00 | 1.00 |
| 1.25 | 0.334 | 0.334 | 0.088 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.50 | 0.394 | 0.394 | 0.096 | 1.00 | 1.00 | 1.00 | 1.00 |

The machine-readable result is in
[`results/objective_recovery_curve.json`](results/objective_recovery_curve.json).

The objective ordering is structural, not accidental. Let `H_manual` contain the two
true motifs and let `H_auto` be the automatically generated superset. For
the observed assignment and every randomized assignment `b`,

```text
max_{P in H_auto} T_P(b) >= max_{P in H_manual} T_P(b).
```

The implementation additionally compares each pruned observed maximum with
exhaustive enumeration and asserts equality before recording a trial. The
`Exact` column is therefore an empirical regression check of the exact-search
theorem, not an approximation ratio.

Here `H_manual` is a binary typed-pattern control inside CausalScope's current
grammar. It is not the full WWW normalized motif-count representation. A
strict common-objective dominance claim over WWW itself requires the planned
cardinality-exposure grammar and official-code adapter.

The larger family also has a larger raw maximum under the null (`0.119` versus
`0.061`), so objective dominance alone is not evidence of causal discovery.
The true-argmax, adjusted-power, and null-FWER columns are required to separate
signal recovery from maximization over noise.

This experiment still uses an explicit, flat one-hop family. Its mean pruning
fraction is only about `0.4%`, so it establishes exact correctness and
recovery, not a runtime advantage over exhaustive mining. Demonstrating the
computational claim requires the planned multi-level canonical pattern-growth
benchmark. The hand-specified controls also share CausalScope's maxT inference;
they are not the authors' `causalPartition` program. The official WWW
implementation remains a separate end-to-end baseline.

## Planned layout

```text
benchmarks/
  configs/
    null.yaml
    in_dictionary.yaml
    hidden_pattern.yaml
    typed_property_graph.yaml
    correlated_decoys.yaml
  adapters/
    causal_motifs.py
    causalscope.py
  generators/
    network_experiment.py
    property_graph_experiment.py
  metrics.py
  run.py
```

The external baseline should be installed or cloned by the researcher at run
time. Its source code must not be copied into this repository.
