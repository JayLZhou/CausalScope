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
- a typed-feature oracle upper reference;
- the fraction of statistic evaluations avoided by envelope pruning.

### Reference result

The checked-in reference run uses 100 repeated experiments, 80 focal units,
199 conditional randomizations, `alpha=0.05`, and seed `20260730`.

| Scenario | CausalScope any | CausalScope both | Fixed motifs any | Typed oracle |
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

The equality between CausalScope and the typed oracle at `beta=1.5` is a
ceiling effect. It must not be interpreted as evidence that automatic search
is as powerful as being told the correct motif.

## Specified oracle versus automatic mining

`oracle_power_curve.py` performs the comparison that exposes the cost of
automatic discovery:

1. `Oracle-specified` is handed exactly `FRIEND` and `WORKS_WITH`, the two
   data-generating motifs.
2. `CausalScope-automatic` reads the graph schema and searches those two motifs
   together with 20 causally irrelevant typed relations.
3. `Fixed-untyped` receives the complete one-hot basis for the total number of
   treated neighbors, but no relation types.

All three use the same residual statistic and conditional randomization
procedure. The experiment therefore isolates motif specification and
multiplicity, not differences in effect estimators.

```bash
python -m benchmarks.oracle_power_curve \
  --repetitions 100 \
  --decoy-relations 20 \
  --output benchmarks/results/oracle_power_curve.json
```

### Oracle-gap reference result

The checked-in run uses 100 repetitions, 80 focal units, 199 conditional
randomizations, `alpha=0.05`, and seed `20260730`.

| Spillover `beta` | Automatic: true motif | Specified oracle | Oracle gap | Fixed untyped | Any decoy |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.01 | 0.05 | 0.04 | 0.05 | 0.03 |
| 0.25 | 0.07 | 0.22 | 0.15 | 0.05 | 0.05 |
| 0.50 | 0.27 | 0.73 | 0.46 | 0.06 | 0.03 |
| 0.75 | 0.76 | 0.95 | 0.19 | 0.06 | 0.03 |
| 1.00 | 0.96 | 1.00 | 0.04 | 0.05 | 0.04 |
| 1.25 | 1.00 | 1.00 | 0.00 | 0.05 | 0.03 |
| 1.50 | 1.00 | 1.00 | 0.00 | 0.04 | 0.04 |

The machine-readable result is in
[`results/oracle_power_curve.json`](results/oracle_power_curve.json).

This ordering is structural, not accidental. Let `H_oracle` contain the two
true motifs and let `H_auto` be the automatically generated superset. For
every randomized assignment `b`,

```text
max_{P in H_auto} T_P(b) >= max_{P in H_oracle} T_P(b).
```

The maxT adjusted p-value of either true motif under automatic search is
therefore never smaller than its oracle p-value. Automatic rejection of a true
motif implies oracle rejection in every paired trial. The scientifically
useful question is how quickly the automatic method closes this unavoidable
oracle gap while retaining valid global discovery over unknown patterns.

`Oracle-specified` is a favorable representation oracle under shared maxT
inference; it is not a relabeling of the authors' `causalPartition` program.
The official WWW implementation still needs a separate end-to-end experiment
for regime recovery and Hajek effect estimation.

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
