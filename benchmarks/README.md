# Benchmarks

The benchmark suite compares CausalScope with both representation-matched
baselines and the official `facebookresearch/CausalMotifs` implementation.

## Executable typed-edge experiment

The first experiment isolates pattern representation. Every focal user has one
`FRIEND` and one `WORKS_WITH` neighbor, and the outcome contains

```text
beta * (Z_work - Z_friend).
```

The fixed CausalMotifs-style baseline receives the complete one-hot basis for
the untyped treated-neighbor count `0, 1, 2`. Consequently, it can express any
rule based on the dyad/open-triad treatment count, but its conditional expected
signal is zero. CausalScope discovers the two edge-typed patterns from the
property graph itself.

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
