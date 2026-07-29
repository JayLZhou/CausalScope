# Benchmarks

The benchmark suite will compare CausalScope with the official
`facebookresearch/CausalMotifs` implementation.

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

The first executable benchmark will be the null-calibration experiment. It
must estimate the probability of at least one reported pattern across repeated
randomized experiments and compare that probability with the requested
familywise level.

The external baseline should be installed or cloned by the researcher at run
time. Its source code must not be copied into this repository.

