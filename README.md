# CausalScope

CausalScope is a research prototype for randomization-aware causal exposure
pattern mining over property graphs.

The core algorithm searches a rooted pattern-growth tree while carrying a
randomization-specific upper envelope for every branch. A branch is pruned only
when no descendant can improve any randomization maximum. A second pass uses
those exact maxima to prune subtrees that cannot contain a maxT-significant
exposure pattern.

## Current scope

The correctness-first prototype currently supports:

- directed, labeled property graphs;
- rooted connected patterns with node labels, property predicates, and
  treatment literals;
- anchored subgraph-monomorphism matching;
- conditional Bernoulli randomization;
- exact brute-force randomization maxima;
- assignment-wise randomization-envelope pruning;
- adjusted-p subtree pruning.

The first implementation deliberately uses a finite pattern-growth tree and a
naive matcher. The next milestone replaces that generator with canonical
DFS-code growth and projected embedding databases without changing the
randomization search interface.

The current exposure grammar is binary and existential. It does not yet
subsume Causal Network Motifs' normalized motif-count and ratio features.
Consequently, the current common-objective dominance result applies to a
correctly hand-specified binary-pattern subset, not to the complete WWW
algorithm.

An exact sub-two search design for a separate additive motif-count grammar is
documented in
[`docs/sub_two_closed_motif_search.md`](docs/sub_two_closed_motif_search.md).
It reduces score-preserving causal closures to weighted maximal bicliques and
adds witness-anchored exact search with `O*(R * 2^s)` complexity, where `s` is
the maximum number of atoms simultaneously satisfied by one embedding rather
than the global atom vocabulary. Generator-bounded search replaces `2^s` by
`sum(j=0..l, binomial(s,j))`. The IPS track targets a branching factor of
approximately `1.3954`. The checked-in formal-context and terminal-DP code are
correctness oracles; the full production weighted-IPS recursion remains.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
causalscope-demo
```

## Statistical target

For focal units `F`, CausalScope conditions on their observed own-treatment
assignments and resamples the remaining assignments from the known experiment
design. The initial test statistic is

```text
T_P(z) = |sum_i a_i x_P(i, z)|,
```

where `a_i` is the outcome centered within the focal unit's own-treatment
stratum and `x_P(i, z)` indicates whether a treatment-colored rooted pattern is
present around unit `i`.

The finite-sample randomization interpretation is currently limited to the
sharp no-spillover null on the focal units. Pattern-specific effect estimation
and observational extensions are intentionally outside the MVP.

## Research positioning

The primary comparison is Yuan, Altenburger, and Kooti's *Causal Network
Motifs* (WWW 2021). See
[docs/comparison_causal_motifs.md](docs/comparison_causal_motifs.md) for the
claim boundary and the experiments required to establish an empirical
advantage without overstating the current prototype.

The executable
[objective-recovery benchmark](benchmarks/README.md#exact-objective-and-recovery-curve)
compares automatic search over 22 typed motifs with a correctly hand-specified
two-motif subset and a misspecified untyped dictionary. It reports exact
equality with exhaustive maximization, true-pattern recovery, adjusted power,
and null error rather than treating a larger raw maximum as sufficient
evidence.
