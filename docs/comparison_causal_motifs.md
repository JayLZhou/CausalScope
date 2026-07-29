# Comparison with Causal Network Motifs

This document defines the comparison between CausalScope and:

> Yuan Yuan, Kristen M. Altenburger, and Farshad Kooti. Causal Network
> Motifs: Identifying Heterogeneous Spillover Effects in A/B Tests. WWW 2021.
> <https://arxiv.org/abs/2010.09911>

The authors' reference implementation is available at
<https://github.com/facebookresearch/CausalMotifs>.

## What Causal Network Motifs does

The WWW 2021 method has two stages:

1. The analyst specifies an ego-network radius and a collection of network
   motifs. Counts of treatment-colored motifs are normalized by the
   corresponding uncolored motif counts to form an interference vector in
   `[0, 1]^m`.
2. A modified regression tree partitions this fixed interference-vector
   space. Its split criterion uses inverse-exposure-probability weighted SSE,
   Monte Carlo positivity checks, and honest sample splitting. Each leaf is an
   exposure condition for which a Hajek average potential outcome is
   estimated.

The method automatically discovers useful thresholds and combinations inside
the supplied feature vector. It does not automatically generate the structural
motif dictionary. The paper explicitly requires the analyst to choose the
ego-network radius and motif family; its real-data experiment uses only dyads
and triads because of computational constraints.

## What CausalScope changes

CausalScope treats the structural pattern itself as the search object:

```text
rooted PG pattern
    + edge direction/type
    + node/edge property predicates
    + treatment literals.
```

It searches a canonical pattern-growth tree rather than first materializing a
fixed motif feature table. Under a monotone exposure grammar, each descendant
has an assignment-wise exposed-unit set contained in its parent. This supports
safe randomization-envelope pruning and maxT-adjusted inference over the full
implicit pattern family.

## Claims that can be proved

### Search completeness

Let `G_h` be the finite rooted pattern family induced by the grammar and size
limit. If the canonical generator is complete, CausalScope examines or safely
prunes every pattern in `G_h`. A causally relevant pattern need not be named in
advance.

Causal Network Motifs can use a pattern only when the analyst included its
corresponding motif feature in the interference vector.

### Exact pruning equivalence

For every conditional randomization assignment `z_b`, CausalScope returns

```text
M_b = max_{P in G_h} T_P(z_b)
```

exactly, while visiting only branches that can improve at least one current
maximum. This result is equality with exhaustive pattern enumeration, not an
approximation guarantee.

The WWW tree is a greedy recursive partitioner over a fixed feature table. It
does not claim a globally optimal motif dictionary or a certificate over an
implicit structural pattern space.

### Randomization-valid global discovery

Under the sharp no-spillover null on focal units, CausalScope's conditional
maxT procedure controls the probability of reporting any pattern at level
`alpha`, up to the standard finite Monte Carlo correction. The adjusted-p
subtree lower bound is also safe for every descendant.

The WWW method uses honest splitting to avoid reusing the tree-training sample
for leaf effect estimation. This is valuable, but it is a different guarantee:
it does not provide maxT familywise control over an adaptively mined,
exponentially large motif family.

### Property-graph expressiveness

CausalScope patterns may distinguish edge types, directions, node labels, and
pre-treatment property predicates. Ordinary dyad/triad/tetrad counts can be
represented structurally, but the current implementation exposes only binary
existence events. Normalized motif counts, ratios, and arbitrary tree
thresholds require a cardinality-exposure extension before the implemented
grammar strictly contains the WWW feature space.

This expressiveness claim assumes all structural properties are measured
before treatment. Post-treatment graph properties must not enter either
method.

## Claims that must not be made yet

CausalScope does not currently dominate Causal Network Motifs in every task.
The WWW implementation is currently stronger in:

- normalized count and ratio exposure features;
- Hajek estimation of average potential outcomes;
- positivity-aware exposure-region trees;
- heterogeneous direct-effect estimation;
- demonstrated large-scale real-world A/B analysis.

The current CausalScope theorem is a finite-sample discovery guarantee for the
sharp no-spillover null. Pattern-specific effect estimation under arbitrary
simultaneous spillovers is not implied by that theorem.

## Benchmark protocol

The main comparison must include the authors' public implementation without
rewriting its objective. Every experiment should report both statistical and
computational outcomes.

| Experiment | Data-generating process | Primary metric | Intended conclusion |
|---|---|---|---|
| Null calibration | Direct effect but no spillover | `P(any false pattern)` | Verify CausalScope control at `alpha` |
| In-dictionary motif | Dyad/triad spillover used by WWW | Power, runtime | Ensure no loss on their home setting |
| Hidden higher-order pattern | Spillover from an unlisted 4-6 node pattern | Pattern recall, power | Test automatic generation |
| Typed PG pattern | Effect requires edge type or node property | Pattern recall, false discoveries | Test PG expressiveness |
| Correlated decoys | Many frequent noncausal motifs correlate structurally | Exact recovery, false discoveries | Test causal rather than frequency ranking |
| Scale-up | Increasing nodes, degree, pattern size, randomizations | Runtime, memory, pruning ratio | Test search advantage |

The comparison should use at least these baselines:

1. `CausalMotifs-dyad`: the WWW tree with treated-neighbor features.
2. `CausalMotifs-specified`: the WWW tree with a hand-specified motif set that
   contains the ground-truth pattern.
3. `CausalMotifs-misspecified`: the same method without the ground-truth
   pattern.
4. `Enumerate+maxT`: exhaustive enumeration of CausalScope's pattern family.
5. `CausalScope-no-envelope`: canonical pattern growth without causal pruning.
6. `CausalScope`: the complete method.

## Success criteria

CausalScope should be described as better only if the experiments establish:

1. empirical familywise error at or below the requested `alpha` under the null;
2. higher recovery or power when the causal pattern is not pre-specified;
3. identical discoveries to exhaustive maxT on small instances;
4. materially fewer visited patterns or lower runtime than exhaustive mining;
5. competitive power when the WWW method is given the correct motif dictionary.

If criterion 5 fails, the paper should claim a validity and automation
advantage, not universal statistical dominance.

## Objective dominance versus testing power

There are two different comparisons and they must not be conflated:

1. **Misspecified WWW dictionary:** the true structural motif is absent. An
   automatic structural miner can have much higher power because it searches a
   richer representation.
2. **Correctly specified WWW dictionary:** the true motif is supplied by the
   analyst. Automatic mining pays for searching irrelevant alternatives and
   should not be claimed to have uniformly greater testing power.

The second statement has an exact maxT argument in the representation-matched
experiment. Let `H_manual` be the two true motifs and
`H_manual subset H_auto`. For every assignment `z_b`,

```text
M_auto(b) = max_{P in H_auto} T_P(z_b)
          >= max_{P in H_manual} T_P(z_b)
          = M_manual(b).
```

Thus the exact automatic raw objective can never be lower than the objective
of its manual subset. For either true motif `P`, however, maxT adjustment also
implies

```text
p_auto(P) >= p_manual(P).
```

Thus an automatic true-motif rejection implies a hand-specified rejection in
every paired trial. The relevant power cost is

```text
R(beta, D) = Power_manual(beta) - Power_auto(beta, D),
```

where `D` is the number of irrelevant candidate relations. A good automatic
miner exactly maximizes the larger objective, identifies the true maximizer,
and makes this testing-power gap small at practically relevant effect sizes.

## First executable comparison

`benchmarks/typed_edge_experiment.py` implements the representation comparison
before the full official-code replication.

Every focal unit has exactly one `FRIEND` neighbor and one `WORKS_WITH`
neighbor. The spillover response is

```text
beta * (Z_work - Z_friend).
```

The fixed CausalMotifs-style baseline receives one-hot features for every
possible untyped treated-neighbor count. Conditional on count zero, one, or
two, its expected spillover response is zero. CausalScope reads the edge types
from the property graph and automatically generates the two typed patterns.

With 100 repeated experiments, 80 focal units, 199 randomizations, and
`alpha=0.05`, the checked-in run produced:

| Scenario | CausalScope any | CausalScope both | Fixed motifs any | Hand-specified |
|---|---:|---:|---:|---:|
| No spillover | 0.05 | 0.00 | 0.02 | 0.05 |
| Hidden typed spillover | 1.00 | 0.99 | 0.07 | 1.00 |

At `beta=1.5`, this result establishes that both the automatic search and
hand-specified control have reached the ceiling while the signal remains
absent from the fixed untyped motif representation. It does not show that
automatic discovery is cost-free.

It does not yet establish a universal advantage over the official WWW
algorithm. The next comparison must run the authors' `causalPartition` code on
the in-dictionary dyad/triad data-generating processes and compare regime
recovery and Hajek effect error.

## Exact-objective and recovery curve

`benchmarks/objective_recovery_curve.py` adds 20 zero-effect typed relations to the
two data-generating relations. CausalScope discovers and tests all 22 relation
patterns, whereas the hand-specified control receives only the two correct
ones. With 100 repetitions, 80 focal units, 199 randomizations, and
`alpha=0.05`:

| `beta` | Automatic objective | Hand-specified objective | Fixed-untyped objective | True argmax | Automatic power | Hand-specified power | Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.119 | 0.061 | 0.065 | 0.07 | 0.01 | 0.05 | 1.00 |
| 0.25 | 0.128 | 0.094 | 0.066 | 0.29 | 0.07 | 0.22 | 1.00 |
| 0.50 | 0.162 | 0.153 | 0.070 | 0.73 | 0.27 | 0.73 | 1.00 |
| 0.75 | 0.215 | 0.213 | 0.075 | 0.94 | 0.76 | 0.95 | 1.00 |
| 1.00 | 0.274 | 0.274 | 0.081 | 0.99 | 0.96 | 1.00 | 1.00 |
| 1.25 | 0.334 | 0.334 | 0.088 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.50 | 0.394 | 0.394 | 0.096 | 1.00 | 1.00 | 1.00 | 1.00 |

The exact automatic objective is never below its correctly hand-specified
subset, and every pruned maximum equals exhaustive enumeration. As signal
strength grows, the probability that the global maximizer is one of the two
true typed patterns rises from its null chance level to one. The fixed untyped
objective remains much smaller because the opposite typed effects cancel
after conditioning on total treated-neighbor count.

The larger automatic space also raises the null raw maximum, which is why
objective value alone is insufficient. Adjusted power and null FWER remain
necessary. This experiment does not run the official WWW regression tree and
does not demonstrate a runtime advantage: the current flat one-hop family
prunes only about `0.4%` of statistic evaluations. Those claims require the
official adapter and a multi-level pattern-growth benchmark.
