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
embedded as a strict special case of this grammar.

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

