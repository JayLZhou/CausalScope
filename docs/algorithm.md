# Algorithm contract

This document records the assumptions that make CausalScope's pruning exact.
The implementation should not weaken these conditions silently.

## Conditional randomization target

Let `F` be a set of focal units. CausalScope conditions on the observed own
treatments of `F` and resamples the remaining assignments from the known
experimental design. The finite-sample target is the sharp no-spillover null on
the focal outcomes:

```text
Y_i(z) = Y_i(z') whenever z_i = z'_i, for every i in F.
```

This allows arbitrary direct treatment effects because each focal unit's own
treatment is fixed across the conditional randomization distribution.

## Monotone pattern grammar

A child pattern may:

- add a node and an incident labeled edge;
- add an edge between existing roles;
- refine an unconstrained node label;
- add conjunctive property predicates;
- add a treatment literal to a previously unconstrained role.

It may not remove or relax a parent constraint. Role identifiers from the
parent must remain stable in the child. These conditions imply

```text
x_child(i, z) <= x_parent(i, z)
```

for every root unit and every treatment assignment. `PatternFamily` validates
the corresponding syntactic invariant before search begins.

## Assignment-wise envelope

For fixed residuals `a_i`, define

```text
T_P(z) = abs(sum_i a_i x_P(i, z)).
```

For a search-tree node `P`, define

```text
U_P(z) = max(
    sum positive a_i exposed by P,
    abs(sum negative a_i exposed by P),
).
```

Every descendant `Q` has an exposed set contained in `P`'s exposed set, so
`T_Q(z) <= U_P(z)`.

During the first search pass, assignment `b` is active at `P` only when
`U_P(z_b)` exceeds its current maximum `M_b`. An empty active set safely prunes
the subtree. At completion, every `M_b` equals the exhaustive maximum over the
full pattern family.

## Adjusted-p subtree bound

For an observed statistic `T`, the Monte Carlo maxT adjusted p-value is

```text
(1 + count_b[M_b >= T]) / (B + 1).
```

For every descendant `Q` of `P`, `T_Q(z_obs) <= U_P(z_obs)`. Therefore

```text
p_adjusted(Q) >=
    (1 + count_b[M_b >= U_P(z_obs)]) / (B + 1).
```

If this lower bound exceeds `alpha`, the complete subtree is untestable at the
requested level.

## Next implementation milestone

The current `PatternFamily` is an explicit correctness oracle. The production
miner will replace it with a lazy child provider implementing:

1. minimum canonical DFS codes;
2. rightmost-path extensions;
3. projected embedding databases grouped by root unit;
4. structural minimum-support pruning;
5. assignment bitsets for treatment literals;
6. child ordering by the assignment-wise envelope.

The randomization search must consume a generic lazy pattern-tree interface so
that its statistical logic remains unchanged when the gSpan engine replaces
the explicit family.

