# Exact sub-two search through causal closure

This note specifies a route to a literal exponential base below two. It is a
separate algorithmic track from the current binary-existential prototype. The
claim must not be applied to binary exposure without a new reduction.

## 1. Why motif-count exposure changes the search problem

Fix a rooted connected role skeleton `tau` with `k` roles. Its mandatory
wildcard edges guarantee connectivity. Node labels, edge types, property
predicates, treatment literals, and optional chords form a finite atom set
`A_tau`.

For assignment `z_b`, let `W_tau,b` contain every rooted embedding of `tau`.
Each assignment-embedding witness `(b, phi)` has a transaction

```text
t_b(phi) subseteq A_tau
```

containing exactly the atoms satisfied by that embedding. A pattern intent
`P subseteq A_tau` matches `phi` iff `P subseteq t_b(phi)`.

For a fixed root normalization `D_i > 0`, define count exposure

```text
m_P(i, z_b) =
    (1 / D_i) *
    count{phi in W_tau,b : root(phi) = i and P subseteq t_b(phi)}.
```

This includes raw motif counts when `D_i = 1` and motif densities when `D_i`
is a fixed root-specific opportunity count. Crucially, `D_i` may not depend
on `P`.

With residual `a_i`, the signed statistic linearizes over witnesses:

```text
S_b(P)
  = sum_i a_i * m_P(i, z_b)
  = sum_{phi: P subseteq t_b(phi)} a_root(phi) / D_root(phi).

T_b(P) = abs(S_b(P)).
```

This is the property missing from binary existential exposure. If a root has
three matching embeddings, all three intentionally contribute to a
motif-count exposure instead of being collapsed by an OR.

## 2. Randomization-stable causal closure

Build one stacked witness set across the observed assignment and all `B`
randomizations:

```text
W_tau,* = disjoint_union_{b=0..B} {(b, phi) : phi in W_tau,b}.
```

Give witness `(b, phi)` the vector weight

```text
omega(b, phi)
  = e_b * a_root(phi) / D_root(phi),
```

where `e_b` is the one-hot vector for coordinate `b`. Then

```text
S(P) = sum_{(b, phi): P subseteq t_b(phi)} omega(b, phi)
     = (S_0(P), ..., S_B(P)).
```

The atom-by-stacked-witness incidence relation is a formal context. For intent
`P`, write

```text
P'  = {(b, phi) : P subseteq t_b(phi)}
P'' = intersection_{(b, phi) in P'} t_b(phi).
```

For an empty extent, use the standard closure `P'' = A_tau`.

### Lemma 1: closure invariance

For every intent `P`,

```text
(P'')' = P'.
```

Therefore the complete randomization vector is invariant:

```text
S(P'') = S(P).
```

Proof: `P subseteq P''` gives `(P'')' subseteq P'`. Every atom in `P''`
is shared by every witness in `P'`, giving the reverse inclusion.

### Corollary 1: closed-pattern sufficiency

Every pattern has a closed representative with exactly the same stacked
embedding set and randomization score vector. It consequently has the same
observed statistic, null statistics, and pattern-level randomization p-value.
For every assignment `b`, hence

```text
max_{P subseteq A_tau} T_b(P)
  = max_{P = P''} T_b(P).
```

Closed intents and extents are exactly the two sides of maximal bicliques in
the atom-witness incidence graph.

The skeleton is fixed before closure, so every optional intent is a legal
refinement of a connected pattern. The theorem does not permit an explicit
cap on the number of refinement atoms: closure could cross such a cap.
Interpretability is recovered after search by deleting closure atoms whose
removal preserves the extent, producing an irredundant generator.

## 3. Lossless twin compression

Stacked witnesses with the same atom transaction are false twins in the
incidence graph. Every pattern contains either all of them in its extent or
none. Replace each twin class `C` by one witness with vector weight

```text
w(C) = sum_{(b, phi) in C} omega(b, phi).
```

This preserves `S(P)` for every pattern, not only the optimum. Let `R_tau` be
the resulting set of distinct stacked witness signatures.

## 4. Witness-anchored exact search

The global atom powerset is the wrong baseline. Define the transaction width

```text
s_tau = max_{C in R_tau} |t(C)|.
```

Every pattern with nonempty extent matches at least one witness signature
`C`, and therefore is a subset of `t(C)`. Enumerating subsets inside each
compressed signature covers every supported pattern:

```text
union_{C in R_tau} powerset(t(C)).
```

Closing each generated subset and deduplicating closed intents gives an exact
algorithm with bound

```text
O*(R_tau * 2^s_tau),
```

instead of `2^|A_tau|`. The global label, type, and predicate vocabulary may
make `|A_tau|` large without increasing the number of atoms simultaneously
satisfied by one embedding.

For interpretable motifs, bound the size of a generator rather than the size
of its closure. Let

```text
C_tau,l = {G'' : G subseteq t(C), C in R_tau, |G| <= l}.
```

The resulting closed pattern `G''` may contain more than `l` atoms; only its
shortest searched explanation is bounded. Exact optimization over this
declared family takes

```text
O*(
  R_tau * sum_{j=0..l} binomial(s_tau, j)
)
  <= O*(R_tau * (e * s_tau / l)^l).
```

### Theorem 1: witness anchoring is complete

Every supported closed intent occurs in the unrestricted witness-anchored
search. Every closed intent having a generator of size at most `l` occurs in
the generator-bounded search.

Proof: choose any witness in the nonempty extent of the intent. The intent,
or its size-`l` generator, is contained in that witness transaction and is
therefore enumerated. Applying closure recovers the closed intent.

### Property-graph width guarantee

Transaction width is controlled by the local grammar rather than graph size.
For at most `K` roles, at most `L` simultaneously active labels per role,
`q` property slots of predicate-lattice depth at most `h`, one treatment
literal per role, and at most `E` relationship atoms per ordered role pair,

```text
s_tau <= K * (L + q*h + 1) + K*(K-1)*E.
```

Thus, for fixed local motif size and schema slots, unrestricted search is FPT
in `s_tau` and generator-bounded search is polynomial in the graph size and
global atom vocabulary. These quantities are measured directly from the
projected witness database, so the condition is checkable before search.

## 5. Weighted IPS optimization

For each assignment coordinate `b` and sign `s in {-1, +1}`, assign a
compressed witness scalar weight `s * w(C)[b]` and solve the maximum-weight
closed-biclique problem.
Atom vertices have weight zero. The better of the two signs gives
`max_P T_b(P)`.

Adapt the partition-oriented IPS recursion for maximal bicliques:

1. Maintain the standard branch `(S, C, X)`.
2. Apply structural closure and the randomization upper envelope.
3. If `X` is empty and the complement of the remaining incidence graph has
   maximum degree at most two, it is a disjoint union of paths and cycles.
4. Instead of outputting every maximal biclique in this terminal branch, run
   dynamic programming for a maximum-weight maximal independent set on those
   paths and cycles.
5. Otherwise use the IPS partition and pivot, branching on the pivot and its
   non-neighbors.

The `X = empty` condition is essential: without it, maximality inside the
terminal graph need not imply maximality against vertices excluded earlier in
the branch. The path/cycle DP carries constant boundary states recording
selection and domination, so maximality is enforced in linear time. A single
additional bit records whether a treatment atom has been selected.

### Theorem 2: exactness

Weighted IPS returns the same maximum `T_b` as exhaustive scoring over every
pattern in the fixed skeleton family.

Proof outline:

- Lemma 1 maps every pattern to a score-equivalent maximal biclique.
- IPS branches cover every maximal biclique exactly as in maximal-biclique
  enumeration.
- Whenever `X` is empty, the terminal DP optimizes over all maximal bicliques
  in the 2-biplex branch instead of materializing them. Other branches
  continue the IPS recursion.
- Taking both signs recovers the absolute statistic.

### Theorem 3: literal sub-two branching

Let

```text
N_tau = |A_tau| + |R_tau|
```

after twin compression. The worst IPS branching case is

```text
T(N) <= 2 T(N - 3) + T(N - 4).
```

Its branching factor is the largest positive root of

```text
x^4 - 2x - 1 = 0,
```

namely `alpha approximately 1.3954`. Because the terminal branch is solved by
one optimization DP rather than outputting every maximal biclique, there is
no maximal-biclique output-size term. For a fixed maximum role count `K`,

```text
O(
  2 * (B + 1) *
  sum_{k=2..K} sum_{tau in rooted_skeletons(k)}
  m_tau * alpha^(N_tau)
)
```

is an exact bound up to polynomial factors. The leading two is for the two
signs and does not affect the exponential base.

This is a base below two, not a replacement of `2^M` by `2^w`. The exponent is
the compressed incidence-instance size used by the branching algorithm.

### Corollary 2: exact hybrid without a global atom powerset

Let `A = |A_tau|`, `R = |R_tau|`, and `s = s_tau`. Choose the cheaper exact
solver for the unrestricted supported-pattern family:

```text
O*(min{R * 2^s, alpha^(A + R), alpha^gamma}).
```

IPS's inclusion-exclusion decomposition can also be carried over: every
maximal biclique is assigned to exactly one rooted incidence subinstance, and
optimization takes the maximum across subinstances. If

```text
gamma = max_v (|N_1(v)| + |N_2(v)| - 1)
```

in the twin-compressed incidence graph, the output-free optimization bound is

```text
O*(alpha^gamma).
```

The production solver selects witness anchoring, direct IPS, or decomposed
IPS from the measured `s`, `A`, `R`, and `gamma`. For generator length `l`,
the first term is replaced by
`R * sum_{j=0..l} binomial(s,j)`.

## 6. Topology and canonicality

Run the incidence optimization for each rooted connected wildcard skeleton up
to `K` roles. Every connected property-graph pattern contains such a skeleton.
Optional chords are refinement atoms. Canonical graph codes deduplicate a
pattern reached from multiple skeletons.

For fixed `K`, the number of skeletons is a parameter-only multiplier.
Projected embedding databases construct witness transactions without issuing
one GQL query per atom subset.

## 7. Randomization inference

Run the exact optimizer for the observed assignment and every conditional
randomization. The resulting maxima are identical to exhaustive maxima over
the declared count-exposure grammar, so the existing maxT adjusted p-value
calculation remains finite-sample exact under the sharp no-spillover null.

Because closure is formed on the stacked context rather than separately for
each assignment, a reported closed pattern also preserves its complete
randomization vector and therefore its individual randomization p-value.

The optimization can additionally use assignment-wise causal envelopes.
Those bounds only remove branches; they are not needed to obtain the
sub-two recurrence.

## 8. Claim boundary

The reduction requires:

- additive embedding-count or fixed-denominator motif-density exposure;
- a finite predicate vocabulary;
- a fixed maximum number of roles;
- connectedness supplied by a mandatory rooted skeleton;
- at least one selected treatment literal and one matched witness, enforced as
  constant-size DP/branch states;
- bounded generator length is allowed, but a hard cap on closure size is not.

It does not yet cover:

- binary existential exposure, where several witnesses of one root must be
  counted only once;
- pattern-dependent normalization;
- effect estimators whose denominator changes with the selected pattern;
- observational identification assumptions.

The executable `RandomizationFormalContext` is a correctness oracle for
vector-preserving closure, twin compression, and witness-anchored generator
search. The checked-in path/cycle terminal DP is exact. The remaining
production milestone is the full weighted IPS recursion.

With complementary literals available as grammar atoms, unrestricted
optimization is Maximum Monomial Agreement; even its monotone form is
NP-hard. The width or generator parameter is therefore substantive, rather
than cosmetic.

## 9. Algorithmic lineage

- Yuan, Altenburger, and Kooti,
  [Causal Network Motifs](https://arxiv.org/abs/2010.09911), WWW 2021:
  motif-based treatment-exposure features.
- Yu and Long,
  [FastQC](https://arxiv.org/abs/2305.14047), SIGMOD 2024: a demonstration
  that a stopping condition and its pivot must be co-designed to beat a
  powerset branching bound.
- Wang, Yu, and Long,
  [IPS](https://arxiv.org/abs/2602.21700), SIGMOD 2026: the
  partition-oriented maximal-biclique recursion and the `1.3954` branch bound
  adapted here from enumeration to causal optimization.
- Eckstein and Goldberg,
  [Maximum Monomial Agreement](https://doi.org/10.1287/ijoc.1110.0459),
  INFORMS Journal on Computing 2012: NP-hardness context and exact
  branch-and-bound for weighted conjunction discovery.
- Feldman,
  [Optimal Hardness Results for Maximizing Agreements with Monomials](https://doi.org/10.1109/CCC.2006.31),
  CCC 2006: hardness of monotone monomial agreement.
