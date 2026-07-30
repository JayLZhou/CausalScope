"""Small correctness oracle for causal closure over embedding transactions."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from causalscope.biplex_dp import maximum_weight_maximal_independent_set


@dataclass(frozen=True)
class WeightedWitness:
    """One rooted embedding and its additive contribution to a motif-count score."""

    witness_id: str
    atoms: frozenset[str]
    weight: float


@dataclass(frozen=True)
class VectorWeightedWitness:
    """One assignment-embedding witness with randomization-vector weight."""

    witness_id: str
    atoms: frozenset[str]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class ClosedIntentResult:
    intent: frozenset[str]
    score: float


@dataclass(frozen=True)
class WeightedFormalContext:
    """A finite atom-by-embedding incidence context.

    This is deliberately an exponential reference implementation. Production
    search will optimize over maximal bicliques without materializing every
    atom subset.
    """

    atoms: tuple[str, ...]
    witnesses: tuple[WeightedWitness, ...]

    def __post_init__(self) -> None:
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("atoms must be unique")
        if len({witness.witness_id for witness in self.witnesses}) != len(
            self.witnesses
        ):
            raise ValueError("witness identifiers must be unique")
        atom_set = set(self.atoms)
        for witness in self.witnesses:
            unknown = witness.atoms - atom_set
            if unknown:
                raise ValueError(
                    f"witness {witness.witness_id} uses unknown atoms: "
                    f"{sorted(unknown)}"
                )

    def intents(self) -> tuple[frozenset[str], ...]:
        return tuple(
            frozenset(combination)
            for size in range(len(self.atoms) + 1)
            for combination in itertools.combinations(self.atoms, size)
        )

    def extent(self, intent: frozenset[str]) -> tuple[WeightedWitness, ...]:
        self._validate_intent(intent)
        return tuple(
            witness
            for witness in self.witnesses
            if intent.issubset(witness.atoms)
        )

    def closure(self, intent: frozenset[str]) -> frozenset[str]:
        """Return the Galois closure of an atom intent."""

        extent = self.extent(intent)
        if not extent:
            return frozenset(self.atoms)
        common = set(self.atoms)
        for witness in extent:
            common.intersection_update(witness.atoms)
        return frozenset(common)

    def is_closed(self, intent: frozenset[str]) -> bool:
        return self.closure(intent) == intent

    def score(self, intent: frozenset[str]) -> float:
        """Return the absolute additive embedding-count statistic."""

        return abs(sum(witness.weight for witness in self.extent(intent)))

    def maximum(self, *, closed_only: bool = False) -> ClosedIntentResult:
        candidates = (
            intent
            for intent in self.intents()
            if not closed_only or self.is_closed(intent)
        )
        return max(
            (
                ClosedIntentResult(intent=intent, score=self.score(intent))
                for intent in candidates
            ),
            key=lambda result: (result.score, -len(result.intent), sorted(result.intent)),
        )

    def compress_twins(self) -> WeightedFormalContext:
        """Merge witnesses with identical atom neighborhoods.

        Every intent contains either all witnesses in a twin class or none, so
        summing their weights preserves every intent score exactly.
        """

        grouped: dict[frozenset[str], list[WeightedWitness]] = {}
        for witness in self.witnesses:
            grouped.setdefault(witness.atoms, []).append(witness)

        compressed = tuple(
            WeightedWitness(
                witness_id="+".join(sorted(witness.witness_id for witness in group)),
                atoms=atoms,
                weight=sum(witness.weight for witness in group),
            )
            for atoms, group in sorted(
                grouped.items(),
                key=lambda item: tuple(sorted(item[0])),
            )
        )
        return WeightedFormalContext(self.atoms, compressed)

    def _validate_intent(self, intent: frozenset[str]) -> None:
        unknown = intent - set(self.atoms)
        if unknown:
            raise ValueError(f"intent uses unknown atoms: {sorted(unknown)}")


@dataclass(frozen=True)
class RandomizationFormalContext:
    """An assignment-stacked context that preserves a full score vector."""

    atoms: tuple[str, ...]
    witnesses: tuple[VectorWeightedWitness, ...]

    def __post_init__(self) -> None:
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("atoms must be unique")
        if len({witness.witness_id for witness in self.witnesses}) != len(
            self.witnesses
        ):
            raise ValueError("witness identifiers must be unique")
        dimensions = {len(witness.weights) for witness in self.witnesses}
        if len(dimensions) > 1:
            raise ValueError("all witness weight vectors must have equal length")
        atom_set = set(self.atoms)
        for witness in self.witnesses:
            unknown = witness.atoms - atom_set
            if unknown:
                raise ValueError(
                    f"witness {witness.witness_id} uses unknown atoms: "
                    f"{sorted(unknown)}"
                )

    @property
    def dimension(self) -> int:
        return len(self.witnesses[0].weights) if self.witnesses else 0

    def intents(self) -> tuple[frozenset[str], ...]:
        return tuple(
            frozenset(combination)
            for size in range(len(self.atoms) + 1)
            for combination in itertools.combinations(self.atoms, size)
        )

    def extent(
        self,
        intent: frozenset[str],
    ) -> tuple[VectorWeightedWitness, ...]:
        self._validate_intent(intent)
        return tuple(
            witness
            for witness in self.witnesses
            if intent.issubset(witness.atoms)
        )

    def closure(self, intent: frozenset[str]) -> frozenset[str]:
        extent = self.extent(intent)
        if not extent:
            return frozenset(self.atoms)
        common = set(self.atoms)
        for witness in extent:
            common.intersection_update(witness.atoms)
        return frozenset(common)

    def score_vector(self, intent: frozenset[str]) -> tuple[float, ...]:
        totals = [0.0] * self.dimension
        for witness in self.extent(intent):
            for coordinate, weight in enumerate(witness.weights):
                totals[coordinate] += weight
        return tuple(totals)

    def compress_twins(self) -> RandomizationFormalContext:
        grouped: dict[frozenset[str], list[VectorWeightedWitness]] = {}
        for witness in self.witnesses:
            grouped.setdefault(witness.atoms, []).append(witness)

        compressed = tuple(
            VectorWeightedWitness(
                witness_id="+".join(sorted(witness.witness_id for witness in group)),
                atoms=atoms,
                weights=tuple(
                    sum(witness.weights[index] for witness in group)
                    for index in range(self.dimension)
                ),
            )
            for atoms, group in sorted(
                grouped.items(),
                key=lambda item: tuple(sorted(item[0])),
            )
        )
        return RandomizationFormalContext(self.atoms, compressed)

    def project(self, coordinate: int) -> WeightedFormalContext:
        if not 0 <= coordinate < self.dimension:
            raise IndexError("coordinate is outside the score vector")
        return WeightedFormalContext(
            self.atoms,
            tuple(
                WeightedWitness(
                    witness_id=witness.witness_id,
                    atoms=witness.atoms,
                    weight=witness.weights[coordinate],
                )
                for witness in self.witnesses
            ),
        )

    def _validate_intent(self, intent: frozenset[str]) -> None:
        unknown = intent - set(self.atoms)
        if unknown:
            raise ValueError(f"intent uses unknown atoms: {sorted(unknown)}")


def maximum_closed_intent_in_two_biplex(
    context: WeightedFormalContext,
) -> ClosedIntentResult:
    """Optimize the absolute closed-intent score in a 2-biplex context.

    The bipartite complement has an edge exactly when an atom is absent from a
    witness transaction. A maximal biclique in the incidence graph is a
    maximal independent set in this complement.
    """

    atom_vertices = {
        atom: f"atom:{index}" for index, atom in enumerate(context.atoms)
    }
    witness_vertices = {
        witness.witness_id: f"witness:{index}"
        for index, witness in enumerate(context.witnesses)
    }
    vertices = tuple(atom_vertices.values()) + tuple(witness_vertices.values())
    complement_edges = tuple(
        (atom_vertices[atom], witness_vertices[witness.witness_id])
        for atom in context.atoms
        for witness in context.witnesses
        if atom not in witness.atoms
    )

    candidates: list[ClosedIntentResult] = []
    for sign in (-1.0, 1.0):
        weights = {vertex: 0.0 for vertex in atom_vertices.values()}
        weights.update(
            {
                witness_vertices[witness.witness_id]: sign * witness.weight
                for witness in context.witnesses
            }
        )
        solution = maximum_weight_maximal_independent_set(
            vertices,
            complement_edges,
            weights,
        )
        intent = frozenset(
            atom
            for atom, vertex in atom_vertices.items()
            if vertex in solution.vertices
        )
        candidates.append(
            ClosedIntentResult(intent=intent, score=context.score(intent))
        )

    return max(
        candidates,
        key=lambda result: (
            result.score,
            -len(result.intent),
            tuple(sorted(result.intent)),
        ),
    )
