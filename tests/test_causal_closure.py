from __future__ import annotations

import math

from causalscope.incidence import (
    RandomizationFormalContext,
    VectorWeightedWitness,
    WeightedFormalContext,
    WeightedWitness,
    maximum_closed_intent_in_two_biplex,
)


def build_context() -> WeightedFormalContext:
    return WeightedFormalContext(
        atoms=("a", "b", "c"),
        witnesses=(
            WeightedWitness("w0", frozenset({"a", "b", "c"}), 3.0),
            WeightedWitness("w1", frozenset({"a", "b"}), -4.0),
            WeightedWitness("w2", frozenset({"a", "c"}), 2.0),
            WeightedWitness("w3", frozenset({"a"}), -1.0),
        ),
    )


def test_closure_preserves_extent_and_score() -> None:
    context = build_context()

    for intent in context.intents():
        closure = context.closure(intent)
        assert context.extent(closure) == context.extent(intent)
        assert context.score(closure) == context.score(intent)
        assert context.is_closed(closure)


def test_closed_search_matches_all_atom_subsets() -> None:
    context = build_context()

    exhaustive = context.maximum()
    closed = context.maximum(closed_only=True)

    assert closed.score == exhaustive.score
    assert context.is_closed(closed.intent)


def test_witness_weights_linearize_rooted_embedding_counts() -> None:
    residuals = (2.0, -3.0)
    matching_embedding_counts = (2, 1)
    context = WeightedFormalContext(
        atoms=("treated_triangle",),
        witnesses=(
            WeightedWitness("root0-embedding0", frozenset({"treated_triangle"}), 2.0),
            WeightedWitness("root0-embedding1", frozenset({"treated_triangle"}), 2.0),
            WeightedWitness("root1-embedding0", frozenset({"treated_triangle"}), -3.0),
        ),
    )

    direct = abs(
        sum(
            residual * count
            for residual, count in zip(residuals, matching_embedding_counts)
        )
    )
    assert context.score(frozenset({"treated_triangle"})) == direct


def test_twin_compression_preserves_every_intent_score() -> None:
    context = WeightedFormalContext(
        atoms=("a", "b", "c"),
        witnesses=(
            WeightedWitness("w0", frozenset({"a", "b"}), 2.5),
            WeightedWitness("w1", frozenset({"a", "b"}), -1.0),
            WeightedWitness("w2", frozenset({"a", "c"}), 4.0),
            WeightedWitness("w3", frozenset({"a", "c"}), -4.0),
        ),
    )

    compressed = context.compress_twins()

    assert len(compressed.witnesses) == 2
    for intent in context.intents():
        assert math.isclose(context.score(intent), compressed.score(intent))
    assert context.maximum().score == compressed.maximum().score


def test_two_biplex_optimizer_matches_closed_exhaustive_search() -> None:
    context = WeightedFormalContext(
        atoms=("a0", "a1", "a2"),
        witnesses=(
            WeightedWitness("w0", frozenset({"a1", "a2"}), 5.0),
            WeightedWitness("w1", frozenset({"a0", "a2"}), -8.0),
            WeightedWitness("w2", frozenset({"a0", "a1"}), 4.0),
        ),
    )

    optimized = maximum_closed_intent_in_two_biplex(context)
    exhaustive = context.maximum(closed_only=True)

    assert optimized.score == exhaustive.score
    assert context.is_closed(optimized.intent)


def test_two_biplex_optimizer_handles_empty_complement() -> None:
    context = WeightedFormalContext(
        atoms=("a", "b"),
        witnesses=(
            WeightedWitness("w0", frozenset({"a", "b"}), -2.0),
            WeightedWitness("w1", frozenset({"a", "b"}), 7.0),
        ),
    )

    optimized = maximum_closed_intent_in_two_biplex(context)

    assert optimized.score == context.maximum(closed_only=True).score


def test_stacked_closure_preserves_entire_randomization_vector() -> None:
    context = RandomizationFormalContext(
        atoms=("triangle", "treated-role-1", "treated-role-2"),
        witnesses=(
            VectorWeightedWitness(
                "assignment0-embedding0",
                frozenset({"triangle", "treated-role-1"}),
                (2.0, 0.0),
            ),
            VectorWeightedWitness(
                "assignment0-embedding1",
                frozenset({"triangle", "treated-role-1"}),
                (-3.0, 0.0),
            ),
            VectorWeightedWitness(
                "assignment1-embedding0",
                frozenset({"triangle", "treated-role-2"}),
                (0.0, 2.0),
            ),
        ),
    )
    intent = frozenset({"triangle"})
    closure = context.closure(intent)

    assert closure == intent
    assert context.extent(closure) == context.extent(intent)
    assert context.score_vector(closure) == context.score_vector(intent)


def test_vector_twin_compression_and_projection_are_lossless() -> None:
    context = RandomizationFormalContext(
        atoms=("a", "b"),
        witnesses=(
            VectorWeightedWitness("w0", frozenset({"a"}), (2.0, 0.0)),
            VectorWeightedWitness("w1", frozenset({"a"}), (-1.0, 4.0)),
            VectorWeightedWitness("w2", frozenset({"b"}), (0.0, -3.0)),
        ),
    )
    compressed = context.compress_twins()

    for intent in context.intents():
        assert compressed.score_vector(intent) == context.score_vector(intent)
        for coordinate in range(context.dimension):
            assert (
                compressed.project(coordinate).score(intent)
                == abs(context.score_vector(intent)[coordinate])
            )


def test_witness_anchored_search_finds_every_supported_closed_intent() -> None:
    context = RandomizationFormalContext(
        atoms=("a", "b", "c", "d"),
        witnesses=(
            VectorWeightedWitness("w0", frozenset({"a", "b", "c"}), (2.0,)),
            VectorWeightedWitness("w1", frozenset({"a", "b"}), (-4.0,)),
            VectorWeightedWitness("w2", frozenset({"a", "d"}), (3.0,)),
        ),
    )
    exhaustive = {
        intent
        for intent in context.intents()
        if context.extent(intent) and context.closure(intent) == intent
    }

    assert set(context.supported_closed_intents()) == exhaustive


def test_generator_cap_does_not_cap_closed_pattern_size() -> None:
    context = RandomizationFormalContext(
        atoms=("a", "b", "c"),
        witnesses=(
            VectorWeightedWitness("w0", frozenset({"a", "b", "c"}), (3.0,)),
            VectorWeightedWitness("w1", frozenset({"a", "b"}), (-1.0,)),
        ),
    )

    closed = context.supported_closed_intents(max_generator_size=1)

    assert frozenset({"a", "b", "c"}) in closed


def test_witness_anchored_maximum_matches_exhaustive_closed_search() -> None:
    context = RandomizationFormalContext(
        atoms=("a", "b", "c"),
        witnesses=(
            VectorWeightedWitness("w0", frozenset({"a", "b"}), (5.0, -2.0)),
            VectorWeightedWitness("w1", frozenset({"a", "c"}), (-8.0, 4.0)),
            VectorWeightedWitness("w2", frozenset({"a"}), (1.0, 3.0)),
        ),
    )

    for coordinate in range(context.dimension):
        actual = context.maximum_from_supported_generators(coordinate)
        expected = max(
            (
                abs(context.score_vector(intent)[coordinate])
                for intent in context.intents()
                if context.extent(intent)
            ),
            default=0.0,
        )
        assert actual.score == expected
