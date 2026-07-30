from __future__ import annotations

import itertools
import random

import pytest

from causalscope.biplex_dp import (
    IndependentSetResult,
    maximum_weight_maximal_independent_set,
)


def brute_force(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    weights: dict[str, float],
) -> IndependentSetResult:
    edge_sets = tuple(frozenset(edge) for edge in edges)
    candidates: list[IndependentSetResult] = []
    for size in range(len(vertices) + 1):
        for combination in itertools.combinations(vertices, size):
            selected = frozenset(combination)
            if any(edge <= selected for edge in edge_sets):
                continue
            if any(
                vertex not in selected
                and not any(
                    frozenset({vertex, neighbor}) in edge_sets
                    for neighbor in selected
                )
                for vertex in vertices
            ):
                continue
            candidates.append(
                IndependentSetResult(
                    selected,
                    sum(weights[vertex] for vertex in selected),
                )
            )
    return max(
        candidates,
        key=lambda result: (
            result.weight,
            -len(result.vertices),
            tuple(sorted(result.vertices)),
        ),
    )


@pytest.mark.parametrize("is_cycle", [False, True])
def test_terminal_dp_matches_exhaustive_search(is_cycle: bool) -> None:
    generator = random.Random(17 if is_cycle else 11)
    minimum_size = 3 if is_cycle else 1

    for size in range(minimum_size, 10):
        vertices = tuple(f"v{index}" for index in range(size))
        edges = tuple(
            (vertices[index], vertices[index + 1])
            for index in range(size - 1)
        )
        if is_cycle:
            edges += ((vertices[-1], vertices[0]),)

        for _ in range(30):
            weights = {
                vertex: float(generator.randint(-7, 7))
                for vertex in vertices
            }
            actual = maximum_weight_maximal_independent_set(
                vertices,
                edges,
                weights,
            )
            expected = brute_force(vertices, edges, weights)
            assert actual == expected


def test_disconnected_paths_are_optimized_independently() -> None:
    vertices = ("a", "b", "c", "x", "y")
    edges = (("a", "b"), ("b", "c"), ("x", "y"))
    weights = {"a": -2.0, "b": 4.0, "c": -2.0, "x": 3.0, "y": 1.0}

    actual = maximum_weight_maximal_independent_set(vertices, edges, weights)

    assert actual == IndependentSetResult(frozenset({"b", "x"}), 7.0)


def test_non_terminal_graph_is_rejected() -> None:
    vertices = ("center", "a", "b", "c")
    edges = (("center", "a"), ("center", "b"), ("center", "c"))
    weights = {vertex: 0.0 for vertex in vertices}

    with pytest.raises(ValueError, match="maximum degree"):
        maximum_weight_maximal_independent_set(vertices, edges, weights)
