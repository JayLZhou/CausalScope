"""Linear-time terminal solver for a 2-biplex complement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndependentSetResult:
    """A maximum-weight maximal independent set."""

    vertices: frozenset[str]
    weight: float


def maximum_weight_maximal_independent_set(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    weights: dict[str, float],
) -> IndependentSetResult:
    """Optimize a maximal independent set when every degree is at most two.

    A graph of maximum degree two is a disjoint union of paths and cycles.
    Maximal independent sets are exactly independent dominating sets, which
    can be optimized with constant-state dynamic programming on each component.
    """

    if len(set(vertices)) != len(vertices):
        raise ValueError("vertices must be unique")
    vertex_set = set(vertices)
    if set(weights) != vertex_set:
        raise ValueError("weights must contain exactly the supplied vertices")

    adjacency = {vertex: set() for vertex in vertices}
    for left, right in edges:
        if left not in vertex_set or right not in vertex_set:
            raise ValueError("edges must use supplied vertices")
        if left == right:
            raise ValueError("self-loops are not supported")
        adjacency[left].add(right)
        adjacency[right].add(left)

    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError("terminal graph must have maximum degree at most two")

    selected: set[str] = set()
    score = 0.0
    unseen = set(vertices)
    while unseen:
        component = _take_component(min(unseen), adjacency)
        unseen.difference_update(component)
        order, is_cycle = _component_order(component, adjacency)
        result = (
            _solve_cycle(order, weights)
            if is_cycle
            else _solve_path(order, weights)
        )
        selected.update(result.vertices)
        score += result.weight

    return IndependentSetResult(frozenset(selected), score)


def _take_component(
    start: str,
    adjacency: dict[str, set[str]],
) -> set[str]:
    component: set[str] = set()
    pending = [start]
    while pending:
        vertex = pending.pop()
        if vertex in component:
            continue
        component.add(vertex)
        pending.extend(adjacency[vertex] - component)
    return component


def _component_order(
    component: set[str],
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], bool]:
    is_cycle = len(component) >= 3 and all(
        len(adjacency[vertex]) == 2 for vertex in component
    )
    if is_cycle:
        start = min(component)
    else:
        endpoints = [
            vertex for vertex in component if len(adjacency[vertex]) <= 1
        ]
        start = min(endpoints)

    order: list[str] = []
    previous: str | None = None
    current = start
    while True:
        order.append(current)
        candidates = adjacency[current] - ({previous} if previous else set())
        if is_cycle:
            candidates.discard(start)
        if not candidates:
            break
        following = min(candidates)
        previous, current = current, following

    if len(order) != len(component):
        raise ValueError("component is not a simple path or cycle")
    return tuple(order), is_cycle


def _solve_path(
    order: tuple[str, ...],
    weights: dict[str, float],
) -> IndependentSetResult:
    if not order:
        return IndependentSetResult(frozenset(), 0.0)

    states: dict[tuple[int, int], IndependentSetResult] = {}
    for first in (0, 1):
        chosen = frozenset({order[0]}) if first else frozenset()
        states[(0, first)] = IndependentSetResult(
            chosen,
            weights[order[0]] if first else 0.0,
        )

    for vertex in order[1:]:
        next_states: dict[tuple[int, int], IndependentSetResult] = {}
        for (before_previous, previous), result in states.items():
            for current in (0, 1):
                if previous and current:
                    continue
                if not previous and not (before_previous or current):
                    continue
                candidate = _extend(result, vertex, current, weights)
                _keep_better(next_states, (previous, current), candidate)
        states = next_states

    feasible = [
        result
        for (before_previous, previous), result in states.items()
        if previous or before_previous
    ]
    return max(feasible, key=_result_key)


def _solve_cycle(
    order: tuple[str, ...],
    weights: dict[str, float],
) -> IndependentSetResult:
    if len(order) < 3:
        raise ValueError("a simple cycle must contain at least three vertices")

    feasible: list[IndependentSetResult] = []
    for first in (0, 1):
        for second in (0, 1):
            if first and second:
                continue
            chosen = {
                vertex
                for vertex, take in zip(order[:2], (first, second))
                if take
            }
            states = {
                (first, second): IndependentSetResult(
                    frozenset(chosen),
                    first * weights[order[0]] + second * weights[order[1]],
                )
            }
            for vertex in order[2:]:
                next_states: dict[tuple[int, int], IndependentSetResult] = {}
                for (before_previous, previous), result in states.items():
                    for current in (0, 1):
                        if previous and current:
                            continue
                        if not previous and not (before_previous or current):
                            continue
                        candidate = _extend(result, vertex, current, weights)
                        _keep_better(
                            next_states,
                            (previous, current),
                            candidate,
                        )
                states = next_states

            for (before_last, last), result in states.items():
                if last and first:
                    continue
                if not last and not (before_last or first):
                    continue
                if not first and not (last or second):
                    continue
                feasible.append(result)

    return max(feasible, key=_result_key)


def _extend(
    result: IndependentSetResult,
    vertex: str,
    take: int,
    weights: dict[str, float],
) -> IndependentSetResult:
    if not take:
        return result
    return IndependentSetResult(
        result.vertices | {vertex},
        result.weight + weights[vertex],
    )


def _keep_better(
    states: dict[tuple[int, int], IndependentSetResult],
    state: tuple[int, int],
    candidate: IndependentSetResult,
) -> None:
    incumbent = states.get(state)
    if incumbent is None or _result_key(candidate) > _result_key(incumbent):
        states[state] = candidate


def _result_key(result: IndependentSetResult) -> tuple[float, int, tuple[str, ...]]:
    return (
        result.weight,
        -len(result.vertices),
        tuple(sorted(result.vertices)),
    )
