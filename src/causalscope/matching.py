"""Naive anchored matching used as a reference implementation."""

from __future__ import annotations

from causalscope.graph import PropertyGraph
from causalscope.pattern import RootedPattern


def has_rooted_embedding(
    graph: PropertyGraph,
    pattern: RootedPattern,
    unit: int,
    assignment: tuple[int, ...],
) -> bool:
    """Return whether ``pattern.root`` can be anchored at ``unit``."""

    if not pattern.nodes[pattern.root].matches(graph.node(unit), assignment):
        return False

    mapping: dict[int, int] = {pattern.root: unit}
    used = {unit}

    def edge_constraints_hold(pattern_node: int, graph_node: int) -> bool:
        for edge in pattern.edges:
            if (
                edge.source == pattern_node
                and edge.target in mapping
                and not graph.has_edge(graph_node, mapping[edge.target], edge.label)
            ):
                return False
            if (
                edge.target == pattern_node
                and edge.source in mapping
                and not graph.has_edge(mapping[edge.source], graph_node, edge.label)
            ):
                return False
        return True

    def choose_next_node() -> int | None:
        unmapped = [node_id for node_id in range(len(pattern.nodes)) if node_id not in mapping]
        if not unmapped:
            return None
        return max(
            unmapped,
            key=lambda node_id: sum(
                (edge.source == node_id and edge.target in mapping)
                or (edge.target == node_id and edge.source in mapping)
                for edge in pattern.edges
            ),
        )

    def candidates(pattern_node: int) -> tuple[int, ...]:
        constrained_sets: list[set[int]] = []
        for edge in pattern.edges:
            if edge.source == pattern_node and edge.target in mapping:
                constrained_sets.append(
                    set(graph.in_neighbors(mapping[edge.target], edge.label))
                )
            if edge.target == pattern_node and edge.source in mapping:
                constrained_sets.append(
                    set(graph.out_neighbors(mapping[edge.source], edge.label))
                )
        if not constrained_sets:
            return graph.node_ids
        intersection = set.intersection(*constrained_sets)
        return tuple(sorted(intersection))

    def search() -> bool:
        pattern_node = choose_next_node()
        if pattern_node is None:
            return True

        constraint = pattern.nodes[pattern_node]
        for graph_node in candidates(pattern_node):
            if graph_node in used:
                continue
            if not constraint.matches(graph.node(graph_node), assignment):
                continue
            if not edge_constraints_hold(pattern_node, graph_node):
                continue
            mapping[pattern_node] = graph_node
            used.add(graph_node)
            if search():
                return True
            used.remove(graph_node)
            del mapping[pattern_node]
        return False

    return search()


def exposure_vector(
    graph: PropertyGraph,
    pattern: RootedPattern,
    units: tuple[int, ...],
    assignment: tuple[int, ...],
) -> tuple[bool, ...]:
    return tuple(
        has_rooted_embedding(graph, pattern, unit, assignment)
        for unit in units
    )
