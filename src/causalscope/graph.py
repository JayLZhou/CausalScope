"""Minimal directed property-graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


class PropertyGraph:
    """A small in-memory property graph used by the correctness prototype."""

    def __init__(self) -> None:
        self._nodes: dict[int, GraphNode] = {}
        self._out_edges: dict[int, list[GraphEdge]] = {}
        self._in_edges: dict[int, list[GraphEdge]] = {}

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def add_node(
        self,
        node_id: int,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if node_id in self._nodes:
            raise ValueError(f"node {node_id} already exists")
        node = GraphNode(node_id, label, dict(properties or {}))
        self._nodes[node_id] = node
        self._out_edges[node_id] = []
        self._in_edges[node_id] = []

    def add_edge(
        self,
        source: int,
        target: int,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("both edge endpoints must exist")
        edge = GraphEdge(source, target, label, dict(properties or {}))
        self._out_edges[source].append(edge)
        self._in_edges[target].append(edge)

    def add_undirected_edge(
        self,
        left: int,
        right: int,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.add_edge(left, right, label, properties)
        self.add_edge(right, left, label, properties)

    def node(self, node_id: int) -> GraphNode:
        return self._nodes[node_id]

    def has_edge(self, source: int, target: int, label: str) -> bool:
        return any(
            edge.target == target and edge.label == label
            for edge in self._out_edges[source]
        )

