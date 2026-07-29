"""Rooted property-graph exposure patterns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from causalscope.graph import GraphNode


@dataclass(frozen=True)
class NodeConstraint:
    """Conjunctive constraints for one pattern role."""

    label: str | None = None
    properties: tuple[tuple[str, Any], ...] = ()
    treatment: int | None = None

    def is_refinement_of(self, parent: NodeConstraint) -> bool:
        if parent.label is not None and self.label != parent.label:
            return False
        if parent.treatment is not None and self.treatment != parent.treatment:
            return False
        child_properties = dict(self.properties)
        return all(
            child_properties.get(key) == value
            for key, value in parent.properties
        )

    def matches(self, node: GraphNode, assignment: tuple[int, ...]) -> bool:
        if self.label is not None and node.label != self.label:
            return False
        if self.treatment is not None and assignment[node.node_id] != self.treatment:
            return False
        return all(node.properties.get(key) == value for key, value in self.properties)


@dataclass(frozen=True, order=True)
class PatternEdge:
    source: int
    target: int
    label: str


@dataclass(frozen=True)
class RootedPattern:
    """A connected pattern whose role zero is anchored to the unit."""

    name: str
    nodes: tuple[NodeConstraint, ...]
    edges: tuple[PatternEdge, ...] = ()
    root: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a pattern must contain at least one node")
        if self.root < 0 or self.root >= len(self.nodes):
            raise ValueError("root must index a pattern node")
        for edge in self.edges:
            if (
                edge.source < 0
                or edge.target < 0
                or edge.source >= len(self.nodes)
                or edge.target >= len(self.nodes)
            ):
                raise ValueError("pattern edge endpoint is out of range")
        reached = {self.root}
        while True:
            expanded = reached | {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source, edge.target)
                if edge.source in reached or edge.target in reached
            }
            if expanded == reached:
                break
            reached = expanded
        if len(reached) != len(self.nodes):
            raise ValueError("a rooted pattern must be connected")

    @property
    def size(self) -> int:
        return len(self.nodes) + len(self.edges)

    def is_monotone_extension_of(self, parent: RootedPattern) -> bool:
        """Check the syntactic condition that guarantees nested exposure events."""

        if self.root != parent.root or len(self.nodes) < len(parent.nodes):
            return False
        if not all(
            self.nodes[node_id].is_refinement_of(parent_constraint)
            for node_id, parent_constraint in enumerate(parent.nodes)
        ):
            return False
        return set(parent.edges).issubset(self.edges)

    def add_node(
        self,
        *,
        name: str,
        attach_to: int,
        edge_label: str,
        constraint: NodeConstraint,
        outgoing: bool = True,
    ) -> RootedPattern:
        if attach_to < 0 or attach_to >= len(self.nodes):
            raise ValueError("attach_to is out of range")
        new_id = len(self.nodes)
        edge = (
            PatternEdge(attach_to, new_id, edge_label)
            if outgoing
            else PatternEdge(new_id, attach_to, edge_label)
        )
        return RootedPattern(
            name=name,
            nodes=self.nodes + (constraint,),
            edges=self.edges + (edge,),
            root=self.root,
        )

    def require_treatment(
        self,
        *,
        name: str,
        node_id: int,
        treatment: int,
    ) -> RootedPattern:
        if treatment not in (0, 1):
            raise ValueError("treatment must be zero or one")
        current = self.nodes[node_id]
        if current.treatment is not None and current.treatment != treatment:
            raise ValueError("cannot replace an existing treatment literal")
        nodes = list(self.nodes)
        nodes[node_id] = replace(current, treatment=treatment)
        return RootedPattern(name=name, nodes=tuple(nodes), edges=self.edges, root=self.root)
