"""Small automatic generators that will grow into the canonical DFS engine."""

from __future__ import annotations

from causalscope.graph import PropertyGraph
from causalscope.pattern import NodeConstraint, RootedPattern
from causalscope.search import PatternFamily


def generate_one_hop_treated_patterns(
    graph: PropertyGraph,
    *,
    root_label: str,
    treatment: int = 1,
) -> PatternFamily:
    """Discover all typed one-hop treated-neighbor patterns in ``graph``.

    The schema signatures are read from the graph rather than supplied by the
    caller. This is the first executable automatic-mining baseline; canonical
    multi-hop DFS growth will replace it without changing ``PatternFamily``.
    """

    if treatment not in (0, 1):
        raise ValueError("treatment must be zero or one")

    signatures = {
        (edge.label, graph.node(edge.target).label)
        for source in graph.node_ids
        if graph.node(source).label == root_label
        for edge in graph.out_edges(source)
    }
    root_name = f"root:{root_label}"
    root = RootedPattern(root_name, (NodeConstraint(label=root_label),))
    patterns = {root_name: root}
    child_names: list[str] = []

    for edge_label, target_label in sorted(signatures):
        name = f"{root_label}-[{edge_label}]->{target_label}:z={treatment}"
        child = root.add_node(
            name=name,
            attach_to=0,
            edge_label=edge_label,
            constraint=NodeConstraint(label=target_label, treatment=treatment),
        )
        patterns[name] = child
        child_names.append(name)

    return PatternFamily(
        patterns=patterns,
        children={root_name: tuple(child_names)},
        roots=(root_name,),
    )

