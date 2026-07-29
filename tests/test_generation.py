from __future__ import annotations

from causalscope.generation import generate_one_hop_treated_patterns
from causalscope.graph import PropertyGraph


def test_one_hop_generator_discovers_edge_types_from_graph(
    demo_graph: PropertyGraph,
) -> None:
    family = generate_one_hop_treated_patterns(demo_graph, root_label="User")

    assert set(family.patterns) == {
        "root:User",
        "User-[KNOWS]->User:z=1",
        "User-[WORKS_WITH]->User:z=1",
    }
    assert family.children["root:User"] == (
        "User-[KNOWS]->User:z=1",
        "User-[WORKS_WITH]->User:z=1",
    )

