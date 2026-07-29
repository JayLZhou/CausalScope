from __future__ import annotations

import itertools

from causalscope.graph import PropertyGraph
from causalscope.matching import exposure_vector
from causalscope.search import PatternFamily


def test_every_child_exposure_is_nested_in_parent(
    demo_graph: PropertyGraph,
    demo_family: PatternFamily,
) -> None:
    units = demo_graph.node_ids
    assignments = itertools.product((0, 1), repeat=len(units))

    for assignment in assignments:
        for parent_name, child_names in demo_family.children.items():
            parent = exposure_vector(
                demo_graph,
                demo_family.patterns[parent_name],
                units,
                assignment,
            )
            for child_name in child_names:
                child = exposure_vector(
                    demo_graph,
                    demo_family.patterns[child_name],
                    units,
                    assignment,
                )
                assert all(not child_value or parent_value for parent_value, child_value in zip(parent, child))

