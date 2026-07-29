from __future__ import annotations

import itertools

from causalscope.graph import PropertyGraph
from causalscope.matching import exposure_vector
from causalscope.search import PatternFamily
from causalscope.statistics import linear_statistic, subtree_envelope


def test_parent_envelope_bounds_every_descendant_statistic(
    demo_graph: PropertyGraph,
    demo_family: PatternFamily,
) -> None:
    units = demo_graph.node_ids
    residuals = (2.0, -1.0, 0.5, -3.0, 1.5, -0.25, 0.75, -0.5)

    for assignment in itertools.product((0, 1), repeat=len(units)):
        for parent_name, child_names in demo_family.children.items():
            parent_exposure = exposure_vector(
                demo_graph,
                demo_family.patterns[parent_name],
                units,
                assignment,
            )
            bound = subtree_envelope(residuals, parent_exposure)
            for child_name in child_names:
                child_exposure = exposure_vector(
                    demo_graph,
                    demo_family.patterns[child_name],
                    units,
                    assignment,
                )
                assert linear_statistic(residuals, child_exposure) <= bound

