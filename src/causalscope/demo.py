"""Small end-to-end demonstration."""

from __future__ import annotations

from causalscope.graph import PropertyGraph
from causalscope.matching import exposure_vector
from causalscope.pattern import NodeConstraint, RootedPattern
from causalscope.randomization import BernoulliDesign
from causalscope.search import (
    PatternFamily,
    brute_force_randomization_maxima,
    discover_significant_patterns,
    randomization_max_search,
)
from causalscope.statistics import center_outcomes_within_treatment


def build_demo_graph() -> PropertyGraph:
    graph = PropertyGraph()
    for node_id in range(8):
        graph.add_node(node_id, "User", {"segment": "A" if node_id < 4 else "B"})
    for left, right in ((0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
        graph.add_undirected_edge(left, right, "KNOWS")
    for left, right in ((0, 4), (1, 5), (2, 6), (3, 7)):
        graph.add_undirected_edge(left, right, "WORKS_WITH")
    return graph


def build_demo_patterns() -> PatternFamily:
    root = RootedPattern("user", (NodeConstraint(label="User"),))
    treated_friend = root.add_node(
        name="treated_friend",
        attach_to=0,
        edge_label="KNOWS",
        constraint=NodeConstraint(label="User", treatment=1),
    )
    treated_friend_chain = treated_friend.add_node(
        name="treated_friend_chain",
        attach_to=1,
        edge_label="KNOWS",
        constraint=NodeConstraint(label="User", treatment=1),
    )
    two_treated_friends = treated_friend.add_node(
        name="two_treated_friends",
        attach_to=0,
        edge_label="KNOWS",
        constraint=NodeConstraint(label="User", treatment=1),
    )
    treated_coworker = root.add_node(
        name="treated_coworker",
        attach_to=0,
        edge_label="WORKS_WITH",
        constraint=NodeConstraint(label="User", treatment=1),
    )
    impossible_purchase = root.add_node(
        name="impossible_purchase",
        attach_to=0,
        edge_label="PURCHASED",
        constraint=NodeConstraint(label="Product", treatment=1),
    )
    impossible_purchase_child = impossible_purchase.add_node(
        name="impossible_purchase_child",
        attach_to=1,
        edge_label="RELATED_TO",
        constraint=NodeConstraint(label="Product"),
    )
    patterns = {
        pattern.name: pattern
        for pattern in (
            root,
            treated_friend,
            treated_friend_chain,
            two_treated_friends,
            treated_coworker,
            impossible_purchase,
            impossible_purchase_child,
        )
    }
    return PatternFamily(
        patterns=patterns,
        children={
            "user": ("treated_friend", "treated_coworker", "impossible_purchase"),
            "treated_friend": ("treated_friend_chain", "two_treated_friends"),
            "impossible_purchase": ("impossible_purchase_child",),
        },
        roots=("user",),
    )


def main() -> None:
    graph = build_demo_graph()
    family = build_demo_patterns()
    observed = (0, 1, 1, 0, 1, 0, 0, 1)
    focal = (0, 2, 3, 5, 6, 7)
    outcomes = (0.2, 2.7, 3.1, 0.4, 2.2, 0.3, 0.1, 2.5)
    residuals = center_outcomes_within_treatment(outcomes, observed, focal)
    assignments = BernoulliDesign.constant(len(graph.node_ids), 0.5).conditional_samples(
        observed,
        focal,
        draws=199,
        seed=7,
    )

    def exposures(pattern: RootedPattern, assignment: tuple[int, ...]) -> tuple[bool, ...]:
        return exposure_vector(graph, pattern, focal, assignment)

    exact = brute_force_randomization_maxima(family, assignments, residuals, exposures)
    pruned = randomization_max_search(family, assignments, residuals, exposures)
    assert pruned.maxima == exact

    discovery = discover_significant_patterns(
        family,
        observed,
        residuals,
        exposures,
        exact,
        alpha=0.1,
    )
    print("CausalScope correctness demo")
    print(f"patterns visited in max search: {len(pruned.visited_patterns)}")
    print(f"subtrees pruned in max search: {pruned.pruned_roots}")
    print(f"significant patterns at alpha=0.1: {discovery.significant_patterns}")


if __name__ == "__main__":
    main()

