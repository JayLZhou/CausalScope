from __future__ import annotations

import pytest

from causalscope.pattern import NodeConstraint, RootedPattern
from causalscope.search import PatternFamily


def test_pattern_family_rejects_non_monotone_child() -> None:
    parent = RootedPattern(
        "treated_user",
        (NodeConstraint(label="User", treatment=1),),
    )
    child = RootedPattern(
        "relaxed_user",
        (NodeConstraint(label="User"),),
    )

    with pytest.raises(ValueError, match="not a monotone extension"):
        PatternFamily(
            patterns={parent.name: parent, child.name: child},
            children={parent.name: (child.name,)},
            roots=(parent.name,),
        )


def test_pattern_family_rejects_unreachable_pattern() -> None:
    root = RootedPattern("root", (NodeConstraint(label="User"),))
    orphan = RootedPattern("orphan", (NodeConstraint(label="User"),))

    with pytest.raises(ValueError, match="unreachable"):
        PatternFamily(
            patterns={root.name: root, orphan.name: orphan},
            children={},
            roots=(root.name,),
        )

