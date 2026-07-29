from __future__ import annotations

import pytest

from causalscope.demo import build_demo_graph, build_demo_patterns
from causalscope.graph import PropertyGraph
from causalscope.search import PatternFamily


@pytest.fixture
def demo_graph() -> PropertyGraph:
    return build_demo_graph()


@pytest.fixture
def demo_family() -> PatternFamily:
    return build_demo_patterns()

