"""Known-design assignment generation."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BernoulliDesign:
    probabilities: tuple[float, ...]

    @classmethod
    def constant(cls, size: int, probability: float) -> BernoulliDesign:
        return cls((probability,) * size)

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("the design must contain at least one unit")
        if any(probability < 0.0 or probability > 1.0 for probability in self.probabilities):
            raise ValueError("assignment probabilities must lie in [0, 1]")

    def sample(
        self,
        rng: random.Random,
        fixed: Mapping[int, int] | None = None,
    ) -> tuple[int, ...]:
        fixed_assignments = dict(fixed or {})
        for node_id, treatment in fixed_assignments.items():
            if node_id < 0 or node_id >= len(self.probabilities):
                raise IndexError("fixed assignment index is out of range")
            if treatment not in (0, 1):
                raise ValueError("fixed treatment must be zero or one")
        return tuple(
            fixed_assignments.get(node_id, int(rng.random() < probability))
            for node_id, probability in enumerate(self.probabilities)
        )

    def conditional_samples(
        self,
        observed: tuple[int, ...],
        focal_units: tuple[int, ...],
        *,
        draws: int,
        seed: int,
    ) -> tuple[tuple[int, ...], ...]:
        if len(observed) != len(self.probabilities):
            raise ValueError("observed assignment has the wrong size")
        if draws <= 0:
            raise ValueError("draws must be positive")
        fixed = {unit: observed[unit] for unit in focal_units}
        rng = random.Random(seed)
        return tuple(self.sample(rng, fixed) for _ in range(draws))
