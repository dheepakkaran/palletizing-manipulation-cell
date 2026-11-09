"""Case types for the mixed-case palletizing cell (simulation only, made-up but plausible values).

size = full box dimensions (x, y, z) in metres, mass in kg, crush_kg = the most mass
(kg) that may rest on top of this case. Masses stay at or below the Panda's 3 kg payload.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseType:
    name: str
    size: tuple[float, float, float]
    mass: float
    crush_kg: float
    rgba: tuple[float, float, float, float]


CASE_TYPES = {
    "heavy": CaseType("heavy", (0.13, 0.13, 0.10), 2.8, 6.0, (0.55, 0.35, 0.18, 1)),
    "medium": CaseType("medium", (0.13, 0.13, 0.10), 1.8, 4.0, (0.80, 0.65, 0.40, 1)),
    "light": CaseType("light", (0.12, 0.12, 0.08), 1.0, 1.5, (0.25, 0.45, 0.80, 1)),
    "fragile": CaseType("fragile", (0.12, 0.12, 0.08), 0.6, 0.5, (0.90, 0.30, 0.35, 1)),
}
COPIES = 3  # bodies per type in the scene -> 12 case bodies


def case_bodies() -> list[tuple[str, CaseType]]:
    return [(f"{t}_{i}", ct) for t, ct in CASE_TYPES.items() for i in range(COPIES)]
