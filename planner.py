"""Layer planner for mixed-case pallets + a first-fit baseline + a rule checker.

Pallet = 2 x 3 grid of columns (0.15 m pitch), up to 3 layers. Each case sits on exactly one case
below (or on the pallet). Rules (each break = one violation):
  crush       : mass resting on a case > its crush_kg
  heavy_on_light : a case is heavier than the case right under it
  uneven_layer: cases in the same layer have different heights (layer top is not flat)
  overflow    : no column left (more than 3 layers)
The planner puts tall, strong, heavy cases low and only stacks a case where every case under it can
carry the extra mass. The first-fit baseline fills columns layer by layer in arrival order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from build_scene import PALLET_CENTER, PALLET_TOP
from cases import CASE_TYPES

PITCH = 0.15
NX, NY = 2, 3
MAX_LAYERS = 3
CELLS = [(ix, iy) for iy in range(NY) for ix in range(NX)]


def cell_xy(cell) -> np.ndarray:
    ix, iy = cell
    return np.array([PALLET_CENTER[0] + (ix - (NX - 1) / 2) * PITCH,
                     PALLET_CENTER[1] + (iy - (NY - 1) / 2) * PITCH])


def ctype(case_id: str):
    return CASE_TYPES[case_id.rsplit("_", 1)[0]]


@dataclass
class Pallet:
    """Columns of case ids, bottom first."""
    cols: dict = field(default_factory=lambda: {c: [] for c in CELLS})

    def copy(self) -> "Pallet":
        return Pallet({c: list(v) for c, v in self.cols.items()})

    def layer_heights(self, k: int) -> set:
        return {ctype(v[k]).size[2] for v in self.cols.values() if len(v) > k}

    def place_z(self, cell) -> float:
        """Bottom z of the next case in this column."""
        return PALLET_TOP + sum(ctype(c).size[2] for c in self.cols[cell])

    def add(self, case_id, cell):
        self.cols[cell].append(case_id)


@dataclass
class Placement:
    case: str
    cell: tuple
    layer: int
    z_bottom: float

    @property
    def target(self) -> np.ndarray:
        xy = cell_xy(self.cell)
        return np.array([xy[0], xy[1], self.z_bottom + ctype(self.case).size[2] / 2])


def violations(p: Pallet) -> dict:
    v = {"crush": 0, "heavy_on_light": 0, "uneven_layer": 0, "overflow": 0}
    for col in p.cols.values():
        for i, cid in enumerate(col):
            above = sum(ctype(c).mass for c in col[i + 1:])
            if above > ctype(cid).crush_kg + 1e-9:
                v["crush"] += 1
            if i > 0 and ctype(cid).mass > ctype(col[i - 1]).mass + 1e-9:
                v["heavy_on_light"] += 1
        if len(col) > MAX_LAYERS:
            v["overflow"] += len(col) - MAX_LAYERS
    for k in range(MAX_LAYERS + 3):
        if len(p.layer_heights(k)) > 1:
            v["uneven_layer"] += 1
    v["total"] = sum(v.values())
    return v


def _ok(p: Pallet, cid: str, cell) -> bool:
    col = p.cols[cell]
    k = len(col)
    if k >= MAX_LAYERS:
        return False
    ct = ctype(cid)
    hs = p.layer_heights(k)
    if hs and ct.size[2] not in hs:
        return False
    if k > 0 and ct.mass > ctype(col[-1]).mass:
        return False
    for i, below in enumerate(col):
        load = sum(ctype(c).mass for c in col[i + 1:]) + ct.mass
        if load > ctype(below).crush_kg:
            return False
    return True


def plan(case_ids, start: Pallet | None = None) -> tuple[list[Placement], Pallet]:
    """Rule-aware greedy layer planner. `start` = cases already on the pallet (for replanning)."""
    p = start.copy() if start is not None else Pallet()
    order = sorted(case_ids, key=lambda c: (-ctype(c).size[2], -ctype(c).crush_kg, -ctype(c).mass, c))
    out = []
    for cid in order:
        cands = [c for c in CELLS if _ok(p, cid, c)]
        if not cands:  # soft fallback: lowest column, rules may break (counted by the checker)
            cands = sorted(CELLS, key=lambda c: len(p.cols[c]))[:1]
        cell = min(cands, key=lambda c: (len(p.cols[c]), CELLS.index(c)))
        out.append(Placement(cid, cell, len(p.cols[cell]), p.place_z(cell)))
        p.add(cid, cell)
    return out, p


def first_fit(case_ids, start: Pallet | None = None) -> tuple[list[Placement], Pallet]:
    """Baseline: arrival order, next free spot filling layer by layer, no rules."""
    p = start.copy() if start is not None else Pallet()
    out = []
    for cid in case_ids:
        cell = min(CELLS, key=lambda c: (len(p.cols[c]), CELLS.index(c)))
        out.append(Placement(cid, cell, len(p.cols[cell]), p.place_z(cell)))
        p.add(cid, cell)
    return out, p


def random_order(rng, n_min=7, n_max=10) -> list[str]:
    from cases import case_bodies
    ids = [n for n, _ in case_bodies()]
    n = int(rng.integers(n_min, n_max + 1))
    return [str(x) for x in rng.choice(ids, size=n, replace=False)]
