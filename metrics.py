"""Pick-and-place quality metrics from depth-camera detections (vision.detect_box output)."""
from __future__ import annotations

import numpy as np

from planner import NX, NY, Placement, ctype


def case_metrics(pl: Placement, det) -> dict:
    tgt = pl.target
    if not det.found:
        return {"case": pl.case, "cell": list(pl.cell), "layer": pl.layer, "found": False}
    err = det.center - tgt[:2]
    exp_top = pl.z_bottom + ctype(pl.case).size[2]
    return {
        "case": pl.case, "cell": list(pl.cell), "layer": pl.layer, "found": True,
        "place_err_mm": float(np.linalg.norm(err) * 1000),
        "place_err_x_mm": float(err[0] * 1000), "place_err_y_mm": float(err[1] * 1000),
        "yaw_err_deg": float(np.degrees(det.yaw)),
        "tilt_deg": det.tilt_deg,
        "top_z_err_mm": float((det.top_z - exp_top) * 1000),
        "top_z": det.top_z,
        "size_m": list(det.size),
        "center": det.center.tolist(),
    }


def neighbour_gaps(rows: list[dict]) -> None:
    """Adds `min_gap_mm` to each row: smallest edge-to-edge gap to an adjacent case in the same layer
    (axis-aligned estimate from detected centres and footprints)."""
    by = {(tuple(r["cell"]), r["layer"]): r for r in rows if r.get("found")}
    for (cell, layer), r in by.items():
        gaps = []
        for dx, dy, ax in ((1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1)):
            nb = by.get(((cell[0] + dx, cell[1] + dy), layer))
            if nb is None or not (0 <= cell[0] + dx < NX and 0 <= cell[1] + dy < NY):
                continue
            dist = abs(nb["center"][ax] - r["center"][ax])
            half = (min(r["size_m"]) + min(nb["size_m"])) / 2
            gaps.append((dist - half) * 1000)
        r["min_gap_mm"] = float(min(gaps)) if gaps else None


def layer_flatness(rows: list[dict]) -> dict:
    """Per layer: max - min of detected top heights (mm). 0 = perfectly flat."""
    out = {}
    for layer in sorted({r["layer"] for r in rows if r.get("found")}):
        z = [r["top_z"] for r in rows if r.get("found") and r["layer"] == layer]
        out[str(layer)] = float((max(z) - min(z)) * 1000) if len(z) > 1 else 0.0
    return out


