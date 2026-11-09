"""Randomized evaluation: N pallets (random case mix of 7-10 cases, pick pose noise +-2 cm / +-10 deg,
pad positioning error sigma 12 mm, vacuum level x0.8-1.05, one missing case in ~30 % of pallets).

  python eval.py --pallets 50 --workers 3 --seed 0
Writes results/eval_summary.json, results/eval_cases.jsonl (per-case vision metrics),
results/eval_pallets.jsonl, results/transitions.jsonl (state machine log), results/eval_stdout.log.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def _worker(args):
    ids, seed, use_model, sigma, tag = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from grasp_model import GraspModel
    from metrics import layer_flatness, neighbour_gaps
    from planner import first_fit, random_order, violations
    from sim import Cell
    from state_machine import Controller
    from vision import DepthCamera
    model = GraspModel.load(RES / "grasp_mlp.pt") if use_model else None
    cell = Cell()
    pick, pal = DepthCamera(cell, "pick_cam"), DepthCamera(cell, "pallet_cam")
    out = []
    log_path = RES / f"part{tag}_{ids[0]:03d}.jsonl"
    with open(log_path, "w") as log:
        for pid in ids:
            rng = np.random.default_rng(seed * 1000 + pid)
            order = random_order(rng)
            missing = {order[int(rng.integers(len(order)))]} if rng.random() < 0.3 else set()
            cell.reset()
            t0 = time.time()
            ctl = Controller(cell, pick, pal, rng, log_fp=log, grasp_model=model, grasp_sigma=sigma,
                             pallet_id=pid)
            r = ctl.run(order, missing=missing)
            neighbour_gaps(r["rows"])
            expected = [c for c in order if c not in missing]
            unstable = [x["case"] for x in r["rows"] if x["unstable"]]
            not_found = [x["case"] for x in r["rows"] if not x["found"]]
            success = (sorted(r["placed"]) == sorted(expected) and not unstable and not not_found)
            ff = violations(first_fit(expected)[1])
            out.append({
                "pallet": pid, "order": order, "missing": sorted(missing), "placed": r["placed"],
                "success": bool(success), "events": r["events"], "unstable": unstable,
                "not_found_by_vision": not_found, "flagged_risky": r["flagged"],
                "violations_planner": violations(r["pallet"]), "violations_first_fit": ff,
                "layer_flatness_mm": layer_flatness(r["rows"]), "rows": r["rows"],
                "sim_time_s": cell.t_sim, "wall_s": time.time() - t0,
            })
            cell.t_sim = 0.0
    pick.close()
    pal.close()
    return out


def summarize(pallets):
    rows = [x for p in pallets for x in p["rows"]]
    found = [x for x in rows if x["found"]]
    ev = Counter(e["event"] for p in pallets for e in p["events"])
    err = np.array([x["place_err_mm"] for x in found])
    tilt = np.array([x["tilt_deg"] for x in found])
    gaps = np.array([x["min_gap_mm"] for x in found if x.get("min_gap_mm") is not None])
    flat = np.array([v for p in pallets for k, v in p["layer_flatness_mm"].items()])
    vp = Counter()
    vf = Counter()
    for p in pallets:
        vp.update({k: v for k, v in p["violations_planner"].items()})
        vf.update({k: v for k, v in p["violations_first_fit"].items()})
    n_picks_attempted = len(rows) + sum(v for k, v in ev.items() if k.startswith("drop") or k.startswith("fail"))
    return {
        "pallets": len(pallets),
        "pallet_success": sum(p["success"] for p in pallets),
        "pallet_success_rate": sum(p["success"] for p in pallets) / len(pallets),
        "cases_ordered": sum(len(p["order"]) for p in pallets),
        "cases_missing_injected": sum(len(p["missing"]) for p in pallets),
        "cases_placed": len(rows),
        "events_by_type": dict(ev),
        "drops_by_cause": {k: v for k, v in ev.items() if k.startswith("drop")},
        "unstable_cases": sum(len(p["unstable"]) for p in pallets),
        "vision_not_found": sum(len(p["not_found_by_vision"]) for p in pallets),
        "grasp_model_flags": sum(p["flagged_risky"] for p in pallets),
        "placement_error_mm": {"mean": float(err.mean()), "median": float(np.median(err)),
                               "p95": float(np.percentile(err, 95)), "max": float(err.max()), "n": int(len(err))},
        "tilt_deg_from_depth": {"mean": float(tilt.mean()), "p95": float(np.percentile(tilt, 95)),
                                "max": float(tilt.max())},
        "min_gap_to_neighbour_mm": {"mean": float(gaps.mean()), "min": float(gaps.min()), "n": int(len(gaps))},
        "layer_flatness_mm": {"mean": float(flat.mean()), "max": float(flat.max()), "n_layers": int(len(flat))},
        "rule_violations_planner": dict(vp),
        "rule_violations_first_fit_same_cases": dict(vf),
        "sim_time_per_pallet_s_mean": float(np.mean([p["sim_time_s"] for p in pallets])),
        "picks_attempted_approx": int(n_picks_attempted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pallets", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.012)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from multiprocessing import get_context
    ids = list(range(a.pallets))
    chunks = [(ids[i::a.workers], a.seed, not a.no_model, a.sigma, a.tag) for i in range(a.workers)]
    t = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        pallets = sorted([p for ch in pool.map(_worker, chunks) for p in ch], key=lambda p: p["pallet"])
    tag = a.tag
    with open(RES / f"eval_pallets{tag}.jsonl", "w") as f:
        for p in pallets:
            f.write(json.dumps({k: v for k, v in p.items() if k != "rows"}, default=float) + "\n")
    with open(RES / f"eval_cases{tag}.jsonl", "w") as f:
        for p in pallets:
            for r in p["rows"]:
                f.write(json.dumps({"pallet": p["pallet"], **r}, default=float) + "\n")
    s = summarize(pallets)
    s["config"] = {"pallets": a.pallets, "seed": a.seed, "grasp_model": not a.no_model,
                   "grasp_sigma_m": a.sigma, "wall_s": time.time() - t}
    (RES / f"eval_summary{tag}.json").write_text(json.dumps(s, indent=2))
    # merge per-worker transition logs
    with open(RES / f"transitions{tag}.jsonl", "w") as f:
        for p in sorted(RES.glob(f"part{tag}_*.jsonl")):
            f.write(p.read_text())
            p.unlink()
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
