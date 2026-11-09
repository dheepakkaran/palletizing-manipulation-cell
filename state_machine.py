"""Palletizing state machine: IDLE -> PERCEIVE -> PICK -> TRANSFER -> PLACE -> VERIFY -> NEXT,
plus REPLAN (case missing at the pick spot, or a case was lost), RETRY (seal check failed or the
grasp model flagged a risky pick) and FAIL (case dropped, or retries used up).
Every transition is written to a JSONL log."""
from __future__ import annotations

import json
from enum import Enum

import numpy as np

from build_scene import CONVEYOR_TOP, PICK_XY, park_pos
from planner import Pallet, ctype, plan
from sim import DOWN, pad_overlap, rot_z
from vision import detect_box
from metrics import case_metrics


class S(str, Enum):
    IDLE = "IDLE"
    PERCEIVE = "PERCEIVE"
    PICK = "PICK"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    VERIFY = "VERIFY"
    NEXT = "NEXT"
    REPLAN = "REPLAN"
    RETRY = "RETRY"
    FAIL = "FAIL"
    DONE = "DONE"


ALLOWED = {
    S.IDLE: {S.PERCEIVE, S.DONE},
    S.PERCEIVE: {S.PICK, S.REPLAN, S.RETRY},
    S.PICK: {S.TRANSFER, S.RETRY},
    S.TRANSFER: {S.PLACE, S.FAIL},
    S.PLACE: {S.VERIFY},
    S.VERIFY: {S.NEXT},
    S.NEXT: {S.IDLE},
    S.REPLAN: {S.IDLE},
    S.RETRY: {S.PERCEIVE, S.FAIL},
    S.FAIL: {S.REPLAN},
    S.DONE: set(),
}
SAFE_Z = 0.50
CLEAR = [0.45, 0.35, SAFE_Z]  # above the pick spot: outside the pallet camera view; the pick camera looks from the side
MAX_RETRIES = 2


class Controller:
    def __init__(self, cell, pick_cam, pallet_cam, rng, log_fp=None, grasp_model=None,
                 grasp_sigma=0.012, leak_range=(0.8, 1.05), pallet_id=0):
        self.cell, self.pick_cam, self.pallet_cam = cell, pick_cam, pallet_cam
        self.rng, self.log_fp, self.model = rng, log_fp, grasp_model
        self.grasp_sigma, self.leak_range, self.pallet_id = grasp_sigma, leak_range, pallet_id
        self.state = S.IDLE
        self.transitions: list[tuple[str, str]] = []

    def go(self, new: S, **info):
        if new not in ALLOWED[self.state]:
            raise RuntimeError(f"illegal transition {self.state} -> {new}")
        rec = {"pallet": self.pallet_id, "t_sim": round(self.cell.t_sim, 3), "from": self.state.value,
               "to": new.value, **info}
        if self.log_fp:
            self.log_fp.write(json.dumps(rec, default=float) + "\n")
        self.transitions.append((self.state.value, new.value))
        self.state = new

    # ------------------------------------------------------------- perception
    def perceive_pick(self):
        depth, _ = self.pick_cam.capture()
        pts = self.pick_cam.world_points(depth)
        return detect_box(self.pick_cam, pts, PICK_XY, 0.12, CONVEYOR_TOP + 0.02, CONVEYOR_TOP + 0.3)

    def verify_place(self, pl):
        depth, _ = self.pallet_cam.capture()
        pts = self.pallet_cam.world_points(depth)
        top = pl.z_bottom + ctype(pl.case).size[2]
        det = detect_box(self.pallet_cam, pts, pl.target[:2], 0.08, top - 0.03, top + 0.03)
        return case_metrics(pl, det)

    # -------------------------------------------------------------- one pallet
    def run(self, order: list[str], missing: set[str] = frozenset()) -> dict:
        c = self.cell
        queue, _ = plan([x for x in order])
        actual = Pallet()
        placed, rows, events = [], [], []
        fed = set()
        retries = {}
        flagged = 0
        pl = det = None
        grasp = None
        self.state = S.IDLE
        while True:
            if self.state == S.IDLE:
                if not queue:
                    self.go(S.DONE)
                    break
                pl = queue.pop(0)
                self.go(S.PERCEIVE, case=pl.case, cell=list(pl.cell), layer=pl.layer)
            elif self.state == S.PERCEIVE:
                if pl.case not in fed and pl.case not in missing:
                    c.feed_case(pl.case, self.rng)
                    fed.add(pl.case)
                det = self.perceive_pick()
                if not det.found:
                    events.append({"case": pl.case, "event": "missing"})
                    self.go(S.REPLAN, case=pl.case, reason="case_missing_at_pick")
                    continue
                # pad positioning error (simulated), measured by the pick camera before suction
                off_world = self.rng.normal(0, self.grasp_sigma, 2)
                off_case = rot_z(-det.yaw)[:2, :2] @ off_world
                half = np.array(det.size) / 2
                feats = {"off_x": abs(off_case[0]), "off_y": abs(off_case[1]),
                         "mass": ctype(pl.case).mass, "overlap": pad_overlap(off_case, half),
                         "half_x": half[0], "half_y": half[1]}
                grasp = {"xy": det.center + off_world, "yaw": det.yaw, "feats": feats}
                if self.model is not None:
                    p_risk = self.model.predict_proba(feats)
                    grasp["p_risk"] = p_risk
                    if p_risk > 0.5 and retries.get(pl.case, 0) < MAX_RETRIES:
                        flagged += 1
                        retries[pl.case] = retries.get(pl.case, 0) + 1
                        self.go(S.RETRY, case=pl.case, reason="grasp_model_risky", p_risk=p_risk)
                        continue
                self.go(S.PICK, case=pl.case, grasp_xy=grasp["xy"].tolist(), **{k: float(v) for k, v in feats.items()})
            elif self.state == S.PICK:
                Rt = rot_z(grasp["yaw"]) @ DOWN
                top = det.top_z
                gx, gy = grasp["xy"]
                c.move_tcp([gx, gy, top + 0.12], Rt)
                c.pad_collision(True)
                c.move_linear([gx, gy, top - 0.005], Rt)
                leak = self.rng.uniform(*self.leak_range)
                ok, overlap = c.suction_on(pl.case, leak=leak)
                if not ok:
                    c.move_linear([gx, gy, top + 0.12], Rt)
                    retries[pl.case] = retries.get(pl.case, 0) + 1
                    self.go(S.RETRY, case=pl.case, reason="seal_check_failed", overlap=overlap)
                    continue
                self.go(S.TRANSFER, case=pl.case, overlap=overlap, capacity_n=c.grip.capacity_n)
            elif self.state == S.RETRY:
                if retries.get(pl.case, 0) > MAX_RETRIES:
                    events.append({"case": pl.case, "event": "fail_retries"})
                    self.go(S.FAIL, case=pl.case, reason="retries_used_up")
                else:
                    self.go(S.PERCEIVE, case=pl.case)
            elif self.state == S.TRANSFER:
                Rt = rot_z(grasp["yaw"]) @ DOWN
                gx, gy = grasp["xy"]
                c.move_linear([gx, gy, SAFE_Z], Rt, speed=0.4)
                tgt = pl.target
                # TCP target so the case centre lands on the cell centre (grasp offset kept)
                rel = c.case_pose(pl.case)[0][:2] - c.tcp_pose()[0][:2]
                above = np.array([tgt[0], tgt[1], SAFE_Z])
                # after rotating the tool to DOWN (yaw 0), the case-centre offset rotates with it
                rel_after = rot_z(-grasp["yaw"])[:2, :2] @ rel
                above[:2] -= rel_after
                reached = c.move_tcp(above, DOWN)
                g = c.grip
                if not reached and g.case is not None:
                    events.append({"case": pl.case, "event": "fail_ik_unreachable"})
                    c.release()
                    self._clear(pl.case)
                    self.go(S.FAIL, case=pl.case, reason="ik_unreachable")
                    continue
                if g.dropped or g.case is None:
                    cause = g.dropped or "lost"
                    events.append({"case": pl.case, "event": f"drop_{cause}", "overlap": g.overlap,
                                   "peak_force": g.peak_force, "peak_torque": g.peak_torque})
                    self._clear(pl.case)
                    self.go(S.FAIL, case=pl.case, reason=f"drop_{cause}")
                    continue
                self._place_xy = above[:2]
                self.go(S.PLACE, case=pl.case, peak_force=g.peak_force, peak_torque=g.peak_torque)
            elif self.state == S.PLACE:
                h = ctype(pl.case).size[2]
                top = pl.z_bottom + h
                px, py = self._place_xy
                c.move_linear([px, py, top + 0.04], DOWN, speed=0.4)
                c.move_linear([px, py, top + 0.006], DOWN)
                c.release()
                c.step(150)
                c.move_linear([px, py, SAFE_Z], DOWN, speed=0.4)
                c.move_tcp(CLEAR, DOWN)  # out of the pallet camera's view before VERIFY
                self.go(S.VERIFY, case=pl.case)
            elif self.state == S.VERIFY:
                row = self.verify_place(pl)
                row["tilt_true_deg"] = c.case_tilt_deg(pl.case)
                row["pos_after_place"] = c.case_pose(pl.case)[0].tolist()
                rows.append(row)
                actual.add(pl.case, pl.cell)
                placed.append(pl)
                self.go(S.NEXT, case=pl.case, found=row["found"],
                        place_err_mm=row.get("place_err_mm"), tilt_deg=row.get("tilt_deg"))
            elif self.state == S.NEXT:
                self.go(S.IDLE)
            elif self.state == S.FAIL:
                self.go(S.REPLAN, case=pl.case, reason="case_lost")
            elif self.state == S.REPLAN:
                rest = [q.case for q in queue]
                queue, _ = plan(rest, start=actual)
                self.go(S.IDLE, replanned=len(rest))
        # stability: let everything settle, then compare with the pose right after each place
        c.move_tcp(CLEAR, DOWN)
        c.step(500)
        for r in rows:
            p_now = c.case_pose(r["case"])[0]
            r["shift_after_settle_mm"] = float(np.linalg.norm(p_now - np.array(r["pos_after_place"])) * 1000)
            r["tilt_final_true_deg"] = c.case_tilt_deg(r["case"])
            r["unstable"] = bool(r["shift_after_settle_mm"] > 20 or r["tilt_final_true_deg"] > 5)
        return {"rows": rows, "events": events, "placed": [p.case for p in placed],
                "flagged": flagged, "pallet": actual}

    def _clear(self, name):
        """A dropped case is cleared from the cell (moved back to its parking spot)."""
        i = self.cell.case_names.index(name)
        self.cell.place_case(name, park_pos(i)[:2], 0.0, park=True)

