import numpy as np
import pytest

from planner import Pallet, Placement, first_fit, plan, random_order, violations
from state_machine import ALLOWED, S, Controller


# ------------------------------------------------------------------ planner rules
def test_planner_far_fewer_violations_than_first_fit():
    # the greedy planner is not perfect (its soft fallback can break a rule), so compare totals
    rng = np.random.default_rng(3)
    vp = vf = 0
    for _ in range(100):
        order = random_order(rng)
        vp += violations(plan(order)[1])["total"]
        vf += violations(first_fit(order)[1])["total"]
    assert vp * 10 < vf


def test_first_fit_breaks_rules_on_a_bad_order():
    order = ["fragile_0", "light_0", "heavy_0", "medium_0", "heavy_1", "medium_1", "heavy_2"]
    assert violations(first_fit(order)[1])["total"] > 0
    assert violations(plan(order)[1])["total"] == 0


def test_crush_rule_counted():
    p = Pallet()
    p.add("fragile_0", (0, 0))
    p.add("heavy_0", (0, 0))  # 2.8 kg on a 0.5 kg-rated case
    v = violations(p)
    assert v["crush"] == 1 and v["heavy_on_light"] == 1


def test_replan_keeps_placed_cases():
    _, p = plan(["heavy_0", "heavy_1"])
    pls, p2 = plan(["light_0"], start=p)
    assert pls[0].layer in (0, 1)
    assert sum(len(c) for c in p2.cols.values()) == 3


# ------------------------------------------------------------------ state machine
def test_transition_table():
    assert S.REPLAN in ALLOWED[S.PERCEIVE]
    assert S.RETRY in ALLOWED[S.PICK]
    assert S.FAIL in ALLOWED[S.TRANSFER]
    assert ALLOWED[S.DONE] == set()


def test_illegal_transition_raises():
    ctl = Controller(None, None, None, None)
    with pytest.raises(RuntimeError):
        ctl.go(S.PLACE)  # IDLE -> PLACE is not allowed


@pytest.fixture(scope="module")
def cell_and_cams():
    from sim import Cell
    from vision import DepthCamera
    c = Cell()
    return c, DepthCamera(c, "pick_cam"), DepthCamera(c, "pallet_cam")


def test_one_pallet_with_missing_case(cell_and_cams):
    cell, pick, pal = cell_and_cams
    cell.reset()
    order = ["heavy_0", "medium_0", "light_0"]
    ctl = Controller(cell, pick, pal, np.random.default_rng(0))
    r = ctl.run(order, missing={"medium_0"})
    tr = ctl.transitions
    assert ("PERCEIVE", "REPLAN") in tr  # missing case -> REPLAN
    for st in ("PICK", "TRANSFER", "PLACE", "VERIFY", "NEXT"):
        assert any(b == st for _, b in tr)
    assert sorted(r["placed"]) == ["heavy_0", "light_0"]
    assert all(x["found"] and not x["unstable"] for x in r["rows"])


def test_seal_check_fails_far_off_centre(cell_and_cams):
    from sim import DOWN
    cell, _, _ = cell_and_cams
    cell.reset()
    cell.feed_case("heavy_0", np.random.default_rng(1), pos_noise=0.0, yaw_noise=0.0)
    cp, _ = cell.case_pose("heavy_0")
    top = cp[2] + 0.05
    cell.move_tcp([cp[0] + 0.06, cp[1] + 0.06, top + 0.1], DOWN)
    cell.move_linear([cp[0] + 0.06, cp[1] + 0.06, top - 0.005], DOWN)
    ok, overlap = cell.suction_on("heavy_0")
    assert not ok and overlap < 0.5


# ------------------------------------------------------------------ vision metrics
def test_metrics_on_known_placement(cell_and_cams):
    from metrics import case_metrics
    from vision import detect_box
    cell, _, pal = cell_and_cams
    cell.reset()
    pl = Placement("medium_0", (0, 1), 0, 0.10)
    tgt = pl.target
    cell.place_case("medium_0", (tgt[0] + 0.010, tgt[1] - 0.005), 0.0, z=tgt[2])
    cell.step(50)
    depth, _ = pal.capture()
    det = detect_box(pal, pal.world_points(depth), tgt[:2], 0.08, 0.17, 0.23)
    m = case_metrics(pl, det)
    assert m["found"]
    assert abs(m["place_err_x_mm"] - 10) < 3 and abs(m["place_err_y_mm"] + 5) < 3
    assert m["tilt_deg"] < 1.0
    assert abs(m["size_m"][0] - 0.13) < 0.006
