"""MuJoCo cell wrapper: physics stepping, joint-space moves on the Panda's position actuators,
damped-least-squares IK for the vacuum tool, and the vacuum gripper itself.

Vacuum gripper model (simulation, not a real vacuum physics model):
  * seal check = the pad geom touches the case geom AND the pad overlaps the case's top face by
    at least SEAL_MIN_OVERLAP of its area (overlap computed from the pad centre in the case frame).
  * grip = a MuJoCo `weld` equality between the tool body and the case, switched on/off.
  * holding capacity = vacuum pressure x sealed area (pressure varies per pick, +-leak). Every
    physics step the weld's constraint force and torque are read from `efc_force`; if the pull
    force exceeds the capacity, or the peel torque exceeds capacity x PEEL_ARM, the weld switches
    off and the case is dropped (cause logged: "pull" or "peel").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from build_scene import CONVEYOR_TOP, PICK_XY, park_pos  # noqa: E402
from cases import case_bodies  # noqa: E402

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene" / "cell.xml"
ARM = 7
PAD_R = 0.04
PAD_AREA = np.pi * PAD_R ** 2
VACUUM_PA = 9000.0          # nominal vacuum under the pad -> ~45 N holding force at full seal
SEAL_MIN_OVERLAP = 0.5      # below this the seal check fails (no grip)
PEEL_ARM = 0.4 * PAD_R      # torque capacity = force capacity x this lever
DOWN = np.diag([1.0, -1.0, -1.0])  # tool z axis pointing at the floor
HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
MAX_JOINT_SPEED = 1.0  # rad/s for joint-space interpolation


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def pad_overlap(offset_xy, half_xy, n=41) -> float:
    """Fraction of the pad disc that lies on the case top (pad centre offset in the case frame)."""
    g = np.linspace(-PAD_R, PAD_R, n)
    X, Y = np.meshgrid(g, g)
    inside = X ** 2 + Y ** 2 <= PAD_R ** 2
    px, py = X + offset_xy[0], Y + offset_xy[1]
    on = (np.abs(px) <= half_xy[0]) & (np.abs(py) <= half_xy[1])
    return float((on & inside).sum() / inside.sum())


@dataclass
class GripState:
    case: str | None = None
    capacity_n: float = 0.0
    overlap: float = 0.0
    dropped: str | None = None  # cause when a held case was lost
    peak_force: float = 0.0
    peak_torque: float = 0.0
    _v_prev: np.ndarray | None = None
    _a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _z0: float = 0.0


class Cell:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.kd = mujoco.MjData(self.m)  # scratch data for IK (kinematics only)
        self.cases = {n: ct for n, ct in case_bodies()}
        self.case_names = list(self.cases)
        self.body = {n: self.m.body(n).id for n in self.case_names}
        self.geom = {n: self.m.geom(f"{n}_geom").id for n in self.case_names}
        self.qadr = {n: self.m.jnt_qposadr[self.m.body(n).jntadr[0]] for n in self.case_names}
        self.vadr = {n: self.m.jnt_dofadr[self.m.body(n).jntadr[0]] for n in self.case_names}
        self.eq = {n: self.m.equality(f"suck_{n}").id for n in self.case_names}
        self.tool = self.m.body("tool").id
        self.pad = self.m.geom("pad").id
        self.tcp = self.m.site("tcp").id
        self.lo = self.m.jnt_range[:ARM, 0].copy()
        self.hi = self.m.jnt_range[:ARM, 1].copy()
        self.grip = GripState()
        self.vacuum_pa = VACUUM_PA
        self.t_sim = 0.0
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.eq_active[:] = 0
        self.d.qpos[:ARM] = HOME
        self.d.ctrl[:ARM] = HOME
        for i, n in enumerate(self.case_names):
            self.place_case(n, park_pos(i)[:2], 0.0, z=None, park=True)
        mujoco.mj_forward(self.m, self.d)
        self.grip = GripState()
        self.pad_collision(True)
        self.t_sim = 0.0

    def place_case(self, name, xy, yaw, z=None, park=False):
        ct = self.cases[name]
        if z is None:
            z = (0.0 if park else CONVEYOR_TOP) + ct.size[2] / 2 + 0.0005
        a, v = self.qadr[name], self.vadr[name]
        self.d.qpos[a:a + 3] = [xy[0], xy[1], z]
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[v:v + 6] = 0

    def feed_case(self, name, rng, pos_noise=0.02, yaw_noise=np.deg2rad(10)):
        """Conveyor feeds `name` to the pick spot with random pose noise."""
        dx, dy = rng.uniform(-pos_noise, pos_noise, 2)
        yaw = rng.uniform(-yaw_noise, yaw_noise)
        self.place_case(name, (PICK_XY[0] + dx, PICK_XY[1] + dy), yaw)
        mujoco.mj_forward(self.m, self.d)

    def case_pose(self, name):
        a = self.qadr[name]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[a + 3:a + 7])
        return self.d.qpos[a:a + 3].copy(), R.reshape(3, 3)

    def case_tilt_deg(self, name) -> float:
        _, R = self.case_pose(name)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def tcp_pose(self, d=None):
        d = d or self.d
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    # -------------------------------------------------------------------- IK
    def ik(self, target_pos, target_rot=DOWN, q0=None, iters=300, damping=0.05, rng=None, seed_first=False):
        """DLS IK for the tool-centre point (position + orientation). Returns q or None."""
        rng = rng or np.random.default_rng(0)
        m, kd = self.m, self.kd
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        g = HOME.copy()  # elbow-up guess: face the target, wrist yaw follows
        g[0] = np.arctan2(target_pos[1], target_pos[0])
        want = yaw_of(target_rot @ DOWN.T)
        for _ in range(2):  # set joint 7 so the tool yaw matches (numeric slope, wrapped)
            y = []
            for dq7 in (0.0, 0.1):
                kd.qpos[:ARM] = g + np.eye(ARM)[6] * dq7
                mujoco.mj_kinematics(m, kd)
                y.append(yaw_of(kd.site_xmat[self.tcp].reshape(3, 3) @ DOWN.T))
            slope = np.sign(np.angle(np.exp(1j * (y[1] - y[0])))) or 1.0
            q7 = g[6] + slope * np.angle(np.exp(1j * (want - y[0])))
            for k in (0, 2 * np.pi, -2 * np.pi):
                if self.lo[6] <= q7 + k <= self.hi[6]:
                    q7 += k
                    break
            g[6] = np.clip(q7, self.lo[6], self.hi[6])
        seeds = [g, self.d.qpos[:ARM].copy() if q0 is None else np.asarray(q0, float), HOME.copy()]
        seeds += [g + rng.normal(0, 0.3, ARM) for _ in range(5)]
        if seed_first and q0 is not None:  # continuous paths: stay on the current IK branch
            seeds.insert(0, np.asarray(q0, float))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for q in seeds:
            q = np.clip(q, self.lo, self.hi)
            for _ in range(iters):
                kd.qpos[:ARM] = q
                mujoco.mj_kinematics(m, kd)
                p = kd.site_xpos[self.tcp]
                R = kd.site_xmat[self.tcp].reshape(3, 3)
                e_p = target_pos - p
                e_r = 0.5 * sum(np.cross(R[:, i], target_rot[:, i]) for i in range(3))
                if np.linalg.norm(e_p) < 5e-4 and np.linalg.norm(e_r) < 5e-3:
                    return q
                mujoco.mj_comPos(m, kd)
                mujoco.mj_jacSite(m, kd, jp, jr, self.tcp)
                J = np.vstack([jp[:, :ARM], jr[:, :ARM]])
                e = np.concatenate([e_p, e_r])
                Jp = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))
                dq = Jp @ e
                q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return None

    # --------------------------------------------------------------- physics
    def _grip_load(self):
        """Load on the vacuum pad: F = m (a - g) of the held case (acceleration from smoothed
        finite differences of its velocity) and the peel torque |r x F|, r = pad centre -> case
        centre of mass. Standard vacuum-gripper sizing formula, evaluated every physics step."""
        g = self.grip
        n = g.case
        v = self.d.qvel[self.vadr[n]:self.vadr[n] + 3].copy()
        dt = self.m.opt.timestep
        a_raw = (v - g._v_prev) / dt if g._v_prev is not None else np.zeros(3)
        g._v_prev = v
        g._a = 0.9 * g._a + 0.1 * a_raw
        F = self.cases[n].mass * (g._a - self.m.opt.gravity)
        cp = self.d.qpos[self.qadr[n]:self.qadr[n] + 3]
        r = cp - self.d.site_xpos[self.tcp]
        r[2] = 0.0  # horizontal lever only
        return float(np.linalg.norm(F)), float(np.linalg.norm(np.cross(r, F)))

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.t_sim += self.m.opt.timestep
            g = self.grip
            if g.case is not None:
                f, t = self._grip_load()
                if self.d.qpos[self.qadr[g.case] + 2] < g._z0 + 0.01:
                    continue  # still resting on its support
                g.peak_force, g.peak_torque = max(g.peak_force, f), max(g.peak_torque, t)
                if f > g.capacity_n:
                    self.release(dropped="pull")
                elif t > g.capacity_n * PEEL_ARM:
                    self.release(dropped="peel")

    def move_joints(self, q_goal, hold=0.15):
        """Joint-space linear interpolation sent to the position actuators, then a short hold."""
        q0 = self.d.ctrl[:ARM].copy()
        q_goal = np.asarray(q_goal)
        T = max(0.3, float(np.max(np.abs(q_goal - q0))) / MAX_JOINT_SPEED)
        n = int(T / self.m.opt.timestep)
        for k in range(1, n + 1):
            s = k / n
            s = 3 * s ** 2 - 2 * s ** 3  # smooth start/stop
            self.d.ctrl[:ARM] = q0 + s * (q_goal - q0)
            self.step()
        self.step(int(hold / self.m.opt.timestep))

    def move_linear(self, pos, rot=DOWN, n=10, speed=0.25, hold=0.1):
        """Straight-line TCP move: IK at n points along the line (each seeded with the last),
        streamed as one smooth joint trajectory at about `speed` m/s."""
        p0, R0 = self.tcp_pose()
        pos = np.asarray(pos, float)
        y0, y1 = yaw_of(R0 @ DOWN.T), yaw_of(rot @ DOWN.T)
        dy = np.angle(np.exp(1j * (y1 - y0)))
        qs, q = [self.d.ctrl[:ARM].copy()], self.d.ctrl[:ARM].copy()
        for k in range(1, n + 1):
            rk = rot if k == n else rot_z(y0 + dy * k / n) @ DOWN  # tool yaw turns smoothly
            q = self.ik(p0 + (pos - p0) * k / n, rk, q0=q, seed_first=True)
            if q is None:
                return False
            qs.append(q)
        qs = np.array(qs)
        T = max(0.2, float(np.linalg.norm(pos - p0)) / speed)
        steps = int(T / self.m.opt.timestep)
        for k in range(1, steps + 1):
            s = k / steps
            s = (3 * s ** 2 - 2 * s ** 3) * n
            i = min(int(s), n - 1)
            self.d.ctrl[:ARM] = qs[i] + (s - i) * (qs[i + 1] - qs[i])
            self.step()
        self.step(int(hold / self.m.opt.timestep))
        return True

    def move_tcp(self, pos, rot=DOWN):
        q = self.ik(np.asarray(pos, float), rot, q0=self.d.ctrl[:ARM])
        if q is None:
            return False
        self.move_joints(q)
        return True

    # --------------------------------------------------------------- vacuum
    def pad_contact(self, name) -> bool:
        g = self.geom[name]
        for c in self.d.contact[: self.d.ncon]:
            if {c.geom1, c.geom2} == {g, self.pad} and c.dist < 0.002:
                return True
        return False

    def pad_offset(self, name):
        """Pad centre in the case frame (x, y) and the case half sizes."""
        cp, R = self.case_pose(name)
        p, _ = self.tcp_pose()
        off = R.T @ (p - cp)
        return off[:2], np.array(self.cases[name].size[:2]) / 2

    def seal_check(self, name):
        off, half = self.pad_offset(name)
        overlap = pad_overlap(off, half)
        ok = self.pad_contact(name) and overlap >= SEAL_MIN_OVERLAP
        return ok, overlap, off

    def suction_on(self, name, leak=1.0) -> tuple[bool, float]:
        ok, overlap, _ = self.seal_check(name)
        if not ok:
            return False, overlap
        eq = self.eq[name]
        self.d.ctrl[:ARM] = self.d.qpos[:ARM]  # stop pressing down before the grip engages
        # weld at the current relative pose: data = anchor(3), relpos(3), relquat(4), torquescale
        p1, q1 = self.d.xpos[self.tool], self.d.xquat[self.tool]
        p2, q2 = self.d.xpos[self.body[name]], self.d.xquat[self.body[name]]
        R1 = self.d.xmat[self.tool].reshape(3, 3)
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, q1)
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, q1inv, q2)
        self.m.eq_data[eq, 0:3] = 0
        self.m.eq_data[eq, 3:6] = R1.T @ (p2 - p1)  # case in tool frame
        self.m.eq_data[eq, 6:10] = rel
        self.m.eq_data[eq, 10] = 1.0
        self.d.eq_active[eq] = 1
        # while held, the pad no longer collides (the weld is the pad-case interface)
        self.m.geom_contype[self.pad] = 0
        self.m.geom_conaffinity[self.pad] = 0
        self.grip = GripState(case=name, capacity_n=self.vacuum_pa * leak * PAD_AREA * overlap,
                              overlap=overlap, _z0=float(p2[2]))
        return True, overlap

    def pad_collision(self, on: bool):
        self.m.geom_contype[self.pad] = int(on)
        self.m.geom_conaffinity[self.pad] = int(on)

    def release(self, dropped: str | None = None):
        if self.grip.case is not None:
            self.d.eq_active[self.eq[self.grip.case]] = 0
            # pad collision stays off until the next approach (it may overlap the case top by ~1 mm)
            if dropped:
                self.grip.dropped = dropped
            self.grip.case = None
