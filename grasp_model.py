"""Grasp-risk model: curate logged picks from a randomized pick sweep, train a small PyTorch MLP,
compare with a simple offset rule on a held-out split.

Sweep: for each sample, a random case is fed to the pick spot, the pad is placed at a random offset
from the case centre (case frame), the vacuum seals (or not), the case is lifted and carried to the
pallet. Label = 1 (risky) if the seal check failed or the case was dropped, else 0.
Features (all known before the pick): |offset x|, |offset y|, case mass, pad overlap fraction,
case half sizes. The per-pick vacuum level (leak) is NOT a feature (hidden noise).

  python grasp_model.py sweep --n 480 --workers 4   -> results/grasp_dataset.csv
  python grasp_model.py train                       -> results/grasp_model.json, results/grasp_mlp.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FEATS = ["off_x", "off_y", "mass", "overlap", "half_x", "half_y"]


# ------------------------------------------------------------------ sweep
def _sweep_worker(args):
    seed, n = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from build_scene import park_pos
    from sim import DOWN, Cell, pad_overlap, rot_z
    rng = np.random.default_rng(seed)
    cell = Cell()
    rows = []
    for _ in range(n):
        cell.reset()
        name = str(rng.choice(cell.case_names))
        ct = cell.cases[name]
        cell.feed_case(name, rng)
        cp, R = cell.case_pose(name)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        off = rng.uniform(-0.045, 0.045, 2)  # pad offset in the case frame
        gxy = cp[:2] + rot_z(yaw)[:2, :2] @ off
        top = cp[2] + ct.size[2] / 2
        Rt = rot_z(yaw) @ DOWN
        cell.move_tcp([gxy[0], gxy[1], top + 0.12], Rt)
        cell.move_linear([gxy[0], gxy[1], top - 0.005], Rt)
        leak = rng.uniform(0.8, 1.05)
        ok, overlap = cell.suction_on(name, leak=leak)
        cause = "seal_fail"
        if ok:
            cell.move_linear([gxy[0], gxy[1], 0.50], Rt, speed=0.4)
            cell.move_tcp([0.45, -0.30, 0.50], DOWN)
            g = cell.grip
            cause = g.dropped or ("held" if g.case else "lost")
        half = np.array(ct.size[:2]) / 2
        rows.append({"case": name, "off_x": abs(off[0]), "off_y": abs(off[1]), "mass": ct.mass,
                     "overlap": pad_overlap(off, half), "half_x": half[0], "half_y": half[1],
                     "leak": leak, "outcome": cause, "label": int(cause != "held")})
        cell.release()
        cell.place_case(name, park_pos(cell.case_names.index(name))[:2], 0.0, park=True)
    return rows


def sweep(n: int, workers: int, seed: int = 100):
    from multiprocessing import get_context
    per = [(seed + i, n // workers) for i in range(workers)]
    with get_context("spawn").Pool(workers) as pool:
        rows = [r for chunk in pool.map(_sweep_worker, per) for r in chunk]
    RES.mkdir(exist_ok=True)
    with open(RES / "grasp_dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows, risky = {sum(r['label'] for r in rows)}")


# ------------------------------------------------------------------ model
def load_dataset():
    with open(RES / "grasp_dataset.csv") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATS] for r in rows], np.float32)
    y = np.array([int(r["label"]) for r in rows], np.float32)
    return X, y, rows


class GraspModel:
    """Small MLP (6 -> 32 -> 32 -> 1) on standardized features."""

    def __init__(self, net=None, mu=None, sd=None):
        self.net, self.mu, self.sd = net, mu, sd

    @staticmethod
    def build():
        import torch.nn as nn
        return nn.Sequential(nn.Linear(len(FEATS), 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                             nn.Linear(32, 1))

    def fit(self, X, y, epochs=400, seed=0):
        import torch
        torch.manual_seed(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.net = self.build()
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        Xt = torch.tensor((X - self.mu) / self.sd)
        yt = torch.tensor(y)[:, None]
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(self.net(Xt), yt)
            loss.backward()
            opt.step()
        return float(loss)

    def proba(self, X) -> np.ndarray:
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor((np.asarray(X, np.float32) - self.mu) / self.sd))
        return torch.sigmoid(z).numpy()[:, 0]

    def predict_proba(self, feats: dict) -> float:
        return float(self.proba(np.array([[feats[k] for k in FEATS]], np.float32))[0])

    def save(self, path):
        import torch
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    @classmethod
    def load(cls, path):
        import torch
        ck = torch.load(path, weights_only=False)
        m = cls(cls.build(), ck["mu"], ck["sd"])
        m.net.load_state_dict(ck["state"])
        m.net.eval()
        return m


def rule_fit(X, y):
    """Baseline: risky if the pad offset distance > threshold (threshold picked on the train split)."""
    d = np.hypot(X[:, 0], X[:, 1])
    best = max(np.unique(d), key=lambda t: ((d > t) == y).mean())
    return float(best)


def rule_predict(X, t):
    return (np.hypot(X[:, 0], X[:, 1]) > t).astype(np.float32)


def train(seed=0):
    X, y, _ = load_dataset()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = len(y) // 4
    te, tr = idx[:n_test], idx[n_test:]
    m = GraspModel()
    loss = m.fit(X[tr], y[tr])
    acc_mlp = float(((m.proba(X[te]) > 0.5) == y[te]).mean())
    t = rule_fit(X[tr], y[tr])
    acc_rule = float((rule_predict(X[te], t) == y[te]).mean())
    pred = m.proba(X[te]) > 0.5
    tp = int((pred & (y[te] == 1)).sum())
    out = {
        "n_total": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "risky_rate_total": float(y.mean()), "risky_rate_test": float(y[te].mean()),
        "features": FEATS, "model": "MLP 6-32-32-1, Adam lr 3e-3, 400 full-batch epochs, CPU",
        "final_train_loss": loss,
        "heldout_accuracy_mlp": acc_mlp,
        "heldout_accuracy_rule": acc_rule, "rule": f"risky if hypot(off_x, off_y) > {t:.4f} m",
        "heldout_accuracy_majority": float(max(y[te].mean(), 1 - y[te].mean())),
        "mlp_recall_risky": tp / max(1, int((y[te] == 1).sum())),
        "mlp_precision_risky": tp / max(1, int(pred.sum())),
        "split_seed": seed,
    }
    RES.mkdir(exist_ok=True)
    m.save(RES / "grasp_mlp.pt")
    (RES / "grasp_model.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "train"])
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(a.n, a.workers)
    else:
        train()
