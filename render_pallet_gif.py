"""Replay eval pallet 0 (`eval.py --pallets 50 --seed 0`, grasp model on) with a camera on, and save
docs/assets/pallet_build.gif plus docs/assets/pallet_build_strip.png.

Same random generator, same case order, same grasp model and a fresh cell, exactly as the eval worker
starts its first pallet; the result is checked against results/eval_pallets.jsonl. Rendering only reads
the simulation state, so it does not change the run.
"""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from eval import RES, ROOT  # noqa: E402
from grasp_model import GraspModel  # noqa: E402
from planner import random_order  # noqa: E402
from sim import Cell  # noqa: E402
from state_machine import Controller  # noqa: E402
from vision import DepthCamera  # noqa: E402

W, H, EVERY_S = 480, 360, 0.5
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


class RecordingCell(Cell):
    def __init__(self):
        super().__init__()
        self.ren = mujoco.Renderer(self.m, H, W)
        self.frames, self.caption, self._next = [], "", 0.0
        self.placed = 0

    def snap(self):
        self.ren.update_scene(self.d, "overview")
        img = Image.fromarray(self.ren.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 22], fill=(255, 255, 255))
        dr.text((8, 5), f"t = {self.t_sim:5.1f} s   {self.caption}   placed {self.placed}", fill=(10, 10, 10))
        self.frames.append(img)

    def step(self, n=1):
        for _ in range(n):
            super().step(1)
            if self.t_sim >= self._next:
                self._next += EVERY_S
                self.snap()


class RecordingController(Controller):
    def go(self, new, **info):
        super().go(new, **info)
        self.cell.caption = new.value
        if new.value == "VERIFY":
            self.cell.placed += 1


def main():
    model = GraspModel.load(RES / "grasp_mlp.pt")
    cell = RecordingCell()
    pick, pal = DepthCamera(cell, "pick_cam"), DepthCamera(cell, "pallet_cam")
    rng = np.random.default_rng(0 * 1000 + 0)
    order = random_order(rng)
    missing = {order[int(rng.integers(len(order)))]} if rng.random() < 0.3 else set()
    cell.reset()
    ctl = RecordingController(cell, pick, pal, rng, grasp_model=model, grasp_sigma=0.012, pallet_id=0)
    r = ctl.run(order, missing=missing)
    cell.caption = "done"
    for _ in range(6):
        cell.snap()
    logged = json.loads(open(RES / "eval_pallets.jsonl").readline())
    assert r["placed"] == logged["placed"] and abs(cell.t_sim - logged["sim_time_s"]) < 1e-6, (r["placed"], cell.t_sim)
    print(f"replay matches eval pallet 0: {len(r['placed'])} cases placed, {cell.t_sim:.1f} s simulated")
    pal_frames = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in cell.frames]
    pal_frames[0].save(OUT / "pallet_build.gif", save_all=True, append_images=pal_frames[1:], duration=70, loop=0,
                       optimize=True)
    idx = np.linspace(0, len(cell.frames) - 1, 6).astype(int)
    strip = Image.new("RGB", (W * 3, H * 2), "white")
    for k, i in enumerate(idx):
        strip.paste(cell.frames[i], (W * (k % 3), H * (k // 3)))
    strip.save(OUT / "pallet_build_strip.png", optimize=True)
    print(len(cell.frames), "frames")
    pick.close()
    pal.close()


if __name__ == "__main__":
    main()
