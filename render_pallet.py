"""Builds one pallet (seed 7) and saves results/pallet_overview.png (side view) and
results/pallet_depth_contours.png (pallet camera depth image with the OpenCV contours found)."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from planner import ctype, random_order  # noqa: E402
from sim import Cell  # noqa: E402
from state_machine import Controller  # noqa: E402
from vision import DepthCamera  # noqa: E402

rng = np.random.default_rng(7)
cell = Cell()
pick, pal = DepthCamera(cell, "pick_cam"), DepthCamera(cell, "pallet_cam")
r = Controller(cell, pick, pal, rng).run(random_order(rng))
ren = mujoco.Renderer(cell.m, 480, 640)
ren.update_scene(cell.d, "overview")
cv2.imwrite("results/pallet_overview.png", ren.render()[:, :, ::-1])
depth, _ = pal.capture()
pts = pal.world_points(depth)
z = pts[..., 2]
img = cv2.applyColorMap(np.clip((z - 0.0) / 0.45 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
for row in r["rows"]:
    if not row["found"]:
        continue
    # redraw the detected footprint as a rotated rectangle in pixels
    cx, cy = pal.to_pixel([*row["center"], row["top_z"]])
    w = row["size_m"][0] * pal.f / (1.30 - row["top_z"])
    h = row["size_m"][1] * pal.f / (1.30 - row["top_z"])
    box = cv2.boxPoints(((cx, cy), (w, h), row["yaw_err_deg"]))
    cv2.polylines(img, [box.astype(np.int32)], True, (0, 0, 255), 1)
    cv2.putText(img, f"{row['place_err_mm']:.0f}mm", (int(cx) - 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1)
cv2.imwrite("results/pallet_depth_contours.png", img)
print("placed", r["placed"], "types", [ctype(c).name for c in r["placed"]])
