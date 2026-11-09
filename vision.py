"""Fixed overhead RGB-D cameras (MuJoCo offscreen render: RGB + depth, EGL) and OpenCV case detection.

Depth image -> per-pixel world height map -> threshold a height band -> OpenCV contours ->
minAreaRect gives the case centre (x, y), footprint and yaw; a least-squares plane through the
3-D points inside the contour gives the tilt of the top face. Ground truth from the simulator is
never used here (only later, in tests, to score the detector).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np


@dataclass
class Detection:
    found: bool
    center: np.ndarray | None = None   # world x, y (m)
    top_z: float | None = None         # median top-face height (m)
    size: tuple | None = None          # footprint (m, m) from minAreaRect
    yaw: float | None = None           # rad, in [-45, 45] deg (square-ish boxes)
    tilt_deg: float | None = None
    area_px: int = 0
    rect_px: tuple | None = None


class DepthCamera:
    def __init__(self, cell, name: str, width=640, height=480):
        self.cell, self.name = cell, name
        m = cell.m
        self.cam = m.camera(name).id
        self.w, self.h = width, height
        self.rgb_r = mujoco.Renderer(m, height, width)
        self.dep_r = mujoco.Renderer(m, height, width)
        self.dep_r.enable_depth_rendering()
        fovy = np.deg2rad(m.cam_fovy[self.cam])
        self.f = (height / 2) / np.tan(fovy / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        self.ray = np.stack([(u - self.cx) / self.f, -(v - self.cy) / self.f, -np.ones_like(u, float)], -1)

    def capture(self, rgb=False):
        d = self.cell.d
        mujoco.mj_forward(self.cell.m, d)
        self.dep_r.update_scene(d, self.name)
        depth = self.dep_r.render().copy()
        img = None
        if rgb:
            self.rgb_r.update_scene(d, self.name)
            img = self.rgb_r.render().copy()
        return depth, img

    def world_points(self, depth) -> np.ndarray:
        d = self.cell.d
        R = d.cam_xmat[self.cam].reshape(3, 3)
        return d.cam_xpos[self.cam] + (self.ray * depth[..., None]) @ R.T

    def to_pixel(self, p) -> tuple[float, float]:
        d = self.cell.d
        R = d.cam_xmat[self.cam].reshape(3, 3)
        pc = R.T @ (np.asarray(p, float) - d.cam_xpos[self.cam])
        z = -pc[2]
        return self.cx + self.f * pc[0] / z, self.cy - self.f * pc[1] / z

    def close(self):
        self.rgb_r.close()
        self.dep_r.close()


def detect_box(cam: DepthCamera, pts: np.ndarray, center_xy, half_roi: float,
               z_lo: float, z_hi: float, min_area_px=300) -> Detection:
    """Find the largest top face with height in [z_lo, z_hi] inside a square ROI around center_xy."""
    Z = pts[..., 2]
    X, Y = pts[..., 0], pts[..., 1]
    roi = (np.abs(X - center_xy[0]) < half_roi) & (np.abs(Y - center_xy[1]) < half_roi)
    mask = (roi & (Z > z_lo) & (Z < z_hi)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Detection(False)
    cnt = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(cnt))
    if area < min_area_px:
        return Detection(False, area_px=area)
    inside = np.zeros_like(mask)
    cv2.drawContours(inside, [cnt], -1, 255, -1)
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8)) > 0  # keep away from the edges
    P = pts[inside]
    top = float(np.median(P[:, 2]))
    # world-space footprint: minAreaRect on the contour points mapped to world x, y
    cpix = cnt[:, 0, :]
    W = pts[cpix[:, 1], cpix[:, 0], :2].astype(np.float32)
    (wx, wy), (sw, sh), ang = cv2.minAreaRect(W)
    yaw = np.deg2rad(((ang + 45) % 90) - 45)
    # plane fit z = a x + b y + c -> tilt of the top face
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    (a, b, _), *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    return Detection(True, np.array([wx, wy]), top, (float(sw), float(sh)), float(yaw), tilt, area,
                     cv2.minAreaRect(cpix.astype(np.float32)))
