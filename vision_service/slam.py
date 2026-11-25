"""Keyframe SLAM for a slow robot with a single-beam rangefinder.

Why this runs on the Mac and why it can be this simple
------------------------------------------------------
The robot has no 2-D lidar. It has a TF-Luna on the pan/tilt head, so a
"scan" is the head sweeping ±45° while the base turns in 90° steps: a sparse
360° ring of ranges taken over ~20 s, at one height. That is far too slow for
a real-time SLAM package — but the environment is static and nobody is in a
hurry, so the problem becomes *batch* SLAM on keyframes:

    robot stops ─► sweeps ─► POST keyframe (odom pose + rays)
                                   │
       Mac: predict pose from odometry, correct it by matching the rays
            against the map so far, fold the rays into the map, find the
            nearest unexplored edge, plan a path to it
                                   │
    robot drives the path ◄── pose correction + next goal

Everything is numpy on a ~500x500 occupancy grid: a keyframe costs tens of
milliseconds, against ~20 s for the sweep that produced it.

Pieces (each small, each tested in tests/test_slam.py):
  * `OccupancyGrid`  — log-odds grid; rays carve free space, endpoints mark walls
  * `match_scan`     — correlative scan matcher: brute-force search of
                       (dx, dy, dθ) around the odometry prediction, scoring
                       ray endpoints against a blurred copy of the walls
  * `plan_path`      — BFS over free, obstacle-inflated cells, then shortcut
  * `frontiers`      — free cells next to unknown ones, clustered
  * `viewpoints`     — N reachable poses around a target, facing it

With the 8x8 ToF head the robot sends more than wall rays, and all of it is kept:
  * `clear`  — rays along which the floor was seen and nothing stood: free space
               with no wall at the end (the wall is beyond the sensor's range)
  * `low`    — things too low for a level beam (a step, a box): never driven over
  * `drops`  — where the floor falls away (stairs down): never driven near
  * `cloud`  — every reading as an (x, y, height) point: the room in 3-D, saved
               with the map

Loop closure (`_close_loop`): drift between keyframes is corrected by matching
against the map, which stops it growing without bound in a room, but a long loop
can still come back a little off and leave a seam (a wall drawn twice) - which
shrinks or stretches the room. So when a keyframe is back near one taken at least
`loop_gap` keyframes and `loop_min_travel` metres ago, it is matched twice: against
the last few keyframes only (where the drifted chain says the robot is) and against
the old keyframes only (where it really is). If the old map agrees well enough and
the two differ, the difference is spread along the keyframes in between (none for
the old one, all of it for this one) and the map is rebuilt from every keyframe's
stored readings at the corrected poses.

Paths keep clear of walls where they can (`pref_clearance`) and come with
`clear_m`: how close each leg passes to anything mapped, for the robot to size its
legs. `walls()` fits straight lines to the wall cells: wall lengths, and the
explored area's size along its main directions.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .nav_map import distance_field as _distance_field
    from .nav_map import door_approach as _door_approach
    from .nav_map import theta_star as _theta_star
except Exception:  # noqa: BLE001 — standalone import fallback
    _distance_field = _door_approach = _theta_star = None

log = logging.getLogger(__name__)

# Single-ray free-space carving (L_FREE < FREE_T) prevents uncarved radial slivers
# between ToF columns, while OCC_T + isolated-speckle filtering stops single stray
# returns from sealing doorways. Two free rays cleanly erase a transient hit.
L_FREE, L_OCC, L_MIN, L_MAX = -0.65, 0.85, -4.0, 4.0
OCC_T, FREE_T = 0.70, -0.45
TOF_WEDGE_RAD = math.radians(6.5)  # 8x8 ToF zone width (~7.5 deg): fill inter-ray slivers


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a: Sequence[float], b: Sequence[float]) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def between(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Motion from a to b, in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def transform(pose: Sequence[float], pts: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    r = np.array([[c, -s], [s, c]])
    return pts @ r.T + np.array(pose[:2])


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, size_m: float = 24.0, res: float = 0.05) -> None:
        self.res = res
        self.n = int(round(size_m / res))
        self.origin = -size_m / 2.0          # world coord of cell (0, 0)'s corner
        self.logodds = np.zeros((self.n, self.n), np.float32)   # [iy, ix]

    def cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(np.int64)

    def world(self, ij: np.ndarray) -> np.ndarray:
        return (np.asarray(ij) + 0.5) * self.res + self.origin

    def inside(self, ij: np.ndarray) -> np.ndarray:
        return (ij[..., 0] >= 0) & (ij[..., 0] < self.n) & (ij[..., 1] >= 0) & (ij[..., 1] < self.n)

    def occupied(self) -> np.ndarray:
        return self.logodds > OCC_T

    def clean_occupied(self) -> np.ndarray:
        """Occupied cells with isolated 1-cell stray ToF speckles removed."""
        return _despeckle(self.occupied())

    def free(self) -> np.ndarray:
        return self.logodds < FREE_T

    def unknown(self) -> np.ndarray:
        return ~(self.occupied() | self.free())

    def _bump(self, ij: np.ndarray, delta: float) -> None:
        ij = ij[self.inside(ij)]
        if len(ij):
            ij = np.unique(ij, axis=0)
            self.logodds[ij[:, 1], ij[:, 0]] = np.clip(
                self.logodds[ij[:, 1], ij[:, 0]] + delta, L_MIN, L_MAX)

    def integrate(self, origin_xy: np.ndarray, hits_xy: np.ndarray, max_range: float,
                  wedge_rad: float = 0.0) -> None:
        """origin_xy: (n,2) sensor positions; hits_xy: (n,2) ray endpoints (world).
        Rays shorter than max_range end in an obstacle. When wedge_rad > 0 (e.g.
        8x8 ToF columns), free space is carved across the full angular wedge with
        sub-cell angular density so zero inter-ray slivers survive at any range."""
        if not len(hits_xy):
            return
        free = []
        occ = []
        step = self.res / 2.0
        for o, h in zip(origin_xy, hits_xy):
            d = h - o
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            carve_len = min(length, max_range) if max_range > 0 else length
            base_ang = math.atan2(d[1], d[0])
            if wedge_rad > 1e-4:
                n_sub = max(3, int(math.ceil(carve_len * wedge_rad / (self.res * 0.85))) + 1)
                angs = base_ang + np.linspace(-0.5 * wedge_rad, 0.5 * wedge_rad, n_sub)
                stop_len = max(0.0, carve_len - (1.4 * self.res if max_range > 0 else 0.0))
                if stop_len > 1e-6:
                    ts = np.arange(0.0, stop_len, step)
                    dirs = np.stack([np.cos(angs), np.sin(angs)], axis=1)        # (n_sub, 2)
                    pts = o[None, None, :] + ts[:, None, None] * dirs[None, :, :]  # (nt, n_sub, 2)
                    free.append(self.cells(pts.reshape(-1, 2)))
            else:
                stop_len = max(0.0, carve_len - (self.res if max_range > 0 else 0.0))
                if stop_len > 1e-6:
                    ts = np.arange(0.0, stop_len, step)
                    free.append(self.cells(o + np.outer(ts / length, d)))
            if max_range > 0 and length < max_range - 1e-6:
                occ.append(self.cells(h[None, :]))
        if free:
            fc = np.concatenate(free)
            if occ:
                # Never carve through the endpoint cells of this same scan.
                oc = np.concatenate(occ)
                key = lambda a: a[:, 0] * (self.n + 7) + a[:, 1]
                fc = fc[~np.isin(key(fc), key(oc))]
            self._bump(fc, L_FREE)
        if occ:
            self._bump(np.concatenate(occ), L_OCC)

    def mark_free_disc(self, xy: Sequence[float], radius: float) -> None:
        """The robot is standing here, so it is certainly traversable."""
        c = self.cells(np.array(xy)[None, :])[0]
        r = int(math.ceil(radius / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = xx ** 2 + yy ** 2 <= r * r
        ij = np.stack([xx[m] + c[0], yy[m] + c[1]], axis=1)
        ij = ij[self.inside(ij)]
        self.logodds[ij[:, 1], ij[:, 0]] = np.minimum(self.logodds[ij[:, 1], ij[:, 0]], FREE_T - 0.4)

    def mark_occupied(self, xy: np.ndarray) -> None:
        if len(xy):
            self._bump(self.cells(np.asarray(xy, float).reshape(-1, 2)), L_OCC * 2)


def _despeckle(mask: np.ndarray) -> np.ndarray:
    """Drop isolated 1-cell speckles (occupied cells with zero occupied 8-neighbours).
    Real walls/furniture legs have at least one adjacent cell; a lone stray ToF reading
    otherwise inflates to block a 0.5 m doorway."""
    if not mask.any():
        return mask
    p = np.pad(mask.astype(np.uint8), 1)
    nbrs = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return mask & (nbrs > 0)


def _blur(mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    r = max(1, int(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2).astype(np.float32)
    out = mask.astype(np.float32)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)
    m = out.max()
    return out / m if m > 0 else out


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    out = mask.copy()
    r = radius_cells
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[ys, xs] = mask[yd, xd]
            out |= shifted
    return out


# --------------------------------------------------------------------------
# scan matching & pose-graph optimization
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    pose: tuple
    score: float          # mean likelihood of endpoints, 0..1
    accepted: bool


def _refine_subcell(grid: OccupancyGrid, field_: np.ndarray, pose: tuple,
                    pts_local: np.ndarray, prior: Sequence[float],
                    xy_window: float, th_window: float, prior_weight: float,
                    iters: int = 6) -> tuple:
    """Damped Gauss-Newton sub-cell refinement maximizing bilinearly-interpolated
    likelihood `field_(T(pose) @ pts_local)` minus the odometry prior penalty."""
    n = grid.n
    gy, gx = np.gradient(field_)
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    px, py = pts_local[:, 0], pts_local[:, 1]
    inv_res = 1.0 / grid.res
    w_xy = 2.0 * prior_weight / max(xy_window * xy_window, 1e-6)
    w_th = 2.0 * prior_weight / max(th_window * th_window, 1e-6)

    def eval_obj(cx: float, cy: float, cth: float) -> tuple:
        c, s = math.cos(cth), math.sin(cth)
        wx = cx + c * px - s * py
        wy = cy + s * px + c * py
        fx = (wx - grid.origin) * inv_res - 0.5
        fy = (wy - grid.origin) * inv_res - 0.5
        ok = (fx >= 0.0) & (fx < n - 1.001) & (fy >= 0.0) & (fy < n - 1.001)
        if ok.sum() < 10:
            return -1e9, 0.0, None
        ix = np.floor(fx[ok]).astype(np.int64)
        iy = np.floor(fy[ok]).astype(np.int64)
        ux, uy = fx[ok] - ix, fy[ok] - iy
        v = ((1 - ux) * (1 - uy) * field_[iy, ix]
             + ux * (1 - uy) * field_[iy, ix + 1]
             + (1 - ux) * uy * field_[iy + 1, ix]
             + ux * uy * field_[iy + 1, ix + 1])
        score = float(v.sum()) / len(pts_local)
        pen = prior_weight * (((cx - prior[0]) ** 2 + (cy - prior[1]) ** 2) / max(xy_window ** 2, 1e-6)
                              + (wrap(cth - prior[2]) / max(th_window, 1e-6)) ** 2)
        return score - pen, score, (ok, ix, iy, ux, uy, v, c, s)

    best_obj, best_score, cache = eval_obj(x, y, th)
    if cache is None:
        return pose, 0.0
    for _ in range(iters):
        ok, ix, iy, ux, uy, v, c, s = cache
        # Interpolated gradients in world units (per metre)
        dfx = (((1 - ux) * (1 - uy) * gx[iy, ix] + ux * (1 - uy) * gx[iy, ix + 1]
                + (1 - ux) * uy * gx[iy + 1, ix] + ux * uy * gx[iy + 1, ix + 1]) * inv_res)
        dfy = (((1 - ux) * (1 - uy) * gy[iy, ix] + ux * (1 - uy) * gy[iy, ix + 1]
                + (1 - ux) * uy * gy[iy + 1, ix] + ux * uy * gy[iy + 1, ix + 1]) * inv_res)
        # Residual r_i = 1 - v_i; minimize 0.5 * mean(r_i^2) + prior
        r = 1.0 - v
        dwx_dth = -s * px[ok] - c * py[ok]
        dwy_dth = c * px[ok] - s * py[ok]
        J = np.stack([-dfx, -dfy, -(dfx * dwx_dth + dfy * dwy_dth)], axis=1)
        inv_m = 1.0 / len(pts_local)
        H = (J.T @ J) * inv_m + np.diag([w_xy + 0.5, w_xy + 0.5, w_th + 1.0])
        g = (J.T @ r) * inv_m + np.array([
            w_xy * (x - prior[0]),
            w_xy * (y - prior[1]),
            w_th * wrap(th - prior[2]),
        ])
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if np.hypot(step[0], step[1]) > 0.08 or abs(step[2]) > math.radians(4.0):
            scale = min(0.08 / max(np.hypot(step[0], step[1]), 1e-9),
                        math.radians(4.0) / max(abs(step[2]), 1e-9), 1.0)
            step *= scale
        nx_, ny_, nth_ = x + float(step[0]), y + float(step[1]), wrap(th + float(step[2]))
        if math.hypot(nx_ - prior[0], ny_ - prior[1]) > xy_window * 1.15 or abs(wrap(nth_ - prior[2])) > th_window * 1.15:
            break
        obj, sc, ncache = eval_obj(nx_, ny_, nth_)
        if ncache is None or obj < best_obj + 1e-5:
            break
        x, y, th, best_obj, best_score, cache = nx_, ny_, nth_, obj, sc, ncache
        if np.hypot(step[0], step[1]) < 1e-3 and abs(step[2]) < math.radians(0.05):
            break
    return (x, y, th), best_score


def match_scan(grid: OccupancyGrid, field_: np.ndarray, prior: Sequence[float],
               pts_local: np.ndarray, xy_window: float = 0.5, th_window: float = math.radians(15),
               th_step: float = math.radians(1), min_score: float = 0.35,
               prior_weight: float = 0.08) -> MatchResult:
    """Coarse-to-fine correlative search + Gauss-Newton sub-cell refinement around
    `prior` for the pose that puts the scan's endpoints on the walls already in the map.

    `field_` is the blurred wall map (see `_blur`); a small penalty on distance from
    the odometry prior keeps symmetric rooms from pulling the pose to a look-alike spot.
    """
    if len(pts_local) < 15 or field_.max() <= 0:
        return MatchResult(tuple(prior), 0.0, False)
    k = int(round(xy_window / grid.res))

    def _search(center_pose, xy_cells: np.ndarray, th_vals: np.ndarray):
        top = []  # keep top-3 peaks (obj, score, pose)
        for dth in th_vals:
            pose = (center_pose[0], center_pose[1], wrap(center_pose[2] + dth))
            base = grid.cells(transform(pose, pts_local))                 # (n,2)
            idx = base[None, :, :] + xy_cells[:, None, :]                 # (m,n,2)
            ok = grid.inside(idx)
            ix = np.clip(idx[..., 0], 0, grid.n - 1)
            iy = np.clip(idx[..., 1], 0, grid.n - 1)
            vals = np.where(ok, field_[iy, ix], 0.0).mean(axis=1)         # (m,)
            dx_w = (center_pose[0] - prior[0]) + xy_cells[:, 0] * grid.res
            dy_w = (center_pose[1] - prior[1]) + xy_cells[:, 1] * grid.res
            dth_w = wrap(pose[2] - prior[2])
            pen = prior_weight * ((np.hypot(dx_w, dy_w) / max(xy_window, 1e-6)) ** 2
                                  + (dth_w / max(th_window, 1e-6)) ** 2)
            objs = vals - pen
            j = int(np.argmax(objs))
            cand = (float(objs[j]), float(vals[j]),
                    (center_pose[0] + float(xy_cells[j, 0]) * grid.res,
                     center_pose[1] + float(xy_cells[j, 1]) * grid.res,
                     pose[2]))
            top.append(cand)
        top.sort(key=lambda t: -t[0])
        return top[:3]

    # Coarse stage when search window is large (e.g. loop closure), then fine stage
    if k > 10:
        c_step = max(2, k // 8)
        c_rng = np.arange(-k, k + 1, c_step)
        c_offs = np.stack(np.meshgrid(c_rng, c_rng), -1).reshape(-1, 2)
        c_th = np.arange(-th_window, th_window + 1e-9, max(th_step * 3.0, math.radians(3.0)))
        seeds = _search(tuple(prior), c_offs, c_th)
        f_rng = np.arange(-c_step, c_step + 1)
        f_offs = np.stack(np.meshgrid(f_rng, f_rng), -1).reshape(-1, 2)
        f_th = np.arange(-max(th_step * 3.0, math.radians(3.0)),
                         max(th_step * 3.0, math.radians(3.0)) + 1e-9, th_step)
        best = (-1e9, 0.0, tuple(prior))
        for _, _, seed_pose in seeds:
            cand = _search(seed_pose, f_offs, f_th)[0]
            if cand[0] > best[0]:
                best = cand
    else:
        offs = np.stack(np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1)), -1).reshape(-1, 2)
        th_vals = np.arange(-th_window, th_window + 1e-9, th_step)
        best = _search(tuple(prior), offs, th_vals)[0]

    _, score, pose = best
    if score >= min_score * 0.85:
        pose_ref, score_ref = _refine_subcell(grid, field_, pose, pts_local, prior,
                                              xy_window, th_window, prior_weight)
        if score_ref >= score:
            pose, score = pose_ref, score_ref
    return MatchResult(pose, round(float(score), 3), score >= min_score)


@dataclass(frozen=True)
class PoseEdge:
    """Relative SE(2) constraint from node i to node j in node i's local frame."""
    i: int
    j: int
    z: tuple              # (dx, dy, dth) in frame i
    w_xy: float = 25.0    # information weight (1 / sigma_m^2)
    w_th: float = 100.0   # information weight (1 / sigma_rad^2)
    huber: float = 0.0    # >0 enables Huber robust weighting (for loop edges)


def optimize_pose_graph(poses: Sequence[tuple], edges: Sequence[PoseEdge],
                        iters: int = 12, damping: float = 1e-3) -> list[tuple]:
    """Pure-NumPy 2-D SE(2) pose-graph Gauss-Newton optimizer with node 0 anchored
    and optional Huber loss on loop-closure edges. Solves 100-node graphs in <10 ms."""
    n = len(poses)
    if n <= 1 or not edges:
        return [tuple(p) for p in poses]
    x = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in poses], dtype=np.float64)
    for _ in range(iters):
        H = np.zeros((3 * n, 3 * n), dtype=np.float64)
        b = np.zeros(3 * n, dtype=np.float64)
        # Strong anchor on node 0 so the map frame stays fixed at initial_pose
        H[0:3, 0:3] += np.diag([1e9, 1e9, 1e9])
        for e in edges:
            if not (0 <= e.i < n and 0 <= e.j < n) or e.i == e.j:
                continue
            xi, yi, thi = x[e.i]
            xj, yj, thj = x[e.j]
            ci, si = math.cos(thi), math.sin(thi)
            dx, dy = xj - xi, yj - yi
            # Predicted relative pose in frame i minus measurement z
            ex = ci * dx + si * dy - e.z[0]
            ey = -si * dx + ci * dy - e.z[1]
            eth = wrap(thj - thi - e.z[2])
            err = np.array([ex, ey, eth], dtype=np.float64)
            W = np.array([e.w_xy, e.w_xy, e.w_th], dtype=np.float64)
            if e.huber > 0.0:
                r_norm = math.sqrt(float((W * err * err).sum()))
                if r_norm > e.huber:
                    W = W * (e.huber / r_norm)
            Ji = np.array([
                [-ci, -si, -si * dx + ci * dy],
                [si, -ci, -ci * dx - si * dy],
                [0.0, 0.0, -1.0],
            ], dtype=np.float64)
            Jj = np.array([
                [ci, si, 0.0],
                [-si, ci, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            JiW = Ji.T * W
            JjW = Jj.T * W
            si3, sj3 = 3 * e.i, 3 * e.j
            H[si3:si3 + 3, si3:si3 + 3] += JiW @ Ji
            H[si3:si3 + 3, sj3:sj3 + 3] += JiW @ Jj
            H[sj3:sj3 + 3, si3:si3 + 3] += JjW @ Ji
            H[sj3:sj3 + 3, sj3:sj3 + 3] += JjW @ Jj
            b[si3:si3 + 3] += JiW @ err
            b[sj3:sj3 + 3] += JjW @ err
        H += np.eye(3 * n) * damping
        try:
            delta = -np.linalg.solve(H, b).reshape(n, 3)
        except np.linalg.LinAlgError:
            break
        delta[0] = 0.0
        x[:, 0] += delta[:, 0]
        x[:, 1] += delta[:, 1]
        for k in range(n):
            x[k, 2] = wrap(float(x[k, 2] + delta[k, 2]))
        if float(np.max(np.abs(delta))) < 1e-4:
            break
    return [(float(p[0]), float(p[1]), float(p[2])) for p in x]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

_N8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def traversable(grid: OccupancyGrid, robot_radius: float) -> np.ndarray:
    blocked = _dilate(grid.clean_occupied(), int(math.ceil(robot_radius / grid.res)))
    return grid.free() & ~blocked


def _bfs(trav: np.ndarray, start: tuple, goal_mask: Optional[np.ndarray] = None,
         goal: Optional[tuple] = None) -> tuple:
    """8-connected BFS from `start` (ix, iy). Returns (reached_cell, parents) or (None, parents)."""
    n_y, n_x = trav.shape
    parent = -np.ones(trav.shape, np.int64)
    sx, sy = start
    if not (0 <= sx < n_x and 0 <= sy < n_y):
        return None, parent
    parent[sy, sx] = sy * n_x + sx
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        if (goal is not None and (x, y) == goal) or (goal_mask is not None and goal_mask[y, x]):
            return (x, y), parent
        for dx, dy in _N8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n_x and 0 <= ny < n_y and parent[ny, nx] < 0 and trav[ny, nx]:
                parent[ny, nx] = y * n_x + x
                q.append((nx, ny))
    return None, parent


def _unwind(parent: np.ndarray, end: tuple) -> list:
    n_x = parent.shape[1]
    x, y = end
    out = [(x, y)]
    while True:
        p = int(parent[y, x])
        px, py = p % n_x, p // n_x
        if (px, py) == (x, y):
            break
        x, y = px, py
        out.append((x, y))
    return out[::-1]


def _line_clear(trav: np.ndarray, a: tuple, b: tuple) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(trav[ys, xs].all())


def _shortcut(trav: np.ndarray, cells: list) -> list:
    """Greedy line-of-sight smoothing: fewest straight segments, i.e. fewest
    turn-then-drive steps for the robot."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_clear(trav, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _start_cell(grid: OccupancyGrid, trav: np.ndarray, xy: Sequence[float]) -> Optional[tuple]:
    """Robot's cell, or the nearest traversable one (it may be parked close to
    a wall, inside the inflation)."""
    c = grid.cells(np.array(xy[:2])[None, :])[0]
    for r in range(0, 8):
        ys, xs = np.nonzero(trav[max(0, c[1] - r):c[1] + r + 1, max(0, c[0] - r):c[0] + r + 1])
        if len(xs):
            k = int(np.argmin((xs + max(0, c[0] - r) - c[0]) ** 2 + (ys + max(0, c[1] - r) - c[1]) ** 2))
            return (int(xs[k] + max(0, c[0] - r)), int(ys[k] + max(0, c[1] - r)))
    return None


def plan_path(grid: OccupancyGrid, trav: np.ndarray, start_xy, goal_xy) -> Optional[list]:
    s = _start_cell(grid, trav, start_xy)
    g = _start_cell(grid, trav, goal_xy)
    if s is None or g is None:
        return None
    end, parent = _bfs(trav, s, goal=g)
    if end is None:
        return None
    cells = _shortcut(trav, _unwind(parent, end))
    return [tuple(map(float, grid.world(np.array(c)))) for c in cells]


def frontiers(grid: OccupancyGrid, min_cells: int = 6) -> list:
    """Clusters of free cells that touch unknown space: list of (n_cells, [ix, iy] cells)."""
    free, unk = grid.free(), grid.unknown()
    touch = np.zeros_like(free)
    touch[1:, :] |= unk[:-1, :]
    touch[:-1, :] |= unk[1:, :]
    touch[:, 1:] |= unk[:, :-1]
    touch[:, :-1] |= unk[:, 1:]
    front = free & touch
    seen = np.zeros_like(front)
    out = []
    for y, x in zip(*np.nonzero(front)):
        if seen[y, x]:
            continue
        comp, q = [], deque([(x, y)])
        seen[y, x] = True
        while q:
            cx, cy = q.popleft()
            comp.append((cx, cy))
            for dx, dy in _N8:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid.n and 0 <= ny < grid.n and front[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((nx, ny))
        if len(comp) >= min_cells:
            out.append((len(comp), comp))
    return out


def fit_walls(xy: np.ndarray, res: float, min_len: float = 0.5, gap: float = 0.25,
              max_walls: int = 40) -> list:
    """Straight walls through wall points (x, y): a Hough vote for the strongest line,
    its points (within one cell) split where there is a gap, each run of at least
    min_len kept; repeat on what is left. Returns [(x1, y1, x2, y2), ...], longest first."""
    pts = np.asarray(xy, float).reshape(-1, 2)
    thetas = np.radians(np.arange(0.0, 180.0, 1.0))
    cs, sn = np.cos(thetas), np.sin(thetas)
    out = []
    while len(pts) >= max(3, int(min_len / res)) and len(out) < max_walls:
        rho = pts[:, :1] * cs + pts[:, 1:] * sn                        # (n, 180)
        q = np.round(rho / res).astype(np.int64)
        best = (0, 0, 0)
        for j in range(len(thetas)):
            v, c = np.unique(q[:, j], return_counts=True)
            i = int(np.argmax(c))
            if c[i] > best[0]:
                best = (int(c[i]), j, int(v[i]))
        votes, j, r = best
        if votes < min_len / res * 0.6:
            break
        on = np.abs(rho[:, j] - r * res) <= res * 1.0
        line = pts[on]
        d = np.array([-sn[j], cs[j]])                                  # along the line
        t = np.sort(line @ d)
        cuts = np.nonzero(np.diff(t) > gap)[0]
        p0 = np.array([cs[j], sn[j]]) * r * res
        for seg in np.split(t, cuts + 1):
            if seg[-1] - seg[0] >= min_len:
                a, b = p0 + d * seg[0], p0 + d * seg[-1]
                out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
        pts = pts[~on]
    return sorted(out, key=lambda w: -math.hypot(w[2] - w[0], w[3] - w[1]))


def extent(walls: list) -> Optional[dict]:
    """Size of what the walls enclose along its main directions: the length-weighted
    wall angle (folded to 90 degrees) and the spread of the wall ends across it."""
    if not walls:
        return None
    w = np.array(walls, float)
    ln = np.hypot(w[:, 2] - w[:, 0], w[:, 3] - w[:, 1])
    ang = np.arctan2(w[:, 3] - w[:, 1], w[:, 2] - w[:, 0])
    a = math.atan2(float((ln * np.sin(4 * ang)).sum()), float((ln * np.cos(4 * ang)).sum())) / 4.0
    ends = np.concatenate([w[:, :2], w[:, 2:]])
    u = ends @ np.array([math.cos(a), math.sin(a)])
    v = ends @ np.array([-math.sin(a), math.cos(a)])
    dims = sorted([float(u.max() - u.min()), float(v.max() - v.min())], reverse=True)
    return {"length_m": round(dims[0], 2), "width_m": round(dims[1], 2), "angle_deg": round(math.degrees(a), 1)}


def in_scope(start: Sequence[float], radius_m: Optional[float], no_behind: bool, xy: np.ndarray) -> np.ndarray:
    """Which points (N x 2, map frame) the run may explore: within `radius_m` of
    the start pose (None / <= 0 = no limit) and, with `no_behind`, on the side
    the start pose faces (the half-plane ahead of it)."""
    d = np.asarray(xy, float).reshape(-1, 2) - np.asarray(start[:2], float)
    ok = np.ones(len(d), bool)
    if radius_m and radius_m > 0:
        ok &= np.hypot(d[:, 0], d[:, 1]) <= radius_m
    if no_behind:
        ok &= d[:, 0] * math.cos(start[2]) + d[:, 1] * math.sin(start[2]) >= 0.0
    return ok


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

@dataclass
class SlamConfig:
    size_m: float = 24.0
    res: float = 0.05
    max_range: float = 3.8          # 8x8 ToF max reliable range (~4 m); rays beyond carve free space only
    wedge_rad: float = TOF_WEDGE_RAD  # angular wedge carve per ToF ray (0 disables for narrow-beam lidar)
    robot_radius: float = 0.18
    clearance: float = 0.05         # extra margin on top of the radius
    min_frontier_cells: int = 6
    max_cloud_points: int = 400_000   # 3-D points kept (one per 5 cm cube), oldest dropped first
    pref_clearance: float = 0.15    # paths keep this much more room where the map allows it
    loop_closure: bool = True
    loop_gap: int = 4               # keyframes between this one and an old one before it is a loop
    loop_radius: float = 2.5        # m: an old keyframe this near counts as a revisit
    loop_min_travel: float = 2.5    # m driven since that old keyframe
    loop_window_m: float = 1.5      # the loop match searches this far...
    loop_window_deg: float = 35.0   # ...and this far round
    loop_min_score: float = 0.45    # and needs at least this good a fit (0..1)
    unreachable_ttl_s: float = 60.0 # cooldown before retrying a frontier LocalNav could not reach
    unreachable_radius_m: float = 0.65


@dataclass
class Keyframe:
    kf_id: int
    odom: tuple
    pose: tuple
    score: float
    accepted: bool


@dataclass
class SlamSession:
    config: SlamConfig = field(default_factory=SlamConfig)
    name: str = "default"

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.grid = OccupancyGrid(self.config.size_m, self.config.res)
        self.keyframes: list[Keyframe] = []
        self.initial_pose: tuple = (0.0, 0.0, 0.0)
        self.scope: tuple = (None, False)          # (radius_m, no_behind), see in_scope
        self.goal: Optional[dict] = None
        self._new_layers()
        self._kf_data: dict = {}                   # kf_id -> its readings in base_link, for rebuilds
        self._sub_scans: dict[int, list] = {}      # kf_id -> [(rel_pose_from_kf, data), ...] continuous scans
        self._edges: list[PoseEdge] = []           # pose-graph constraints (odom, match, loop)
        self._unreachable: list[tuple[float, float, float]] = []   # (x, y, expiry_monotonic)
        self._prev_goal_xy: Optional[tuple] = None
        self._base = None                          # the loaded map a rebuild starts from (None: empty)

    # --- helpers --------------------------------------------------------

    def _new_layers(self) -> None:
        """What the ToF grid adds to the wall map. `low` and `drops` are kept apart
        from the walls (they are not what the wall rays are matched against) and are
        sticky: a later ray passing over a step does not erase it."""
        n = self.grid.n
        self.low = np.zeros((n, n), bool)            # low obstacles           [iy, ix]
        self.drops = np.zeros((n, n), bool)          # the floor falls away
        self.cloud = np.zeros((0, 3), np.float32)    # x, y (map), height above the floor

    def _mark(self, layer: np.ndarray, xy: np.ndarray) -> None:
        ij = self.grid.cells(xy)
        ij = ij[self.grid.inside(ij)]
        layer[ij[:, 1], ij[:, 0]] = True

    def _hard_obstacles(self) -> np.ndarray:
        """Despeckled walls plus sticky low obstacles and drop-offs."""
        return self.grid.clean_occupied() | self.low | self.drops

    def _trav(self, extra: float = 0.0, unknown: bool = False) -> np.ndarray:
        """Where the robot's centre may go. unknown: also floor not seen yet (not only seen free), as
        long as nothing known is within reach."""
        r = self.config.robot_radius + self.config.clearance + extra
        blocked = _dilate(self._hard_obstacles(), int(math.ceil(r / self.grid.res)))
        if unknown:
            return ~blocked
        return self.grid.free() & ~blocked

    def _plan_theta(self, start_xy, goal_xy, trav_yx: np.ndarray) -> Optional[list]:
        """Any-angle Theta* with distance-field clearance preference and doorway alignment
        (borrowed from nav_map.py), falling back to BFS+shortcut if nav_map is unavailable."""
        s = _start_cell(self.grid, trav_yx, start_xy)
        g = _start_cell(self.grid, trav_yx, goal_xy)
        if s is None or g is None:
            return None
        if _theta_star is None or _distance_field is None:
            return plan_path(self.grid, trav_yx, start_xy, goal_xy)
        # Window around start and goal for fast planning on a 480x480 grid
        b = int(3.0 / self.grid.res)
        x0 = max(0, min(s[0], g[0]) - b)
        x1 = min(self.grid.n, max(s[0], g[0]) + b + 1)
        y0 = max(0, min(s[1], g[1]) - b)
        y1 = min(self.grid.n, max(s[1], g[1]) + b + 1)
        # nav_map uses [ix, iy] indexing; slam.py uses [iy, ix] -> transpose
        free_xy = trav_yx[y0:y1, x0:x1].T
        occ_xy = self._hard_obstacles()[y0:y1, x0:x1].T
        ls = (s[0] - x0, s[1] - y0)
        lg = (g[0] - x0, g[1] - y0)
        r_min = self.config.robot_radius + self.config.clearance
        field_xy = _distance_field(occ_xy, self.grid.res, cap_m=r_min + 0.5)
        penalty = np.clip((r_min + 0.35 - field_xy) / 0.35, 0.0, 1.0) * 2.0
        cells = _theta_star(free_xy, ls, lg, penalty=penalty)
        if cells is None:
            return plan_path(self.grid, trav_yx, start_xy, goal_xy)
        if _door_approach is not None:
            cells, _ = _door_approach(cells, occ_xy, field_xy, free_xy, r_min, self.grid.res)
        return [tuple(map(float, self.grid.world(np.array([c[0] + x0, c[1] + y0])))) for c in cells]

    def _plan(self, start_xy, goal_xy) -> Optional[list]:
        """Any-angle Theta* with doorway approach; prefers roomy clearance, falls back to tightest."""
        if self.config.pref_clearance > 0:
            p = self._plan_theta(start_xy, goal_xy, self._trav(self.config.pref_clearance))
            if p is not None:
                return p
        p = self._plan_theta(start_xy, goal_xy, self._trav())
        return p if p is not None else self._plan_theta(start_xy, goal_xy, self._trav(unknown=True))

    def clearances(self, path: Sequence, reach: float = 1.0) -> list:
        """For each path point, the nearest mapped wall, low obstacle or drop (m, capped at
        `reach`) along the leg that arrives there (the first: around the start)."""
        with self.lock:
            return self._clearances(path, reach)

    def _clearances(self, path: Sequence, reach: float = 1.0) -> list:
        g = self.grid
        hard = self._hard_obstacles()
        k = int(math.ceil(reach / g.res))

        def near(xy) -> float:
            c = g.cells(np.array([xy[:2]], float))[0]
            y0, x0 = max(0, c[1] - k), max(0, c[0] - k)
            ys, xs = np.nonzero(hard[y0:c[1] + k + 1, x0:c[0] + k + 1])
            if not len(xs):
                return reach
            return min(reach, float(np.min(np.hypot(xs + x0 - c[0], ys + y0 - c[1]))) * g.res)
        out = []
        for i, p in enumerate(path):
            a = path[i - 1] if i else p
            n = max(2, int(math.hypot(p[0] - a[0], p[1] - a[1]) / g.res) + 1)
            out.append(round(min(near((a[0] + (p[0] - a[0]) * t, a[1] + (p[1] - a[1]) * t))
                                 for t in np.linspace(0.0, 1.0, n)), 3))
        return out

    def _integrate(self, pose: tuple, d: dict) -> None:
        """Fold one scan's readings (base_link, see add_keyframe) into the map at `pose`."""
        cfg = self.config
        pts, sensor_xy = d["pts"], d["sensor_xy"]
        world = transform(pose, pts) if len(pts) else pts
        origin = transform(pose, np.array([sensor_xy], float))
        self.grid.integrate(np.repeat(origin, len(world), 0), world, cfg.max_range, wedge_rad=cfg.wedge_rad)
        self.grid.mark_free_disc(pose, cfg.robot_radius)
        if len(d["obstacles"]):
            self.grid.mark_occupied(transform(pose, d["obstacles"]))
        if len(d["clear"]):
            # max_range 0: no ray ends in an obstacle, so these only carve free space.
            self.grid.integrate(np.repeat(origin, len(d["clear"]), 0), transform(pose, d["clear"]),
                                0.0, wedge_rad=cfg.wedge_rad)
        for layer, pts_ in ((self.low, d["low"]), (self.drops, d["drops"])):
            if len(pts_):
                self._mark(layer, transform(pose, pts_))
        if len(d["cloud"]):
            p3 = d["cloud"]
            p3 = np.column_stack([transform(pose, p3[:, :2].astype(float)), p3[:, 2]]).astype(np.float32)
            both = np.concatenate([self.cloud, p3])
            _, first = np.unique(np.round(both / 0.05).astype(np.int32), axis=0, return_index=True)
            self.cloud = both[np.sort(first)][-cfg.max_cloud_points:]

    def _submap(self, ids, include_sub: bool = True) -> OccupancyGrid:
        """Walls only, from these keyframes (and optionally their attached continuous sub-scans)."""
        g = OccupancyGrid(self.config.size_m, self.config.res)
        for i in ids:
            kf, d = self.keyframes[i], self._kf_data.get(i)
            if d is not None and len(d["pts"]):
                origin = transform(kf.pose, np.array([d["sensor_xy"]], float))
                g.integrate(np.repeat(origin, len(d["pts"]), 0), transform(kf.pose, d["pts"]), self.config.max_range)
            if include_sub:
                for rel, sd in self._sub_scans.get(i, ()):
                    if len(sd["pts"]):
                        sp = compose(kf.pose, rel)
                        origin = transform(sp, np.array([sd["sensor_xy"]], float))
                        g.integrate(np.repeat(origin, len(sd["pts"]), 0), transform(sp, sd["pts"]), self.config.max_range)
        return g

    def _rebuild(self) -> None:
        """Rebuild the entire map cleanly from the loaded base, every keyframe at its
        pose-graph-optimized pose, and all continuous sub-scans attached to each keyframe."""
        self.grid = OccupancyGrid(self.config.size_m, self.config.res)
        self._new_layers()
        if self._base is not None:
            self.grid.logodds, self.low, self.drops, self.cloud = (a.copy() for a in self._base)
        for kf in self.keyframes:
            if kf.kf_id in self._kf_data:
                self._integrate(kf.pose, self._kf_data[kf.kf_id])
            for rel, sd in self._sub_scans.get(kf.kf_id, ()):
                self._integrate(compose(kf.pose, rel), sd)

    def _close_loop(self, prior: tuple, pts_hit: np.ndarray) -> Optional[dict]:
        """Detect a revisit against non-adjacent old keyframes, verify by coarse-to-fine
        correlative + sub-cell scan match, and return the verified loop constraint."""
        cfg, k = self.config, len(self.keyframes)
        if not cfg.loop_closure or k <= cfg.loop_gap or len(pts_hit) < 15:
            return None
        old = [kf.kf_id for kf in self.keyframes[:k - cfg.loop_gap]
               if math.hypot(kf.pose[0] - prior[0], kf.pose[1] - prior[1]) < cfg.loop_radius
               and kf.kf_id in self._kf_data]
        if not old:
            return None
        anchor = max(old)
        chain = [kf.pose for kf in self.keyframes[anchor:]] + [prior]
        if sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(chain, chain[1:])) < cfg.loop_min_travel:
            return None
        recent = self._submap(range(max(anchor + 1, k - 4), k))
        mr = match_scan(recent, _blur(recent.clean_occupied(), 2.0), prior, pts_hit)
        here = mr.pose if mr.accepted else prior
        past = self._submap(old)
        mo = match_scan(past, _blur(past.clean_occupied(), 2.0), here, pts_hit, xy_window=cfg.loop_window_m,
                        th_window=math.radians(cfg.loop_window_deg), min_score=cfg.loop_min_score)
        if not mo.accepted:
            return None
        corr_m = math.hypot(mo.pose[0] - here[0], mo.pose[1] - here[1])
        corr_deg = abs(math.degrees(wrap(mo.pose[2] - here[2])))
        if corr_m < 0.03 and corr_deg < 0.8:
            return None                                           # already tightly consistent
        # Measure relative pose from anchor to current node in anchor's local frame
        z_loop = between(self.keyframes[anchor].pose, mo.pose)
        return {"closed": True, "with_kf": anchor, "score": mo.score, "pose": mo.pose,
                "z_loop": z_loop, "correction_m": round(corr_m, 3),
                "correction_deg": round(math.degrees(wrap(mo.pose[2] - here[2])), 2)}

    def map_to_odom(self) -> tuple:
        if not self.keyframes:
            return self.initial_pose
        kf = self.keyframes[-1]
        # map_T_odom = map_T_base ⊕ (odom_T_base)⁻¹
        return compose(kf.pose, between(kf.odom, (0.0, 0.0, 0.0)))

    def pose_of(self, odom: Sequence[float]) -> tuple:
        return compose(self.map_to_odom(), tuple(odom))

    def _pack_scan(self, rays: Sequence, sensor_xy, obstacles: Sequence = (),
                   low: Sequence = (), drops: Sequence = (), clear: Sequence = (),
                   cloud: Sequence = ()) -> tuple:
        cfg = self.config
        r = np.array([[float(b), float(d)] for b, d in rays if d is not None and d > 0], float).reshape(-1, 2)
        pts = np.stack([sensor_xy[0] + r[:, 1] * np.cos(r[:, 0]),
                        sensor_xy[1] + r[:, 1] * np.sin(r[:, 0])], 1) if len(r) else np.zeros((0, 2))
        hit = r[:, 1] < cfg.max_range if len(r) else np.zeros(0, bool)
        c = np.asarray(clear, float).reshape(-1, 2)
        data = {"pts": pts, "sensor_xy": tuple(map(float, sensor_xy)),
                "obstacles": np.asarray(obstacles, float).reshape(-1, 2),
                "clear": np.stack([sensor_xy[0] + c[:, 1] * np.cos(c[:, 0]),
                                   sensor_xy[1] + c[:, 1] * np.sin(c[:, 0])], 1) if len(c) else np.zeros((0, 2)),
                "low": np.asarray(low, float).reshape(-1, 2), "drops": np.asarray(drops, float).reshape(-1, 2),
                "cloud": np.asarray(cloud, np.float32).reshape(-1, 3)}
        return data, pts[hit]

    # --- api ------------------------------------------------------------

    def reset(self, initial_pose: Sequence[float] = (0.0, 0.0, 0.0), keep_map: bool = False,
              radius_m: Optional[float] = None, no_behind: bool = False) -> None:
        """Start a new run. `initial_pose` is where the robot stands on the map at the run's first
        keyframe (any odometry reading: later keyframes follow from the odometry since then). With
        keep_map (after `load`), the robot is told where it is on the saved map — the "user points
        at the start" case."""
        with self.lock:
            if not keep_map:
                self.grid = OccupancyGrid(self.config.size_m, self.config.res)
                self._new_layers()
            self._base = (self.grid.logodds.copy(), self.low.copy(), self.drops.copy(),
                          self.cloud.copy()) if keep_map else None
            self.keyframes = []
            self._kf_data = {}
            self._sub_scans = {}
            self._edges = []
            self._unreachable = []
            self._prev_goal_xy = None
            self.initial_pose = tuple(map(float, initial_pose))
            self.scope = (radius_m, bool(no_behind))
            self.goal = None

    def add_scan(self, odom: Sequence[float], rays: Sequence, sensor_xy=(0.0, 0.0),
                 obstacles: Sequence = (), low: Sequence = (), drops: Sequence = (),
                 clear: Sequence = (), cloud: Sequence = (), plan: bool = False) -> dict:
        """Fold a lightweight continuous ToF scan taken while driving between anchor keyframes.
        Attaches to the latest anchor keyframe so future loop closures re-align it too."""
        odom = tuple(map(float, odom))
        data, _ = self._pack_scan(rays, sensor_xy, obstacles, low, drops, clear, cloud)
        with self.lock:
            if not self.keyframes:
                # No anchor yet: promote to keyframe 0 so continuous streaming can even bootstrap a map
                return self._add_keyframe_locked(odom, data, data["pts"], plan=plan)
            kf = self.keyframes[-1]
            rel = between(kf.odom, odom)
            pose = compose(kf.pose, rel)
            self._integrate(pose, data)
            self._sub_scans.setdefault(kf.kf_id, []).append((rel, data))
            out = {
                "kf_id": kf.kf_id,
                "pose": [round(v, 4) for v in pose],
                "map_to_odom": [round(v, 4) for v in self.map_to_odom()],
                "explored_m2": round(float(self.grid.free().sum()) * self.grid.res ** 2, 2),
            }
            if plan:
                out.update(self._next_goal(pose))
            return out

    def mark_unreachable(self, xy: Sequence[float], ttl_s: Optional[float] = None,
                         pose: Optional[Sequence[float]] = None) -> dict:
        """Blacklist a frontier area temporarily after LocalNav fails to reach it, and
        return the next-best frontier from `pose` (or the latest keyframe pose)."""
        ttl = self.config.unreachable_ttl_s if ttl_s is None else float(ttl_s)
        with self.lock:
            now = time.monotonic()
            self._unreachable = [u for u in self._unreachable if u[2] > now]
            self._unreachable.append((float(xy[0]), float(xy[1]), now + ttl))
            if self._prev_goal_xy is not None and math.hypot(self._prev_goal_xy[0] - xy[0], self._prev_goal_xy[1] - xy[1]) < 0.6:
                self._prev_goal_xy = None
            cur = tuple(map(float, pose)) if pose is not None else (
                self.keyframes[-1].pose if self.keyframes else self.initial_pose)
            return self._next_goal(cur)

    def add_keyframe(self, odom: Sequence[float], rays: Sequence, sensor_xy=(0.0, 0.0),
                     obstacles: Sequence = (), plan: bool = True, low: Sequence = (),
                     drops: Sequence = (), clear: Sequence = (), cloud: Sequence = ()) -> dict:
        """rays: [[bearing_rad, range_m], ...] in base_link at `odom`, measured
        from `sensor_xy` (the pan axis) in base_link. obstacles: [[x, y]] in
        base_link — low obstacles / drop-offs found by the floor probe.
        From the ToF grid, all in base_link too: low, drops: [[x, y]]; clear:
        [[bearing_rad, range_m]] free space with no wall at the end; cloud:
        [[x, y, height_m]] every reading (see the module docstring)."""
        odom = tuple(map(float, odom))
        data, pts_hit = self._pack_scan(rays, sensor_xy, obstacles, low, drops, clear, cloud)
        with self.lock:
            return self._add_keyframe_locked(odom, data, pts_hit, plan=plan)

    def _add_keyframe_locked(self, odom: tuple, data: dict, pts_hit: np.ndarray, plan: bool) -> dict:
        k = len(self.keyframes)
        if k:
            last = self.keyframes[-1]
            z_odom = between(last.odom, odom)
            prior = compose(last.pose, z_odom)
        else:
            z_odom = None
            prior = tuple(self.initial_pose)

        loop = self._close_loop(prior, pts_hit)
        if loop:
            m = MatchResult(tuple(loop["pose"]), loop["score"], True)
        elif k:
            # Match against confirmed keyframes only (excluding unverified in-transit sub-scans
            # attached to k-1, which were placed by raw odometry and would otherwise lock in drift).
            ref_ids = range(max(0, k - 12), k)
            ref_grid = self._submap(ref_ids, include_sub=False) if self._sub_scans.get(k - 1) else self.grid
            field_ = _blur(ref_grid.clean_occupied(), sigma_cells=2.0)
            m = match_scan(ref_grid, field_, prior, pts_hit)
        elif self.grid.clean_occupied().any():
            field_ = _blur(self.grid.clean_occupied(), sigma_cells=2.0)
            m = match_scan(self.grid, field_, prior, pts_hit)
        else:
            m = MatchResult(prior, 0.0, False)

        pose = m.pose if m.accepted else prior
        kf = Keyframe(k, odom, pose, m.score, m.accepted)
        self.keyframes.append(kf)
        self._kf_data[kf.kf_id] = data

        if k and z_odom is not None:
            # Odometry edge (always added, moderate weight)
            self._edges.append(PoseEdge(k - 1, k, z_odom, w_xy=16.0, w_th=40.0))
            if m.accepted and not loop:
                # Sequential scan-match edge (stronger than pure odometry)
                self._edges.append(PoseEdge(k - 1, k, between(self.keyframes[k - 1].pose, pose),
                                            w_xy=80.0, w_th=200.0))

        if loop:
            anchor = loop["with_kf"]
            self._edges.append(PoseEdge(anchor, k, loop["z_loop"], w_xy=100.0, w_th=250.0, huber=0.35))
            opt = optimize_pose_graph([n.pose for n in self.keyframes], self._edges)
            for node, p_opt in zip(self.keyframes, opt):
                node.pose = p_opt
            pose = self.keyframes[-1].pose
            self._rebuild()
            loop = {kk: vv for kk, vv in loop.items() if kk not in ("pose", "z_loop")}
        else:
            self._integrate(pose, data)

        out = {
            "kf_id": kf.kf_id,
            "pose": [round(v, 4) for v in pose],
            "map_to_odom": [round(v, 4) for v in self.map_to_odom()],
            "match": {"score": m.score, "accepted": m.accepted,
                      "prior": [round(v, 4) for v in prior]},
            "loop": loop or {"closed": False},
        }
        if plan:
            out.update(self._next_goal(pose))
        return out

    def _bfs_dist(self, trav: np.ndarray, start: tuple) -> np.ndarray:
        """8-connected BFS shortest path distance (in cells) from `start` to every reachable cell."""
        n_y, n_x = trav.shape
        dist = np.full((n_y, n_x), np.inf, dtype=np.float32)
        sx, sy = start
        if not (0 <= sx < n_x and 0 <= sy < n_y):
            return dist
        dist[sy, sx] = 0.0
        q = deque([(sx, sy)])
        while q:
            x, y = q.popleft()
            d0 = float(dist[y, x])
            for dx, dy in _N8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n_x and 0 <= ny < n_y and trav[ny, nx]:
                    nd = d0 + (1.4142 if dx and dy else 1.0)
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((nx, ny))
        return dist

    def _next_goal(self, pose: Sequence[float]) -> dict:
        trav = self._trav()
        s = _start_cell(self.grid, trav, pose)
        fr = frontiers(self.grid, self.config.min_frontier_cells)
        explored = float(self.grid.free().sum()) * self.grid.res ** 2
        base = {"explored_m2": round(explored, 2), "frontiers": len(fr)}
        if s is None or not fr:
            self.goal = None
            return {**base, "goal": None, "done": True}

        now = time.monotonic()
        self._unreachable = [u for u in self._unreachable if u[2] > now]
        unreach_r = self.config.unreachable_radius_m

        # Exact shortest-path distance from robot to all reachable cells
        dist_cells = self._bfs_dist(trav, s)
        unk = self.grid.unknown()
        info_r = int(round(1.4 / self.grid.res))
        n = self.grid.n
        best = None  # (score, goal_xy, cluster_size, info_cells)

        for size, cells in fr:
            arr = np.array(cells, dtype=np.int64)
            world_pts = self.grid.world(arr)
            ok_scope = in_scope(self.initial_pose, *self.scope, world_pts)
            reach = []
            for (cx, cy), (wx, wy), inside in zip(cells, world_pts, ok_scope):
                if not inside or not trav[cy, cx]:
                    continue
                d_c = float(dist_cells[cy, cx])
                if not math.isfinite(d_c):
                    continue
                if any(math.hypot(wx - ux, wy - uy) < unreach_r for ux, uy, _ in self._unreachable):
                    continue
                reach.append((d_c, cx, cy, float(wx), float(wy)))
            if not reach:
                continue
            # Pick the cluster's reachable cell nearest its centroid (more open than an extreme edge),
            # falling back to the nearest reachable cell in the cluster.
            med_x = float(np.median([r[3] for r in reach]))
            med_y = float(np.median([r[4] for r in reach]))
            rep = min(reach, key=lambda r: math.hypot(r[3] - med_x, r[4] - med_y))
            d_m = rep[0] * self.grid.res
            cx, cy, wx, wy = rep[1], rep[2], rep[3], rep[4]

            # Expected information gain: unknown cells within sensor reach of this frontier
            y0, y1 = max(0, cy - info_r), min(n, cy + info_r + 1)
            x0, x1 = max(0, cx - info_r), min(n, cx + info_r + 1)
            info = float(unk[y0:y1, x0:x1].sum())
            gain = math.sqrt(max(1.0, info)) + 0.5 * math.sqrt(size)

            # Turn penalty + room-continuity hysteresis (finish current area before crossing house)
            bearing = math.atan2(wy - pose[1], wx - pose[0])
            turn_pen = 0.35 * abs(wrap(bearing - pose[2])) if len(pose) >= 3 else 0.0
            cost = d_m + turn_pen + 0.5
            if self._prev_goal_xy is not None and math.hypot(wx - self._prev_goal_xy[0], wy - self._prev_goal_xy[1]) < 1.5:
                cost *= 0.72
            score = gain / cost
            if best is None or score > best[0]:
                best = (score, (wx, wy), size, int(info))

        if best is None:
            self.goal = None
            return {**base, "goal": None, "done": True}

        _, goal_xy, f_size, info_cells = best
        path = self._plan(pose[:2], goal_xy)
        if path is None:
            self.goal = None
            return {**base, "goal": None, "done": True}
        path = [(round(a, 3), round(b, 3)) for a, b in path]
        self._prev_goal_xy = path[-1]
        self.goal = {"xy": list(path[-1]), "path": [list(p) for p in path],
                     "clear_m": self._clearances(path), "frontier_cells": f_size,
                     "info_cells": info_cells}
        return {**base, "goal": self.goal, "done": False}

    def walls(self, min_len: float = 0.5) -> dict:
        """Straight walls fitted to the map's wall cells, and the size of the area they
        enclose (the whole explored area: several rooms make one extent)."""
        with self.lock:
            g = self.grid
            ys, xs = np.nonzero(g.occupied())
            xy = g.world(np.stack([xs, ys], 1)) if len(xs) else np.zeros((0, 2))
        ws = fit_walls(xy, g.res, min_len)
        return {"walls": [{"from": [round(a, 3), round(b, 3)], "to": [round(c, 3), round(d, 3)],
                           "length_m": round(math.hypot(c - a, d - b), 2)} for a, b, c, d in ws],
                "extent": extent(ws)}

    def plan(self, start_xy, goal_xy) -> Optional[list]:
        with self.lock:
            p = self._plan(start_xy, goal_xy)
        return None if p is None else [[round(a, 3), round(b, 3)] for a, b in p]

    def viewpoints(self, start_xy, target_xy, radius: float = 0.8, n: int = 4,
                   start_angle: Optional[float] = None) -> list:
        """Up to n reachable poses spread around target_xy, each facing it,
        visited in a walk-around order, each with the path from the previous.

        The target itself (a plant) is an obstacle, so line of sight is only
        required up to `radius * 0.5` from it.
        """
        with self.lock:
            trav = self._trav(unknown=True)             # the plant's far side is not seen yet: unknown is fine
            occ = self.grid.occupied()
            tx, ty = map(float, target_xy)
            if start_angle is None:
                start_angle = math.atan2(start_xy[1] - ty, start_xy[0] - tx)
            cands = []
            for k in range(n):
                ang = start_angle + 2 * math.pi * k / n
                for rr in (radius, radius * 1.25, radius * 0.8, radius * 1.5):
                    p = (tx + rr * math.cos(ang), ty + rr * math.sin(ang))
                    c = self.grid.cells(np.array([p]))[0]
                    if not self.grid.inside(c) or not trav[c[1], c[0]]:
                        continue
                    stop = (tx + rr * 0.5 * math.cos(ang), ty + rr * 0.5 * math.sin(ang))
                    a = tuple(c)
                    b = tuple(self.grid.cells(np.array([stop]))[0])
                    if not _line_clear(~occ, a, b):
                        continue
                    cands.append({"angle_deg": round(math.degrees(ang), 1),
                                  "pose": [round(p[0], 3), round(p[1], 3),
                                           round(math.atan2(ty - p[1], tx - p[0]), 4)]})
                    break
            out, cur = [], tuple(start_xy[:2])
            for v in cands:           # already in walk-around order
                path = plan_path(self.grid, trav, cur, v["pose"][:2])
                if path is None:
                    continue
                v["path"] = [[round(a, 3), round(b, 3)] for a, b in path]
                out.append(v)
                cur = tuple(v["pose"][:2])
            return out

    # --- persistence ----------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.npz"
        with self.lock, open(path.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, logodds=self.grid.logodds, res=self.grid.res, origin=self.grid.origin,
                     low=self.low, drops=self.drops, cloud=self.cloud,
                     keyframes=json.dumps([kf.__dict__ for kf in self.keyframes]))
        path.with_suffix(".tmp").replace(path)
        return path

    def load(self, directory: Path) -> None:
        data = np.load(directory / f"{self.name}.npz", allow_pickle=False)
        with self.lock:
            self.grid.logodds = data["logodds"].astype(np.float32)
            self.grid.n = self.grid.logodds.shape[0]
            self.grid.res = float(data["res"])
            self.grid.origin = float(data["origin"])
            self._new_layers()
            if "low" in data:                        # maps saved before the ToF layers have none
                self.low, self.drops = data["low"].astype(bool), data["drops"].astype(bool)
                self.cloud = data["cloud"].astype(np.float32)
            self.keyframes = []

    def state(self) -> dict:
        with self.lock:
            last = self.keyframes[-1] if self.keyframes else None
            return {
                "name": self.name,
                "keyframes": len(self.keyframes),
                "pose": list(last.pose) if last else None,
                "map_to_odom": list(self.map_to_odom()),
                "explored_m2": round(float(self.grid.free().sum()) * self.grid.res ** 2, 2),
                "goal": self.goal,
                "low_cells": int(self.low.sum()), "drop_cells": int(self.drops.sum()),
                "cloud_points": int(len(self.cloud)),
            }

    def render(self, scale: int = 2, marks: Sequence = ()) -> bytes:
        """PNG of the map: unknown grey, free white, walls black, low obstacles
        orange, drops (stairs down) magenta, keyframe trail blue, goal path green,
        marks (e.g. plants) red. marks: (label, x, y) or (label, x, y, height_m);
        marks at one spot (a plant stand) are listed one under the other, lowest
        last, each with its height."""
        import io

        from PIL import Image, ImageDraw
        with self.lock:
            g = self.grid
            img = np.full((g.n, g.n, 3), 160, np.uint8)
            img[g.free()] = (250, 250, 250)
            img[g.occupied()] = (20, 20, 20)
            img[self.low] = (240, 150, 30)
            img[self.drops] = (200, 40, 200)
            kfs = [kf.pose for kf in self.keyframes]
            goal = self.goal
        im = Image.fromarray(img[::-1]).resize((g.n * scale, g.n * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)

        def px(x, y):
            c = (np.array([x, y]) - g.origin) / g.res
            return (c[0] * scale, (g.n - c[1]) * scale)
        if goal:
            d.line([px(*p) for p in goal["path"]], fill=(0, 170, 0), width=2)
        if len(kfs) > 1:
            d.line([px(p[0], p[1]) for p in kfs], fill=(40, 90, 220), width=2)
        for p in kfs[-1:]:
            x, y = px(p[0], p[1])
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(40, 90, 220), width=2)
            d.line([(x, y), (x + 12 * math.cos(p[2]), y - 12 * math.sin(p[2]))], fill=(40, 90, 220), width=2)
        rows: dict = {}                                  # text lines already written beside each spot
        for label, x, y, *z in sorted(marks, key=lambda m: -(m[3] or 0.0) if len(m) > 3 else 0.0):
            cx, cy = px(x, y)
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(210, 40, 40))
            spot = (round(cx / 12), round(cy / 12))
            text = str(label) + (f" {z[0]:.2f}m up" if z and z[0] else "")
            d.text((cx + 6, cy - 6 + 11 * rows.get(spot, 0)), text, fill=(210, 40, 40))
            rows[spot] = rows.get(spot, 0) + 1
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
