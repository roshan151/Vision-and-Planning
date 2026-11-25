"""The global layer's map and planner: Theta* over a 2-D occupancy grid on the Mac.

The Pi navigates locally (DWA on a rolling 4 x 4 m costmap) and asks here, about once a
second, for waypoints to its goal (POST /v1/nav/plan). Each request carries the Pi's whole
local costmap, which is folded into this map first, so the map grows as the robot drives:

  * the Pi's cells are in its odometry frame; `odom_to_map` (from visual SLAM, vslam.py; the
    identity until it has one) puts them where they really are, so a room driven round twice
    lines up instead of being drawn twice as odometry drifts;
  * what the Pi saw recently wins: each fold pulls the cells it covers toward its view
    (occupied, seen free) and leaves the cells it has not seen alone;
  * things the camera recognises (a chair, a sofa, a bin: `add_semantic`) are kept in a
    separate layer that fades over a couple of minutes; the ToF misses chair legs and glass
    tables, the camera does not.

`plan()` runs Theta* (any-angle A*: paths are a few straight legs, not a staircase of cells)
over the cells that keep the robot's radius plus a margin from anything occupied, in a window
round the start and goal, with a cost for passing near things so paths keep to the middle of
gaps. Unknown cells count as free (with a small cost): the Pi's DWA and reflex keep it safe in
what neither has seen yet.
"""

from __future__ import annotations

import base64
import heapq
import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

XY = Tuple[float, float]


def compose(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Pose a then b (x, y, heading)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


def inverse(a: Sequence[float]) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


def apply(t: Sequence[float], xy: np.ndarray) -> np.ndarray:
    """Transform points (N, 2) by pose t."""
    c, s = math.cos(t[2]), math.sin(t[2])
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.stack([t[0] + c * xy[:, 0] - s * xy[:, 1], t[1] + s * xy[:, 0] + c * xy[:, 1]], axis=1)


def distance_field(occ: np.ndarray, res: float, cap_m: float = 1.0) -> np.ndarray:
    """Metres from every cell to the nearest occupied one (capped at cap_m)."""
    if not occ.any():
        return np.full(occ.shape, cap_m, np.float32)
    try:
        import cv2
        d = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * res
        return np.minimum(d, cap_m)
    except Exception:                                        # noqa: BLE001 — no OpenCV: rings by FFT
        out = np.full(occ.shape, cap_m, np.float32)
        f_occ = np.fft.rfft2(occ.astype(np.float32))
        for r_m in np.arange(cap_m, 0.0, -res)[::-1]:        # nearest ring first
            r = int(math.ceil(r_m / res))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            k = np.zeros(occ.shape, np.float32)
            disc = (xx ** 2 + yy ** 2) * res * res <= r_m * r_m
            k[yy[disc] % occ.shape[0], xx[disc] % occ.shape[1]] = 1.0
            near = np.fft.irfft2(f_occ * np.fft.rfft2(k), s=occ.shape) > 0.5
            out = np.where(near & (out >= cap_m), r_m, out)
        out[occ] = 0.0
        return out


def _los(free: np.ndarray, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1   # int(): door_approach's points are fractional cells
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(free[xs, ys].all())


def _los_penalty(free: np.ndarray, penalty: Optional[np.ndarray], a, b) -> Optional[float]:
    """None if a -> b hits an obstacle; otherwise the integrated soft clearance cost along the segment."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    if not bool(free[xs, ys].all()):
        return None
    if penalty is None:
        return 0.0
    return float(penalty[xs, ys].mean()) * math.dist(a, b) * 0.10


def theta_star(free: np.ndarray, start, goal, penalty: Optional[np.ndarray] = None,
               max_expand: int = 200_000) -> Optional[list]:
    """Any-angle A* on a bool grid [ix, iy]: cells [start, ..., goal] or None."""
    nx, ny = free.shape
    if not free[goal]:
        return None
    g, parent = {start: 0.0}, {start: start}
    todo, closed = [(math.dist(start, goal), start)], set()
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while todo and len(closed) < max_expand:
        _, c = heapq.heappop(todo)
        if c in closed:
            continue
        if c == goal:
            path = [c]
            while parent[path[-1]] != path[-1]:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(c)
        for dx, dy in steps:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n] or n in closed:
                continue
            extra = 0.0 if penalty is None else float(penalty[n]) * 0.10
            p = parent[c]
            seg_pen = _los_penalty(free, penalty, p, n) if p != c else None
            if seg_pen is not None:
                cost, via = g[p] + math.dist(p, n) + seg_pen, p
            else:
                cost, via = g[c] + math.dist(c, n) * (1.0 + extra), c
            if cost < g.get(n, math.inf):
                g[n], parent[n] = cost, via
                heapq.heappush(todo, (cost + math.dist(n, goal), n))
    return None


SEM_PERSON_M = 0.8      # camera-named things this near the person being followed are not planned round
DETOUR_M = 0.5          # a narrower-margin path is taken when it is at least this much shorter than the roomy one
DOOR_NARROW_M = 0.20     # a gap narrower than 2 x (radius + this) across is a doorway (1.03 m at the Pi's radius:
                         # a 36" door, 0.91 m, with its frame blurred by the ToF's 7.5 deg zones)
DOOR_APPROACH_M = 0.45   # come in on the gap's axis from this far before its middle...
DOOR_EXIT_M = 0.30       # ...and leave along it to this far past
DOOR_REACH_M = 1.0       # how far either way to look for a gap's sides


def _cross(occ: np.ndarray, c, res: float):
    """(width m, angle rad, middle (fx, fy) in cells) of the narrowest cross-section through cell c that has
    an occupied cell on both sides within DOOR_REACH_M; None if every line through c is open on a side."""
    best, reach = None, int(DOOR_REACH_M / res)
    for deg in range(0, 180, 6):
        a = math.radians(deg)
        dx, dy = math.cos(a), math.sin(a)
        ends = []
        for sgn in (1, -1):
            k, hit = 0, False
            while k < reach:
                x, y = int(round(c[0] + sgn * dx * (k + 1))), int(round(c[1] + sgn * dy * (k + 1)))
                if not (0 <= x < occ.shape[0] and 0 <= y < occ.shape[1]):
                    break                                    # off the window: not known to be closed
                if occ[x, y]:
                    hit = True
                    break
                k += 1
            ends.append(k if hit else None)
        if None in ends:
            continue
        w = (ends[0] + ends[1] + 1) * res
        if best is None or w < best[0]:
            mid = (ends[0] - ends[1]) / 2.0
            best = (w, a, (c[0] + dx * mid, c[1] + dy * mid))
    return best


def door_approach(cells: list, occ: np.ndarray, field: np.ndarray, free: np.ndarray, radius_m: float,
                  res: float) -> Tuple[list, Optional[dict]]:
    """The first narrow gap the path goes through (a door, a gap between furniture), entered on its axis:
    an approach point DOOR_APPROACH_M before its middle, the middle, a point DOOR_EXIT_M past it. The robot
    then turns in the open and crosses straight; Nov 22 it arrived at gaps at an angle and stalled in them
    four times, pivoting and backing off with 1-11 cm to spare. Returns (cells, what it did or None); the
    path is unchanged when the robot is already at the gap, the goal is in it, or a leg would not be free."""
    if len(cells) < 2:
        return cells, None
    lim = 2.0 * (radius_m + DOOR_NARROW_M)
    samples = []                                             # (leg index, (fx, fy)) every cell along the path
    for i, (a, b) in enumerate(zip(cells, cells[1:])):
        n = max(1, int(math.ceil(math.dist(a, b))))
        samples += [(i, (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)) for k in range(n)]
    stretch = []                                             # the first run of narrow samples: (sample #, cross)
    for j, (_, q) in enumerate(samples):
        c = (int(round(q[0])), int(round(q[1])))
        cr = _cross(occ, c, res) if field[c] < radius_m + DOOR_NARROW_M else None
        if cr is not None and cr[0] < lim:
            stretch.append((j, cr))
        elif stretch:
            break
    if not stretch or stretch[-1][0] >= len(samples) - 1:
        return cells, None                                   # no gap, or the goal is in it
    j, (w, ang, mid) = min(stretch, key=lambda s: s[1][0])
    leg = samples[j][0]
    u = np.array([-math.sin(ang), math.cos(ang)])            # along the gap's axis...
    tang = np.subtract(cells[leg + 1], cells[leg])
    if float(u @ tang) < 0:
        u = -u                                               # ...the way the path crosses it
    m = np.array(mid)
    along = lambda pt: float((np.asarray(pt, float) - m) @ u) * res
    if along(cells[0]) > -(DOOR_APPROACH_M - 0.1):
        return cells, None                                   # already at the gap: too late to line up
    # Not rounded to whole cells: in a 0.8 m door there are ~8 cm to spare, and rounding cost 5-8 cm of it.
    new_pts = [(float(p[0]), float(p[1])) for p in (m - u * DOOR_APPROACH_M / res, m, m + u * DOOR_EXIT_M / res)]
    before = list(cells[:leg + 1])
    while len(before) > 1 and along(before[-1]) > -DOOR_APPROACH_M + 0.05:
        before.pop()                                         # turning points inside the approach: replaced
    after = list(cells[samples[stretch[-1][0]][0] + 1:])     # the legs after the one the gap ends on
    while len(after) > 1 and along(after[0]) < DOOR_EXIT_M - 0.05:
        after.pop(0)                                         # ...and inside the exit
    out = before + new_pts + after
    if not all(0 <= c[0] <= free.shape[0] - 1 and 0 <= c[1] <= free.shape[1] - 1 for c in new_pts) or \
            not all(_los(free, a, b) for a, b in zip(out, out[1:]) if a != b):
        return cells, None
    return out, {"width_m": round(w, 2), "middle_cell": [round(float(m[0]), 1), round(float(m[1]), 1)]}


class GlobalMap:
    OCC_T, FREE_T = 0.6, -0.5
    TOP, BOTTOM = 3.0, -2.0

    def __init__(self, size_m: float = 24.0, res: float = 0.05, semantic_keep_s: float = 120.0) -> None:
        self.res, self.n = res, int(round(size_m / res))
        self.origin = -size_m / 2.0                          # map coord of cell (0, 0)'s corner, both axes
        self.lo = np.zeros((self.n, self.n), np.float32)     # [ix, iy]
        self.sem_t = np.zeros((self.n, self.n), np.float64)  # when the camera last saw something here (0: never)
        self.sem_label = {}                                  # cell -> label, for the picture
        self.semantic_keep_s = semantic_keep_s
        self.last_path: List[XY] = []
        self.robot: Optional[tuple] = None
        self._lock = threading.Lock()

    def cell(self, xy) -> Tuple[int, int]:
        return int((xy[0] - self.origin) / self.res), int((xy[1] - self.origin) / self.res)

    def centre(self, c) -> XY:
        return self.origin + (c[0] + 0.5) * self.res, self.origin + (c[1] + 0.5) * self.res

    def _inside(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    def fold_local(self, local: dict, odom_to_map: Sequence[float]) -> int:
        """The Pi's costmap snapshot (RollingCostmap.snapshot: corner ox, oy in the odometry frame,
        res, n, cells as bytes 2 occupied / 1 seen free / 0 unknown), put into this map."""
        n, res = int(local["n"]), float(local["res"])
        cells = np.frombuffer(base64.b64decode(local["cells"]), np.uint8).reshape(n, n)
        ii, jj = np.nonzero(cells)
        if not len(ii):
            return 0
        pts = np.stack([local["ox"] + (ii + 0.5) * res, local["oy"] + (jj + 0.5) * res], axis=1)
        m = apply(odom_to_map, pts)
        ix = np.floor((m[:, 0] - self.origin) / self.res).astype(int)
        iy = np.floor((m[:, 1] - self.origin) / self.res).astype(int)
        ok = self._inside(ix, iy)
        ix, iy, kind = ix[ok], iy[ok], cells[ii, jj][ok]
        with self._lock:
            target = np.where(kind == 2, 2.5, -1.5).astype(np.float32)
            self.lo[ix, iy] = np.clip(0.5 * self.lo[ix, iy] + 0.5 * target, self.BOTTOM, self.TOP)
        return int(ok.sum())

    def add_semantic(self, xy: XY, radius_m: float, label: str) -> None:
        """Something the camera recognised standing at map xy, about radius_m across."""
        c = self.cell(xy)
        r = max(1, int(math.ceil(radius_m / self.res)))
        # The same thing seen again replaces where it was put before: each sighting lands a little elsewhere
        # (0.4 m further at 2 m), and their union grew a suitcase into a 1 m blob over the person beside it,
        # so every way to them went round it (Nov 19).
        k = r + int(math.ceil(0.3 / self.res))
        oi, oj = np.mgrid[c[0] - k:c[0] + k + 1, c[1] - k:c[1] + k + 1]
        old = ((oi - c[0]) ** 2 + (oj - c[1]) ** 2 <= k * k) & self._inside(oi, oj)
        ii, jj = np.mgrid[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1]
        disc = ((ii - c[0]) ** 2 + (jj - c[1]) ** 2 <= r * r) & self._inside(ii, jj)
        with self._lock:
            self.sem_t[oi[old], oj[old]] = 0.0
            self.sem_t[ii[disc], jj[disc]] = time.monotonic()
            self.sem_label[c] = label

    def occupied(self) -> np.ndarray:
        fresh = (self.sem_t > 0) & (time.monotonic() - self.sem_t < self.semantic_keep_s)
        return (self.lo > self.OCC_T) | fresh

    def plan(self, start: XY, goal: XY, radius_m: float, margin_m: float = 0.05,
             border_m: float = 2.0, target: Optional[XY] = None) -> Optional[List[XY]]:
        """Waypoints (map frame) from start to goal, the start first; None: no way. `target`: what the
        goal is for (the person the robot stops short of): the goal must be on the target's side of any
        wall. Nov 19: the goal 0.5 m short of a person standing just behind a doorway wall fell in the
        wall's margin, was moved to the robot's side, and the plan flipped between "through the doorway"
        and "straight at the wall" as their tracked spot moved."""
        s, g = self.cell(start), self.cell(goal)
        t = self.cell(target) if target is not None else g
        if not all(0 <= c[0] < self.n and 0 <= c[1] < self.n for c in (s, g, t)):
            return None
        b = int(border_m / self.res)
        x0, x1 = max(0, min(s[0], g[0], t[0]) - b), min(self.n, max(s[0], g[0], t[0]) + b + 1)
        y0, y1 = max(0, min(s[1], g[1], t[1]) - b), min(self.n, max(s[1], g[1], t[1]) + b + 1)
        with self._lock:
            wall = self.lo[x0:x1, y0:y1] > self.OCC_T
            fresh = (self.sem_t[x0:x1, y0:y1] > 0) & (time.monotonic() - self.sem_t[x0:x1, y0:y1] < self.semantic_keep_s)
            unknown = (np.abs(self.lo[x0:x1, y0:y1]) < 0.3) & ~wall & ~fresh
        # A wall cell with no wall cell round it is one stray ToF reading: planned round, it closed 0.6 m of
        # open floor (Nov 23: a third of all wall cells). The Pi's own costmap and reflex still see it.
        p = np.pad(wall, 1)
        wall &= (sum(np.roll(np.roll(p, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1))[1:-1, 1:-1] > 1)
        if target is not None:                                 # what the camera named near the person is them, or
            ii, jj = np.ogrid[:wall.shape[0], :wall.shape[1]]  # theirs (Nov 23: 172 "suitcases", median 0.6 m away)
            fresh &= np.hypot(ii - (t[0] - x0), jj - (t[1] - y0)) * self.res > SEM_PERSON_M
        occ = wall | fresh
        field = distance_field(occ, self.res, cap_m=radius_m + 0.6)
        ls, lg, lt = (s[0] - x0, s[1] - y0), (g[0] - x0, g[1] - y0), (t[0] - x0, t[1] - y0)
        rb = 0.15 / self.res                                   # the target's own body: its last 0.15 m

        def sees(a, b) -> bool:
            """No occupied cell on the line a -> b, except within rb of b (the person's own body)."""
            n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
            xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
            ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
            far = np.hypot(xs - b[0], ys - b[1]) > rb
            return not bool(occ[xs[far], ys[far]].any())
        penalty = np.clip((radius_m + 0.4 - field) / 0.4, 0.0, 1.0) * 2.0 + unknown * 0.3
        # The roomy margin first (+1.5 cm corner-swing room for the square chassis when pivoting); if that
        # path is the long way round, narrower ones too (down to 0: the Pi's radius_m already holds DWA's
        # body_radius_m + margin + PLAN_EXTRA_M), and the shorter way wins by DETOUR_M.
        best = None
        for margin in (margin_m, 0.03, 0.0):
            free = field >= radius_m + margin
            strict = free.copy()                               # where it may END: never the relaxed area below
            if not free[ls]:                                   # starting in the margin: may leave it, but only
                ii, jj = np.ogrid[:free.shape[0], :free.shape[1]]   # near the start: relaxed everywhere, the path's
                near = np.hypot(ii - ls[0], jj - ls[1]) <= 0.4 / self.res   # end hugged a wall too (Nov 19 sim)
                free |= (field >= max(0.0, field[ls] - 1e-3)) & near
                free[ls] = True
            goal_c = lg
            wrong_side = target is not None and not sees(lg, lt)
            if not strict[goal_c] or wrong_side:               # a person's spot is "occupied" by them; or the goal
                r = int(0.6 / self.res)                        # is across a wall from what it is for
                cx, cy = lt if wrong_side else lg              # wrong side: look round the target instead
                ii, jj = np.mgrid[max(0, cx - r):min(free.shape[0], cx + r + 1),
                                  max(0, cy - r):min(free.shape[1], cy + r + 1)]
                ok = strict[ii, jj]                            # a place to stop with the full margin round it: a goal
                if not ok.any():                               # in the start's relaxed area (0.22 m off a wall) was one
                    continue                                   # DWA could not reach (Nov 19 sim, doorway2)
                d = np.where(ok, np.hypot(ii - lg[0], jj - lg[1]), np.inf)   # nearest to the asked goal...
                goal_c = None
                for k in np.argsort(d, axis=None)[:int(np.isfinite(d).sum())]:
                    c = np.unravel_index(k, d.shape)
                    cand = (int(ii[c]), int(jj[c]))
                    if target is None or sees(cand, lt):       # ...that can see the target
                        goal_c = cand
                        break
                if goal_c is None:
                    continue
            cells = theta_star(free, ls, goal_c, penalty)
            if not cells:
                continue
            cells, door = door_approach(cells, occ, field, free, radius_m, self.res)
            if door is not None:
                c = door.pop("middle_cell")
                door["middle"] = [round(v, 3) for v in self.centre((c[0] + x0, c[1] + y0))]
            path = [start] + [self.centre((c[0] + x0, c[1] + y0)) for c in cells[1:]]
            if goal_c == lg:
                path[-1] = tuple(goal)
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            if best is None or length < best[0] - DETOUR_M:
                best = (length, path, door, margin)
            if length <= math.dist(start, path[-1]) * 1.3 + 0.3:    # no long way round: keep the most room
                break
        if best is None:
            self.last_path, self.last_door = [], None
            return None
        _, path, self.last_door, self.last_margin = best
        self.last_path = path
        return path

    def stats(self) -> dict:
        with self._lock:
            occ = self.occupied()
            return {"occupied_cells": int(occ.sum()), "free_cells": int((self.lo < self.FREE_T).sum()),
                    "semantic_cells": int((self.sem_t > 0).sum()), "size_m": self.n * self.res, "res_m": self.res}

    def png(self) -> bytes:
        """The map as a picture: occupied black, seen free white, unknown grey, camera-found objects
        orange, the last path blue, the robot red. North (map +y) up."""
        from io import BytesIO

        from PIL import Image, ImageDraw
        with self._lock:
            occ = self.occupied()
            img = np.full((self.n, self.n, 3), 160, np.uint8)
            img[self.lo < self.FREE_T] = 255
            img[self.lo > self.OCC_T] = 0
            sem = (self.sem_t > 0) & (time.monotonic() - self.sem_t < self.semantic_keep_s)
            img[sem] = (255, 140, 0)
            del occ
        pic = Image.fromarray(np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2))))
        d = ImageDraw.Draw(pic)
        to_px = lambda xy: ((xy[0] - self.origin) / self.res, self.n - (xy[1] - self.origin) / self.res)  # noqa: E731
        if len(self.last_path) > 1:
            d.line([to_px(p) for p in self.last_path], fill=(0, 90, 255), width=2)
        if self.robot is not None:
            x, y = to_px(self.robot)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 0, 0))
            d.line([x, y, x + 10 * math.cos(self.robot[2]), y - 10 * math.sin(self.robot[2])], fill=(220, 0, 0), width=2)
        buf = BytesIO()
        pic.save(buf, "PNG")
        return buf.getvalue()
