"""Phase 0 of saved maps: replay recorded debug sessions offline and measure how well the robot would
find itself on a saved map from a random start, before any of it is built into the live stack.

  python -m vision_service.replay geo debug/<A> [debug/<B>]   saved map from A; starts taken from B
                                                              (no B: A's two halves against each other)
  python -m vision_service.replay vpr debug/<A> debug/<B>     on the Mac: which image model names the place

geo  builds the saved map from session A's own ToF readings and wheel/gyro poses. The recorded global
     maps are not used: on Nov 22 odom_to_map was corrupted by the old PyMonoVO fallback. For random
     moments in B it takes the next few seconds of ToF as the start-up look-around, searches the WHOLE
     map for the pose (every cell, every 2 deg; exact, by FFT correlation) and compares with the truth.
     Truth across sessions is B's map registered onto A's once, with all of B (far more data than one
     look-around). A pose is only claimed when it is clearly better than every other place
     (score and second-best ratio): "correct or not yet, never confidently wrong".
vpr  for every frame of B, the nearest frames of A by DINOv2-small (what the service already runs) and by
     MegaLoc, scored against the geo truth; times both on this machine.

Reports go to debug/replay/. geo needs only numpy (+ Pillow for the pictures).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from datetime import datetime

import numpy as np

RES = 0.05                 # map cell, m (the nav maps' resolution)
TILT_OFFSET_DEG = 21.0     # head_tilt_offset_deg: uploads before the real-tilt change hold commanded tilts
HEAD_H = 0.22              # head_height_m
ZONE_DEG = 7.5             # 8x8 zones over 60 deg
SELF_M = 0.15              # self_range_m
SELF_ZONES = {(7, 0): 0.22, (7, 1): 0.15, (7, 2): 0.15, (7, 3): 0.15, (6, 0): 0.30}   # tof_self_zones
H_BAND = (0.06, 1.2)       # a return this high above the floor is an obstacle; lower is floor
MAX_R = 2.6                # ToF reach used (99th percentile of real returns 2.41 m)
GOOD_M, GOOD_DEG = 0.30, 10.0   # an answer this close to the truth is correct


# --------------------------------------------------------------------------- poses

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def transform(p, xy):
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.stack([c * xy[:, 0] - s * xy[:, 1] + p[0], s * xy[:, 0] + c * xy[:, 1] + p[1]], 1)


# --------------------------------------------------------------------------- sessions

def load(session: str, clean: bool = True) -> list:
    """Every upload with a pose, a ToF grid and the head's MEASURED angles, in time order.

    clean: only readings the map can trust. Uploads without a camera frame carry the last COMMANDED
    gimbal angle (wrong while the head moves); the ToF grid can be up to ~0.5 s older than the pose;
    and while the base pivots (up to 170 deg/s) or the head pans fast, that age smears walls by tens
    of cm. Self-consistency on Nov 22: 0.31 with everything, 0.73 with these filters.
    Uploads with "tof_seq" (Pi debug_upload since Nov 23) carry EVERY grid with its own time, head
    angles and nav pose: those are used as they are, unfiltered, and the camera frame with its own pose."""
    out = []
    for fn in sorted(glob.glob(os.path.join(session, "[0-9]*.json"))):
        d = json.load(open(fn))
        ex, f, geom = d.get("extras") or {}, d.get("frame") or {}, d.get("geometry") or {}
        on_gimbal = bool(geom.get("imu_on_gimbal"))
        tilt_off = float(geom.get("head_tilt_offset_deg") or TILT_OFFSET_DEG)
        raw_pitch = ex.get("pitch_deg")
        if "tof_seq" in d:
            for g in d["tof_seq"]:
                if g.get("pose") is not None:
                    still = g.get("head_still")
                    if on_gimbal:
                        gtilt = (raw_pitch + tilt_off) if (still is not False and raw_pitch is not None) else g["tilt_deg"]
                        gpitch = 0.0
                    else:
                        gtilt, gpitch = g["tilt_deg"], raw_pitch or 0.0
                    out.append(dict(t=g["t"], pose=np.array(g["pose"], float), pan=g["pan_deg"],
                                    tilt=gtilt, pitch=gpitch, rows=g["rows"],
                                    age_ms=0, image=None, timed=True))
            if f.get("pose") is not None and d.get("image"):
                out.append(dict(t=f["t"], pose=np.array(f["pose"], float), pan=f["pan_deg"], tilt=f["tilt_deg"],
                                pitch=0.0 if on_gimbal else (raw_pitch or 0.0), rows=None, age_ms=0, timed=True,
                                image=os.path.join(session, d["image"])))
            continue
        rows = (d.get("tof") or {}).get("rows")
        if not ex.get("pose") or not rows or f.get("pan_deg") is None:
            continue
        pan, tilt = f["pan_deg"], f["tilt_deg"]
        if "head_tilt_offset_deg" not in geom:
            tilt += TILT_OFFSET_DEG            # recorded before tilts were stored as real
        if on_gimbal and raw_pitch is not None and f.get("settled") is not False:
            tilt, pitch = raw_pitch + tilt_off, 0.0
        else:
            pitch = 0.0 if on_gimbal else (raw_pitch or 0.0)
        out.append(dict(t=datetime.fromisoformat(d["ts"]).timestamp(), pose=np.array(ex["pose"], float),
                        pan=pan, tilt=tilt, pitch=pitch, rows=rows,
                        age_ms=d["tof"].get("age_ms") or 0,
                        image=os.path.join(session, d["image"]) if d.get("image") else None))
    out.sort(key=lambda f: f["t"])
    for i, f in enumerate(out):
        a, b = out[max(0, i - 1)], out[min(len(out) - 1, i + 1)]
        dt = max(b["t"] - a["t"], 1e-3)
        f["turn"] = abs(math.degrees(wrap(b["pose"][2] - a["pose"][2]))) / dt
        f["pan_rate"] = abs(b["pan"] - a["pan"]) / dt
    if clean:
        out = [f for f in out if f.get("timed") or (f["age_ms"] <= 200 and f["turn"] < 30 and f["pan_rate"] < 30)]
    return out


_up = np.radians((3.5 - np.arange(8))[:, None] * ZONE_DEG * np.ones((1, 8)))
_left = np.radians((3.5 - np.arange(8))[None, :] * ZONE_DEG * np.ones((8, 1)))
_near = np.full((8, 8), SELF_M)
for (r, c), m in SELF_ZONES.items():
    _near[r, c] = max(SELF_M, m)


def tof_points(f) -> np.ndarray:
    """The frame's ToF returns as (x ahead, y left, h above floor) in the robot's frame, as head.project
    does it (ranges perpendicular to the sensor, pan axis at the robot centre)."""
    if not f.get("rows"):
        return np.zeros((0, 3))                # a camera frame on its own (new-style uploads)
    d = np.array([[np.nan if v is None else v for v in row] for row in f["rows"]], float)
    ok = np.isfinite(d) & (d >= _near) & (d <= MAX_R)
    if not ok.any():
        return np.zeros((0, 3))
    fwd, left, up = np.ones(ok.sum()), np.tan(_left[ok]), np.tan(_up[ok])
    t, p, b = math.radians(f["tilt"]), math.radians(f["pan"]), math.radians(f["pitch"])
    fwd, up = fwd * math.cos(t) - up * math.sin(t), fwd * math.sin(t) + up * math.cos(t)
    fwd, left = fwd * math.cos(p) - left * math.sin(p), fwd * math.sin(p) + left * math.cos(p)
    fwd, up = fwd * math.cos(b) - up * math.sin(b), fwd * math.sin(b) + up * math.cos(b)
    r = d[ok]
    return np.stack([r * fwd, r * left, HEAD_H + r * up], 1)


# --------------------------------------------------------------------------- the saved map

class Map:
    """Occupancy from ToF hits and the free space along their rays, plus a likelihood field
    (walls blurred by sigma) that scans are scored against."""

    def __init__(self, frames: list, sigma_cells: float = 1.5, free_penalty: float = 0.0):
        hits, ends, starts = [], [], []
        for f in frames:
            P = tof_points(f)
            if not len(P):
                continue
            W = transform(f["pose"], P[:, :2])
            wall = (P[:, 2] >= H_BAND[0]) & (P[:, 2] <= H_BAND[1])
            hits.append(W[wall])
            ends.append(W[(P[:, 2] <= H_BAND[1])])
            starts.append(np.repeat(f["pose"][None, :2], int((P[:, 2] <= H_BAND[1]).sum()), 0))
        hits, ends, starts = np.concatenate(hits), np.concatenate(ends), np.concatenate(starts)
        path = np.array([f["pose"][:2] for f in frames])
        lo = np.minimum(ends.min(0), path.min(0)) - 1.0
        hi = np.maximum(ends.max(0), path.max(0)) + 1.0
        self.origin, self.res = lo, RES
        self.shape = tuple(np.ceil((hi - lo) / RES).astype(int))
        n_hit = self._count(hits)
        # free: samples along each ray, stopping a cell short of its end
        L = np.linalg.norm(ends - starts, axis=1)
        k = np.arange(0, int(MAX_R / RES * 2))
        fr = (k[None, :] * RES / 2) / np.maximum(L[:, None], 1e-6)
        keep = fr * L[:, None] < L[:, None] - RES
        S = starts[:, None, :] + fr[..., None] * (ends - starts)[:, None, :]
        n_free = self._count(S[keep])
        self.occ = (n_hit >= 2) & (n_hit >= 0.3 * n_free)
        # where the robot can stand: seen free, or driven over
        drove = np.zeros(self.shape, bool)
        for x, y in path:
            i, j = self.cell(np.array([[x, y]]))[0]
            drove[max(0, i - 3):i + 4, max(0, j - 3):j + 4] = True
        self.free = ((n_free >= 2) & ~self.occ) | drove
        self.field = self._blur(self.occ.astype(float), sigma_cells) * (2 * math.pi * sigma_cells ** 2)
        self.field = np.minimum(self.field, 1.0)
        # a reading that lands where the map has seen open floor is evidence against the pose
        # (off by default: with Nov 22's leftover smear it pushed true poses down, 9/24 -> 3/24)
        self.field[(n_free >= 3) & (self.field < 0.1)] -= free_penalty

    def cell(self, xy):
        return np.floor((xy - self.origin) / self.res).astype(int)

    def _count(self, xy):
        ij = self.cell(xy)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1])
        flat = ij[ok, 0] * self.shape[1] + ij[ok, 1]
        return np.bincount(flat, minlength=self.shape[0] * self.shape[1]).reshape(self.shape)

    def _blur(self, a, s):
        r = int(3 * s) + 1
        g = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        g /= g.sum()
        a = np.apply_along_axis(lambda v: np.convolve(v, g, "same"), 0, a)
        return np.apply_along_axis(lambda v: np.convolve(v, g, "same"), 1, a)

    def score(self, pose, pts):
        ij = self.cell(transform(pose, pts))
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1])
        return float(self.field[ij[ok, 0], ij[ok, 1]].sum() / len(pts))

    def picture(self, path, marks=()):
        from PIL import Image, ImageDraw
        img = np.full(self.shape + (3,), 128, np.uint8)
        img[self.free] = (235, 235, 235)
        img[self.occ] = (20, 20, 20)
        im = Image.fromarray(np.transpose(img, (1, 0, 2))[::-1]).resize(
            (self.shape[0] * 3, self.shape[1] * 3), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        for (x, y, th), col in marks:
            i, j = (np.array([x, y]) - self.origin) / self.res
            u, v = i * 3, (self.shape[1] - j) * 3
            dr.ellipse((u - 5, v - 5, u + 5, v + 5), outline=col, width=2)
            dr.line((u, v, u + 14 * math.cos(th), v - 14 * math.sin(th)), fill=col, width=2)
        im.save(path)


# --------------------------------------------------------------------------- global search

def global_match(m: Map, pts: np.ndarray, th_step_deg: float = 2.0, anywhere: bool = False):
    """The best pose for `pts` (robot frame) over the whole map: every cell, every th_step_deg,
    scored as the mean likelihood of the points. Exact (no pruning): one FFT correlation per
    heading. Returns best pose, its score, and the best score at least 0.5 m / 20 deg away."""
    H, W = m.shape
    K = int(math.ceil(np.abs(pts).max() / m.res)) + 1
    N, M = H + 2 * K + 1, W + 2 * K + 1
    Ff = np.fft.rfft2(m.field, s=(N, M))
    ths = np.radians(np.arange(0.0, 360.0, th_step_deg))
    vol = np.empty((len(ths), H, W), np.float32)
    allowed = np.ones(m.shape, bool) if anywhere else m.free
    for i, th in enumerate(ths):
        c, s = math.cos(th), math.sin(th)
        q = np.floor(0.5 + np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
                     / m.res).astype(int) + K
        k = np.zeros((N, M))
        np.add.at(k, (q[:, 0], q[:, 1]), 1.0)
        corr = np.fft.irfft2(Ff * np.conj(np.fft.rfft2(k)), s=(N, M))
        vol[i] = np.where(allowed, np.roll(corr, (K, K), (0, 1))[:H, :W] / len(pts), 0.0)
    a, i, j = np.unravel_index(int(np.argmax(vol)), vol.shape)
    best = float(vol[a, i, j])
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    near = np.hypot(ii - i, jj - j) * m.res <= 0.5
    dth = np.abs((np.arange(len(ths)) - a + len(ths) / 2) % len(ths) - len(ths) / 2) * th_step_deg
    vol[(dth <= 20)[:, None, None] & near[None]] = 0.0
    second = float(vol.max())
    x, y = m.origin + (np.array([i, j]) + 0.5) * m.res
    pose = refine(m, np.array([x, y, ths[a]]), pts)
    return pose, best, second


def refine(m, pose, pts, cells=2, deg=3.0, step=0.5):
    best = (m.score(pose, pts), pose)
    for dth in np.radians(np.arange(-deg, deg + 1e-9, step)):
        for di in range(-cells, cells + 1):
            for dj in range(-cells, cells + 1):
                p = np.array([pose[0] + di * m.res, pose[1] + dj * m.res, wrap(pose[2] + dth)])
                s = m.score(p, pts)
                if s > best[0]:
                    best = (s, p)
    return best[1]


def look_around(frames: list, i0: int, seconds: float):
    """ToF obstacle points from frames[i0] on for `seconds`, in the robot's frame at i0 (odometry is
    good over a few seconds), one per cell; and the bearings they cover."""
    p0, t0, pts = frames[i0]["pose"], frames[i0]["t"], []
    inv0 = inverse(p0)
    for f in frames[i0:]:
        if f["t"] - t0 > seconds:
            break
        P = tof_points(f)
        wall = (P[:, 2] >= H_BAND[0]) & (P[:, 2] <= H_BAND[1])
        if wall.any():
            pts.append(transform(compose(inv0, f["pose"]), P[wall, :2]))
    if not pts:
        return np.zeros((0, 2)), 0
    pts = np.concatenate(pts)
    pts = (np.unique(np.floor(pts / RES), axis=0) + 0.5) * RES
    bins = np.unique((np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) // 10).astype(int))
    return pts, 10 * len(bins)


def sweep(src: Map, pose, coverage_deg: float, rng, reach: float = 2.4, step_deg: float = 2.5,
          dropout: float = 0.33, noise_m: float = 0.02) -> np.ndarray:
    """A deliberate start-up look-around, simulated: rays every step_deg over coverage_deg centred
    ahead, cast through `src`'s walls (dilated one cell) out to the ToF's reach, a third of them
    lost as the real sensor loses them. Points in the robot's frame. Built from the OTHER session's
    map, so it carries that run's own view of the room, not the map it is matched against."""
    occ = src.occ.copy()
    occ[1:] |= src.occ[:-1]; occ[:-1] |= src.occ[1:]; occ[:, 1:] |= src.occ[:, :-1]; occ[:, :-1] |= src.occ[:, 1:]
    pts = []
    for b in np.radians(np.arange(-coverage_deg / 2, coverage_deg / 2 + 1e-9, step_deg)):
        if rng.random() < dropout:
            continue
        a = pose[2] + b
        for d in np.arange(0.15, reach, RES / 2):
            i, j = src.cell(np.array([[pose[0] + d * math.cos(a), pose[1] + d * math.sin(a)]]))[0]
            if not (0 <= i < src.shape[0] and 0 <= j < src.shape[1]):
                break
            if occ[i, j]:
                r = d + rng.normal(0, noise_m)
                pts.append((r * math.cos(b), r * math.sin(b)))
                break
    return np.array(pts).reshape(-1, 2)


def geo_sweep(a_dir: str, b_dir: str, n: int, out: str, seed: int = 0):
    """Starts at poses B really stood at, with a simulated look-around seen through B's map,
    searched for on A's map; truth through the B->A registration."""
    A, B = load(a_dir), load(b_dir)
    mA, mB = Map(A), Map(B)
    T = np.array(json.load(open(os.path.join(out, "T_AB.json"))))
    rng = np.random.default_rng(seed)
    rows = []
    for i0 in rng.choice(len(B), size=min(n, len(B)), replace=False):
        pB = B[i0]["pose"]
        truth = compose(T, pB)
        ti, tj = mA.cell(truth[None, :2])[0]
        inside = 0 <= ti < mA.shape[0] and 0 <= tj < mA.shape[1] and mA.free[max(0, ti - 6):ti + 7,
                                                                              max(0, tj - 6):tj + 7].any()
        for cov in (120, 240, 360):
            pts = sweep(mB, pB, cov, rng)
            if len(pts) < 10:
                continue
            t0 = time.time()
            est, s1, s2 = global_match(mA, pts)
            e_m, e_deg = pose_err(est, truth)
            rows.append(dict(case="sweep: B's view on A's map", i0=int(i0), window_s=cov, points=len(pts),
                             coverage_deg=cov, inside=bool(inside), score=round(s1, 3), second=round(s2, 3),
                             ratio=round(s2 / max(s1, 1e-9), 3), err_m=round(e_m, 3), err_deg=round(e_deg, 1),
                             correct=bool(e_m < GOOD_M and e_deg < GOOD_DEG), secs=round(time.time() - t0, 2)))
    json.dump(rows, open(os.path.join(out, "sweep.json"), "w"), indent=1)
    print("(window column = look-around coverage in degrees)")
    summarise(rows)
    return rows


def pose_err(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1])), abs(math.degrees(wrap(a[2] - b[2])))


# --------------------------------------------------------------------------- geo

def geo(a_dir: str, b_dir: str | None, n: int, windows: list, out: str, seed: int = 0):
    os.makedirs(out, exist_ok=True)
    A = load(a_dir)
    cases = []          # (map, query frames, truth transform map<-query frame, label)
    if b_dir:
        B = load(b_dir)
        mA, mB = Map(A), Map(B)
        t0 = time.time()
        occ = np.argwhere(mB.occ)
        ptsB = mB.origin + (occ + 0.5) * RES
        T, s1, s2 = global_match(mA, ptsB, 1.0, anywhere=True)
        print(f"B registered onto A: T={np.round(T, 3).tolist()} score {s1:.3f} vs next {s2:.3f} "
              f"({time.time() - t0:.1f} s, {len(ptsB)} wall cells)")
        mA.picture(os.path.join(out, "map_A.png"))
        mB.picture(os.path.join(out, "map_B.png"))
        _overlay(mA, transform(T, ptsB), os.path.join(out, "B_on_A.png"))
        json.dump(np.asarray(T).tolist(), open(os.path.join(out, "T_AB.json"), "w"))
        cases.append((mA, B, T, "A map, starts from B"))
        mBA = mB
        Tinv = inverse(T)
        cases.append((mBA, A, Tinv, "B map, starts from A"))
    else:
        mid = A[len(A) // 2]["t"]
        h1, h2 = [f for f in A if f["t"] < mid], [f for f in A if f["t"] >= mid]
        cases += [(Map(h1), h2, np.zeros(3), "1st half map, starts from 2nd half"),
                  (Map(h2), h1, np.zeros(3), "2nd half map, starts from 1st half")]
    rng = np.random.default_rng(seed)
    rows = []
    for m, Q, T, label in cases:
        tmax = Q[-1]["t"] - max(windows)
        idx = [i for i, f in enumerate(Q) if f["t"] <= tmax]
        picks = rng.choice(idx, size=min(n, len(idx)), replace=False)
        for i0 in picks:
            truth = compose(T, Q[i0]["pose"])
            ti, tj = m.cell(truth[None, :2])[0]
            inside = 0 <= ti < m.shape[0] and 0 <= tj < m.shape[1] and \
                m.free[max(0, ti - 6):ti + 7, max(0, tj - 6):tj + 7].any()
            for w in windows:
                pts, cov = look_around(Q, int(i0), w)
                if len(pts) < 20:
                    rows.append(dict(case=label, i0=int(i0), window_s=w, points=len(pts), coverage_deg=cov,
                                     inside=bool(inside), skipped=True))
                    continue
                t0 = time.time()
                est, s1, s2 = global_match(m, pts)
                e_m, e_deg = pose_err(est, truth)
                rows.append(dict(case=label, i0=int(i0), window_s=w, points=len(pts), coverage_deg=cov,
                                 inside=bool(inside), score=round(s1, 3), second=round(s2, 3),
                                 ratio=round(s2 / max(s1, 1e-9), 3), err_m=round(e_m, 3),
                                 err_deg=round(e_deg, 1), correct=bool(e_m < GOOD_M and e_deg < GOOD_DEG),
                                 secs=round(time.time() - t0, 2), est=np.round(est, 3).tolist(),
                                 truth=np.round(truth, 3).tolist()))
        print(f"{label}: {len(picks)} starts done")
    json.dump(rows, open(os.path.join(out, "geo.json"), "w"), indent=1)
    summarise(rows)
    return rows


def _overlay(m, pts, path):
    from PIL import Image
    img = np.full(m.shape + (3,), 128, np.uint8)
    img[m.free] = (235, 235, 235)
    img[m.occ] = (20, 20, 20)
    ij = m.cell(pts)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < m.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < m.shape[1])
    img[ij[ok, 0], ij[ok, 1]] = (220, 30, 30)
    Image.fromarray(np.transpose(img, (1, 0, 2))[::-1]).resize(
        (m.shape[0] * 3, m.shape[1] * 3), Image.NEAREST).save(path)


def summarise(rows, ratio_max=0.85, score_min=0.0):
    """Per case and window: how often the best pose is right, and with the acceptance rule
    (second-best below ratio_max of the best) how many are claimed right / claimed wrong / held."""
    keys = sorted({(r["case"], r["window_s"]) for r in rows})
    print(f"\nacceptance: second-best < {ratio_max} x best\n"
          f"{'case':38s} {'win':>4s} {'n':>3s} {'cov':>5s} {'best right':>10s} {'claimed ok':>10s} "
          f"{'claimed WRONG':>13s} {'held':>5s} {'med err':>8s} {'secs':>5s}")
    for case, w in keys:
        rs = [r for r in rows if r["case"] == case and r["window_s"] == w and not r.get("skipped") and r["inside"]]
        if not rs:
            continue
        ok = [r for r in rs if r["correct"]]
        claim = [r for r in rs if r["ratio"] < ratio_max and r["score"] >= score_min]
        good = [r for r in claim if r["correct"]]
        print(f"{case:38s} {w:4.0f} {len(rs):3d} {np.median([r['coverage_deg'] for r in rs]):5.0f} "
              f"{len(ok) / len(rs):10.0%} {len(good) / len(rs):10.0%} {len(claim) - len(good):13d} "
              f"{len(rs) - len(claim):5d} {np.median([r['err_m'] for r in ok]) if ok else float('nan'):8.3f} "
              f"{np.median([r['secs'] for r in rs]):5.1f}")
    outside = [r for r in rows if not r.get("skipped") and not r["inside"]]
    if outside:
        bad = [r for r in outside if r["ratio"] < ratio_max]
        print(f"starts outside the saved map: {len(outside)}, claimed anyway: {len(bad)}")


# --------------------------------------------------------------------------- vpr (on the Mac)

def vpr_eval(A: list, B: list, T, embed, k: int = 5):
    """Top-1 / top-k: one of B's frame's k nearest A frames (by `embed`) is within 1.5 m and 45 deg
    of where B's camera really was (B's pose through the B->A registration, plus head pan).
    Returns (top1, topk, median seconds per image, ceiling: share of B frames with any A frame near)."""
    def cam(f, Tm=None):
        p = f["pose"] if Tm is None else compose(Tm, f["pose"])
        return np.array([p[0], p[1], wrap(p[2] + math.radians(f["pan"]))])

    def emb(fs):
        v, ts = [], []
        for f in fs:
            t0 = time.time()
            v.append(np.asarray(embed(f), float).ravel())
            ts.append(time.time() - t0)
        v = np.array(v)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9), ts
    dbA, tA = emb(A)
    qB, tB = emb(B)
    camA = np.array([cam(f) for f in A])
    order = np.argsort(-(qB @ dbA.T), axis=1)[:, :k]
    hit1 = hitk = reach = 0
    for qi, f in enumerate(B):
        c = cam(f, T)
        near = (np.hypot(camA[:, 0] - c[0], camA[:, 1] - c[1]) < 1.5) & \
               (np.abs(np.degrees((camA[:, 2] - c[2] + math.pi) % (2 * math.pi) - math.pi)) < 45)
        hit1 += bool(near[order[qi, 0]])
        hitk += bool(near[order[qi]].any())
        reach += bool(near.any())
    return hit1 / len(B), hitk / len(B), float(np.median(tA[3:] + tB[3:])), reach / len(B)


def vpr(a_dir: str, b_dir: str, out: str, k: int = 5):
    """On the Mac: DINOv2-small (the plant encoder the service already loads) against MegaLoc, every
    camera frame of B against every camera frame of A, timed on the service's device (mps)."""
    import torch
    from PIL import Image
    from .settings import Settings
    from .backends.real import Dinov2PlantEncoder

    reg = os.path.join(out, "T_AB.json")
    if not os.path.exists(reg):
        raise SystemExit(f"run `geo {a_dir} {b_dir} --out {out}` first: it writes T_AB.json")
    T = np.array(json.load(open(reg)))
    A = [f for f in load(a_dir, clean=False) if f["image"]]
    B = [f for f in load(b_dir, clean=False) if f["image"]]
    cfg = Settings()
    rgb = lambda f: np.asarray(Image.open(f["image"]).convert("RGB"))     # noqa: E731
    models = {}
    dino = Dinov2PlantEncoder(cfg.plant_model, cfg.device)
    models["dinov2-small (loaded today)"] = lambda f: dino.embed(rgb(f))
    try:
        import torchvision.transforms as tv
        ml = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval().to(cfg.device)
        tf = tv.Compose([tv.Resize((322, 322)), tv.ToTensor(),
                         tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        def megaloc(f):
            with torch.inference_mode():
                return ml(tf(Image.open(f["image"]).convert("RGB")).unsqueeze(0).to(cfg.device)).float().cpu().numpy()
        models["megaloc"] = megaloc
    except Exception as exc:                                  # noqa: BLE001
        print("MegaLoc not loaded:", exc)
    res = {}
    for name, fn in models.items():
        top1, topk, sec, ceil = vpr_eval(A, B, T, fn, k)
        res[name] = dict(top1=round(top1, 3), topk=round(topk, 3), ceiling=round(ceil, 3),
                         ms_per_image=round(1000 * sec, 1), images=len(A) + len(B))
        print(f"{name:30s} top-1 {top1:.0%}  top-{k} {topk:.0%}  (ceiling {ceil:.0%})  "
              f"{1000 * sec:.0f} ms/image (incl. decode) on {cfg.device}")
    json.dump(res, open(os.path.join(out, "vpr.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["geo", "vpr"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("-n", type=int, default=40, help="random starts per case")
    ap.add_argument("--windows", default="3,6,12,24", help="look-around lengths, s")
    ap.add_argument("--out", default="debug/replay")
    ap.add_argument("--sweep", action="store_true", help="geo with simulated 360 deg start-up look-arounds")
    args = ap.parse_args()
    if args.mode == "geo" and args.sweep:
        geo_sweep(args.a, args.b, args.n, args.out)
    elif args.mode == "geo":
        geo(args.a, args.b, args.n, [float(w) for w in args.windows.split(",")], args.out)
    else:
        vpr(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
