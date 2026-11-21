"""Dense 3-D reconstruction of a room from explore's scan frames: a TSDF.

At every keyframe the robot stands still and its head looks round (mission._scan_grid). Each
camera look is sent here (POST /v1/recon/frame) with what places it: the odometry pose it was taken
from, the head's pan and tilt, the robot's pitch, the head geometry, and the 8x8 ToF grid seen with
it. Afterwards (POST /v1/recon/build, or `python -m vision_service.recon home`) every frame:

  1. gets its camera pose in the map:  map <- keyframe pose (the SLAM's, after loop closure)
     <- odometry since that keyframe <- pitch <- head mount <- pan <- tilt (+ the head's tilt offset)
     <- camera axes. Roll is taken as level: the robot is standing still on the floor;
  2. gets a depth for every pixel: Depth Anything gives relative inverse depth d (right up to
     1/z = a*d + b, a and b unknown per frame). The ToF zones that land in the picture give true
     depths; a and b are fitted to them (RANSAC, then least squares on the inliers). A frame whose
     fit has too few inliers is skipped. Each frame is also fitted on half its zones and checked on
     the other half: that is the depth accuracy in the report;
  3. is fused into a truncated signed distance volume: per voxel a weighted running average of
     clamp((measured depth - voxel depth) / trunc, -1, 1), weights falling with distance (more so
     beyond what the ToF measured) and with the fit's error. Voxels far behind a surface are not
     touched (unknown); voxels in front are carved free, so one frame's outliers are voted out.

The surface is the volume's zero crossing (marching cubes), written as a coloured PLY mesh (MeshLab,
Blender, CloudCompare open it) with a JSON report beside it. All runs of one map (a new map starts a
new map index; continue_map runs share the map's frame) are fused together.

A camera calibration, if there is one, goes in <slam_dir>/recon/camera.json:
    {"width": 640, "fx": .., "fy": .., "cx": .., "cy": .., "dist": [k1, k2, p1, p2, k3]}
Without it the camera is a pinhole from the head's horizontal field of view, with no distortion.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

TOF_SIZE, TOF_FOV_DEG = 8, 60.0
TOF_RANGE_M = (0.15, 3.5)           # nearer is the robot itself; further is past the sensor
R_OPT = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # camera axes (x right, y down, z ahead) as robot axes


# --------------------------------------------------------------------------
# storage: one folder per run, <slam_dir>/recon/<map>-m<index>-<stamp>/
# --------------------------------------------------------------------------

class ReconStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.run: Optional[Path] = None
        self._n = 0

    def start(self, name: str, map_index: Optional[int]) -> Path:
        """A new run (each /v1/slam/reset): its frames are numbered by the keyframe they belong to."""
        self.run = self.root / f"{name}-m{map_index or 0}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.run.mkdir(parents=True, exist_ok=True)
        self._n = 0
        return self.run

    def add_frame(self, jpeg: bytes, meta: dict, kf_id: int, name: str) -> str:
        if self.run is None:
            self.start(name, None)
        self._n += 1
        stem = self.run / f"{kf_id:04d}-{self._n:04d}"
        stem.with_suffix(".jpg").write_bytes(jpeg)
        stem.with_suffix(".json").write_text(json.dumps({**meta, "kf_id": kf_id}))
        return str(stem.with_suffix(".jpg"))

    def save_keyframes(self, keyframes: Sequence) -> None:
        """The run's keyframe poses (rewritten with every map save, so loop closures reach the mesh)."""
        if self.run is not None:
            kfs = [dict(k.__dict__) if hasattr(k, "__dict__") else dict(k) for k in keyframes]
            for k in kfs:
                k["odom"], k["pose"] = [float(v) for v in k["odom"]], [float(v) for v in k["pose"]]
            (self.run / "keyframes.json").write_text(json.dumps(kfs))

    def runs(self, name: str) -> list:
        """Every run of the map's newest map index (the runs that share its frame)."""
        pat = re.compile(rf"^{re.escape(name)}-m(\d+)-\d{{8}}-\d{{6}}$")
        found = [(int(m.group(1)), p) for p in self.root.glob(f"{name}-m*") if (m := pat.match(p.name))]
        if not found:
            return []
        top = max(i for i, _ in found)
        return sorted(p for i, p in found if i == top)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _compose(a, b) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def _between(a, b) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, b[2] - a[2])


def camera_pose(base: Sequence[float], pitch_deg: Optional[float], pan_deg: float, tilt_deg: float,
                geom: dict) -> tuple:
    """(R, t): camera axes and position in the map. The same chain as Head.project: tilt (plus the
    head's tilt offset: how far up it really looks at tilt 0), then pan, then the robot's pitch."""
    R_base = _rz(base[2]) @ _ry(-math.radians(pitch_deg or 0.0))           # pitch > 0: nose up
    R_head = _rz(math.radians(pan_deg)) @ _ry(-math.radians(tilt_deg + geom.get("tilt_offset_deg", 0.0)))
    mount = np.array([geom.get("pan_axis_x_m", 0.0), geom.get("pan_axis_y_m", 0.0), geom.get("height_m", 0.22)])
    return R_base @ R_head @ R_OPT, np.array([base[0], base[1], 0.0]) + R_base @ mount


def intrinsics(w: int, h: int, hfov_deg: float, calib: Optional[dict]) -> tuple:
    """(K, dist or None) for a w x h picture."""
    if calib:
        k = w / float(calib["width"])
        K = np.array([[calib["fx"] * k, 0, calib["cx"] * k], [0, calib["fy"] * k, calib["cy"] * k], [0, 0, 1.0]])
        return K, np.asarray(calib.get("dist", [0, 0, 0, 0, 0]), float)
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]), None


def tof_points(rows, geom: dict) -> np.ndarray:
    """The ToF zones' readings as points in the CAMERA frame (N x 3). The ToF sits beside the camera
    (the few cm between them are left out) and looks tof_pan/tilt_offset_deg away from it."""
    mid = (TOF_SIZE - 1) / 2.0
    zone = TOF_FOV_DEG / TOF_SIZE
    off = R_OPT.T @ _rz(math.radians(geom.get("tof_pan_offset_deg", 0.0))) \
        @ _ry(-math.radians(geom.get("tof_tilt_offset_deg", 0.0))) @ R_OPT
    pts = []
    for r, row in enumerate(rows or []):
        for c, d in enumerate(row):
            if d is None or not (TOF_RANGE_M[0] <= d <= TOF_RANGE_M[1]):
                continue
            up, left = (mid - r) * zone, (mid - c) * zone
            v = np.array([-math.tan(math.radians(left)), -math.tan(math.radians(up)), 1.0])
            v = d * v if geom.get("tof_perpendicular", True) else d * v / np.linalg.norm(v)
            pts.append(off @ v)
    return np.array(pts, float).reshape(-1, 3)


# --------------------------------------------------------------------------
# metric depth from the relative map and the ToF
# --------------------------------------------------------------------------

def _fit_once(d: np.ndarray, z: np.ndarray, tol_abs: float = 0.05, tol_rel: float = 0.08):
    """RANSAC over pairs for 1/z = a d + b, then weighted least squares on the inliers.
    Returns (a, b, inlier mask) or None."""
    n = len(d)
    if n < 4:
        return None
    best, inv = None, 1.0 / z
    for i in range(n):
        for j in range(i + 1, n):
            if abs(d[i] - d[j]) < 1e-6:
                continue
            a = (inv[i] - inv[j]) / (d[i] - d[j])
            b = inv[i] - a * d[i]
            if a <= 0:
                continue
            q = a * d + b
            pred = np.where(q > 1e-6, 1.0 / np.maximum(q, 1e-6), np.inf)
            inl = np.abs(pred - z) < np.maximum(tol_abs, tol_rel * z)
            err = float(np.abs(pred - z)[inl].sum())
            if best is None or inl.sum() > best[0].sum() or (inl.sum() == best[0].sum() and err < best[1]):
                best = (inl, err)
    if best is None or best[0].sum() < 3:
        return None
    m = best[0]
    w = z[m] ** 2                                   # inverse-depth residuals, weighted back to metres
    A = np.stack([d[m], np.ones(m.sum())], 1) * w[:, None]
    sol, *_ = np.linalg.lstsq(A, inv[m] * w, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if a <= 0:
        return None
    pred = 1.0 / np.maximum(a * d + b, 1e-6)
    return a, b, np.abs(pred - z) < np.maximum(tol_abs, tol_rel * z)


def fit_scale(d: np.ndarray, z: np.ndarray, zone_rc: np.ndarray, min_inliers: int = 6) -> Optional[dict]:
    """The frame's 1/z = a d + b, and its check on held-out zones (fit on one checkerboard half of the
    zones, measured on the other, both ways)."""
    fit = _fit_once(d, z)
    if fit is None or fit[2].sum() < max(min_inliers, 0.5 * len(z)):
        return None
    a, b, inl = fit
    errs = []
    black = (zone_rc[:, 0] + zone_rc[:, 1]) % 2 == 0
    for train in (black, ~black):
        f = _fit_once(d[train], z[train])
        if f is not None and (~train).any():
            pred = 1.0 / np.maximum(f[0] * d[~train] + f[1], 1e-6)
            errs += list(np.abs(pred - z[~train]) / z[~train])
    return {"a": a, "b": b, "inliers": int(inl.sum()), "zones": int(len(z)),
            "fit_rel_err": float(np.median(np.abs(1.0 / np.maximum(a * d[inl] + b, 1e-6) - z[inl]) / z[inl])),
            "holdout_rel_err": float(np.median(errs)) if errs else None, "z_max": float(z[inl].max()),
            "_holdout": errs}


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

@dataclass
class Frame:
    path: Path
    meta: dict
    R: np.ndarray
    t: np.ndarray


def load_frames(runs: Sequence[Path]) -> tuple:
    """Every frame of these runs whose keyframe was posted, with its camera pose; and what was skipped."""
    frames, skipped = [], {}
    for run in runs:
        kf_file = run / "keyframes.json"
        kfs = {k["kf_id"]: k for k in json.loads(kf_file.read_text())} if kf_file.exists() else {}
        for meta_file in sorted(run.glob("*.json")):
            if meta_file.name == "keyframes.json":
                continue
            meta = json.loads(meta_file.read_text())
            kf = kfs.get(meta.get("kf_id"))
            if kf is None:
                skipped["no keyframe"] = skipped.get("no keyframe", 0) + 1
                continue
            base = _compose(kf["pose"], _between(kf["odom"], meta["odom"]))
            R, t = camera_pose(base, meta.get("pitch_deg"), meta["pan_deg"], meta["tilt_deg"], meta.get("geom", {}))
            frames.append(Frame(meta_file.with_suffix(".jpg"), meta, R, t))
    return frames, skipped


def decode_rgb(frame: Frame, calib: Optional[dict]) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    if calib:
        K, dist = intrinsics(bgr.shape[1], bgr.shape[0], 80.0, calib)
        bgr = cv2.undistort(bgr, K, dist)
    return bgr[:, :, ::-1].copy()


# --------------------------------------------------------------------------
# the volume
# --------------------------------------------------------------------------

@dataclass
class TsdfConfig:
    voxel_m: float = 0.03
    trunc_m: float = 0.10           # a few times the per-frame depth error
    max_depth_m: float = 4.5
    min_depth_m: float = 0.2
    z_range_m: tuple = (-0.05, 2.8)  # floor to ceiling
    max_voxels: int = 25_000_000
    edge_rel: float = 0.06          # a pixel whose depth jumps more than this (relative) to a neighbour is an edge: dropped
    w_max: float = 60.0
    min_weight: float = 1.0         # surface kept where at least this much weight agreed


class Reconstruction:
    def __init__(self, frames: Sequence[Frame], cfg: TsdfConfig = TsdfConfig(), calib: Optional[dict] = None) -> None:
        self.frames, self.cfg, self.calib = list(frames), cfg, calib
        if not self.frames:
            raise ValueError("no frames to reconstruct from")
        cams = np.array([f.t for f in self.frames])
        reach = cfg.max_depth_m
        lo = np.array([cams[:, 0].min() - reach, cams[:, 1].min() - reach, cfg.z_range_m[0]])
        hi = np.array([cams[:, 0].max() + reach, cams[:, 1].max() + reach, cfg.z_range_m[1]])
        v = cfg.voxel_m
        while np.prod(np.ceil((hi - lo) / v)) > cfg.max_voxels:
            v *= 1.15
        self.voxel, self.origin = v, lo
        self.shape = tuple(int(s) for s in np.ceil((hi - lo) / v))
        self.F = np.ones(self.shape, np.float32)
        self.W = np.zeros(self.shape, np.float32)
        self.C = np.zeros(self.shape + (3,), np.float32)
        self.report = {"frames": len(self.frames), "fused": 0, "skipped": {}, "per_frame": []}

    def _skip(self, why: str, frame: Frame) -> None:
        self.report["skipped"][why] = self.report["skipped"].get(why, 0) + 1
        self.report["per_frame"].append({"frame": frame.path.name, "skipped": why})

    def add(self, frame: Frame, rgb: np.ndarray, rel: np.ndarray) -> bool:
        """Fuse one frame. rel: the model's relative inverse depth, any resolution."""
        import cv2
        cfg, meta = self.cfg, frame.meta
        if meta.get("settled") is False:
            self._skip("head moving", frame)
            return False
        h, w = rgb.shape[:2]
        K, _ = intrinsics(w, h, meta.get("geom", {}).get("hfov_deg", 80.0), self.calib)
        rel = cv2.resize(np.asarray(rel, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        # 1. scale: the ToF zones that land in the picture
        P = tof_points(meta.get("tof_rows"), meta.get("geom", {}))
        if len(P) < 6:
            self._skip("no ToF", frame)
            return False
        u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
        v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
        half = max(2, int(0.3 * K[0, 0] * math.tan(math.radians(TOF_FOV_DEG / TOF_SIZE))))
        keep = (u >= half) & (u < w - half) & (v >= half) & (v < h - half)
        if keep.sum() < 6:
            self._skip("ToF outside the picture", frame)
            return False
        d = np.array([np.median(rel[int(vv) - half:int(vv) + half + 1, int(uu) - half:int(uu) + half + 1])
                      for uu, vv in zip(u[keep], v[keep])])
        rc = self._zone_rc(meta.get("tof_rows"))[keep]      # same order as tof_points
        fit = fit_scale(d, P[keep, 2], rc)
        if fit is None:
            self._skip("depth does not fit the ToF", frame)
            return False

        # 2. metric depth per pixel, with edges and out-of-range pixels dropped
        q = fit["a"] * rel + fit["b"]
        z = np.where(q > 1.0 / cfg.max_depth_m, 1.0 / np.maximum(q, 1e-6), 0.0).astype(np.float32)
        z[z < cfg.min_depth_m] = 0.0
        gy, gx = np.gradient(z)
        z[np.hypot(gx, gy) > cfg.edge_rel * np.maximum(z, 1e-3)] = 0.0

        # 3. fuse: the voxels within reach of this camera
        self._integrate(frame, rgb, z, K, fit)
        self.report["fused"] += 1
        self.report["per_frame"].append({"frame": frame.path.name, **{k: (round(v_, 4) if isinstance(v_, float) else v_)
                                                                       for k, v_ in fit.items() if not k.startswith("_")}})
        self.report.setdefault("_holdout", []).extend(fit["_holdout"])
        return True

    @staticmethod
    def _zone_rc(rows) -> np.ndarray:
        out = [(r, c) for r, row in enumerate(rows or []) for c, d in enumerate(row)
               if d is not None and TOF_RANGE_M[0] <= d <= TOF_RANGE_M[1]]
        return np.array(out, int).reshape(-1, 2)

    def _integrate(self, frame: Frame, rgb: np.ndarray, z: np.ndarray, K: np.ndarray, fit: dict) -> None:
        cfg, v = self.cfg, self.voxel
        h, w = z.shape
        reach = cfg.max_depth_m + cfg.trunc_m
        i0 = np.maximum(np.floor((frame.t - reach - self.origin) / v).astype(int), 0)
        i1 = np.minimum(np.ceil((frame.t + reach - self.origin) / v).astype(int), self.shape)
        if np.any(i1 <= i0):
            return
        ax = [self.origin[k] + (np.arange(i0[k], i1[k]) + 0.5) * v for k in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        pc = (np.stack([X, Y, Z], -1).reshape(-1, 3) - frame.t) @ frame.R          # world -> camera
        zc = pc[:, 2]
        ok = zc > cfg.min_depth_m
        uu = np.full(len(zc), -1, np.int64)
        vv = np.full(len(zc), -1, np.int64)
        uu[ok] = np.round(K[0, 0] * pc[ok, 0] / zc[ok] + K[0, 2]).astype(np.int64)
        vv[ok] = np.round(K[1, 1] * pc[ok, 1] / zc[ok] + K[1, 2]).astype(np.int64)
        ok &= (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        idx = np.nonzero(ok)[0]
        D = z[vv[idx], uu[idx]]
        sdf = D - zc[idx]
        use = (D > 0) & (sdf > -cfg.trunc_m)
        idx, D, sdf = idx[use], D[use], sdf[use]
        f = np.clip(sdf / cfg.trunc_m, -1.0, 1.0).astype(np.float32)
        # weight: monocular depth gets worse with distance, and past what the ToF measured it is extrapolated
        beyond = np.maximum(0.0, D - 1.2 * fit["z_max"])
        wt = (1.0 / (1.0 + (D / 2.0) ** 2)) * np.exp(-beyond) / (1.0 + (fit["fit_rel_err"] / 0.03) ** 2)
        wt = wt.astype(np.float32)
        sub = tuple(slice(i0[k], i1[k]) for k in range(3))
        Fs, Ws, Cs = self.F[sub].reshape(-1), self.W[sub].reshape(-1), self.C[sub].reshape(-1, 3)
        Wn = Ws[idx] + wt
        Fs[idx] = (Ws[idx] * Fs[idx] + wt * f) / Wn
        col = rgb[vv[idx], uu[idx]].astype(np.float32)
        Cs[idx] = (Ws[idx, None] * Cs[idx] + wt[:, None] * col) / Wn[:, None]
        Ws[idx] = np.minimum(Wn, cfg.w_max)
        self.F[sub], self.W[sub], self.C[sub] = (Fs.reshape(self.F[sub].shape), Ws.reshape(self.W[sub].shape),
                                                 Cs.reshape(self.C[sub].shape))

    def finish(self, out_ply: Path) -> dict:
        """Marching cubes on the zero crossing; faces touching a voxel no frame saw (or saw too little)
        are dropped, so the far side of the truncation band does not become a false surface."""
        from skimage.measure import marching_cubes
        rep = self.report
        hold = rep.pop("_holdout", [])
        rep["depth_check"] = {"holdout_median_rel_err": round(float(np.median(hold)), 4) if hold else None,
                              "holdout_p90_rel_err": round(float(np.percentile(hold, 90)), 4) if hold else None,
                              "zones": len(hold)}
        rep.update(voxel_m=round(self.voxel, 4), trunc_m=self.cfg.trunc_m, grid=list(self.shape),
                   origin=[round(float(x), 3) for x in self.origin])
        if rep["fused"] == 0 or not (self.F < 0).any():
            rep.update(vertices=0, faces=0, mesh=None)
            return rep
        verts, faces, _, _ = marching_cubes(self.F, level=0.0, spacing=(self.voxel,) * 3)
        gi = verts / self.voxel
        lo = np.clip(np.floor(gi).astype(int), 0, np.array(self.shape) - 1)
        hi = np.clip(lo + 1, 0, np.array(self.shape) - 1)
        good = np.ones(len(verts), bool)
        for cx in (lo[:, 0], hi[:, 0]):
            for cy in (lo[:, 1], hi[:, 1]):
                for cz in (lo[:, 2], hi[:, 2]):
                    good &= self.W[cx, cy, cz] >= self.cfg.min_weight
        faces = faces[good[faces].all(1)]
        used = np.unique(faces)
        remap = -np.ones(len(verts), np.int64)
        remap[used] = np.arange(len(used))
        faces, verts, gi = remap[faces], verts[used], gi[used]
        near = np.clip(np.round(gi).astype(int), 0, np.array(self.shape) - 1)
        colors = np.clip(self.C[near[:, 0], near[:, 1], near[:, 2]], 0, 255).astype(np.uint8)
        verts = verts + self.origin + 0.5 * self.voxel        # voxel i's value is at its centre
        write_ply(out_ply, verts.astype(np.float32), colors, faces.astype(np.int32))
        ext = verts.max(0) - verts.min(0) if len(verts) else np.zeros(3)
        rep.update(vertices=int(len(verts)), faces=int(len(faces)), mesh=str(out_ply),
                   extent_m=[round(float(x), 2) for x in ext])
        return rep


def write_ply(path: Path, verts: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vv = np.empty(len(verts), vdt)
    vv["x"], vv["y"], vv["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    vv["r"], vv["g"], vv["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    ff = np.empty(len(faces), fdt)
    ff["n"], ff["i"] = 3, faces
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(vv.tobytes())
        fh.write(ff.tobytes())
    tmp.replace(path)


def load_calib(root: Path) -> Optional[dict]:
    p = Path(root) / "camera.json"
    return json.loads(p.read_text()) if p.exists() else None


def reconstruct(runs: Sequence[Path], depth_fn: Callable[[np.ndarray], np.ndarray], out_ply: Path,
                cfg: TsdfConfig = TsdfConfig(), calib: Optional[dict] = None,
                progress: Callable[[int, int], None] = lambda i, n: None) -> dict:
    """The whole thing, synchronously (the CLI; the service drives the same steps itself so that the
    depth model runs through its GPU queue)."""
    t0 = time.monotonic()
    frames, skipped = load_frames(runs)
    rec = Reconstruction(frames, cfg, calib)
    rec.report["skipped"].update(skipped)
    for i, fr in enumerate(rec.frames, 1):
        rgb = decode_rgb(fr, calib)
        rec.add(fr, rgb, depth_fn(rgb))
        progress(i, len(rec.frames))
    rep = rec.finish(Path(out_ply))
    rep["runs"], rep["seconds"] = [p.name for p in runs], round(time.monotonic() - t0, 1)
    Path(out_ply).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .settings import settings
    ap = argparse.ArgumentParser(description="3-D mesh of a map from explore's scan frames (TSDF)")
    ap.add_argument("map", nargs="?", default="home", help="map name (its newest map index), or a run folder")
    ap.add_argument("--voxel", type=float, default=TsdfConfig.voxel_m)
    ap.add_argument("--trunc", type=float, default=TsdfConfig.trunc_m)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = ReconStore(settings.slam_dir / "recon")
    path = Path(a.map)
    runs = [path] if path.is_dir() else store.runs(a.map)
    if not runs:
        print(f"no recon frames for {a.map!r} under {store.root}")
        return 1
    from .backends.real import DepthAnythingEstimator
    depth = DepthAnythingEstimator(settings.depth_model, settings.device)
    out = store.root / f"{runs[0].name.rsplit('-', 2)[0]}.ply"
    rep = reconstruct(runs, depth.depth_map, out, TsdfConfig(voxel_m=a.voxel, trunc_m=a.trunc), load_calib(store.root),
                      lambda i, n: print(f"\r{i}/{n} frames", end="", flush=True))
    print()
    print(json.dumps({k: v for k, v in rep.items() if k != "per_frame"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
