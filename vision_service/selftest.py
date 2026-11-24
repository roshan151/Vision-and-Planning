"""Live end-to-end test of the RUNNING vision service: every endpoint, ORB-SLAM3 included.

    make test                                   (= python -m vision_service.selftest)
    python -m vision_service.selftest --url http://127.0.0.1:8080 [--orb-frames 150] [--keep]

It assumes the service is up (make start) and the robot idle: it resets the follow map (/v1/nav/reset)
and the explore SLAM session, which it runs under its own map name "selftest", so the robot's maps
and plants are not touched. Everything it adds (a face, a plant, a debug session, the selftest map
and its 3-D scan) is deleted again at the end (--keep: left for a look).

ORB-SLAM3 is fed a stretch of a recorded follow run (the newest debug session with enough camera
frames and poses), the way the Pi streams it, and must track without crashing; the service must still
report ORB-SLAM3 ready at the very end. Without a recording that part is skipped.

PASS / WARN (worked, but not as well as it should: does not fail the run) / SKIP / FAIL.
Exit code 1 if anything failed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .settings import settings

NAME = "selftest"                      # map / face / debug-session name used for everything this adds


class Skip(Exception):
    pass


class Warn(Exception):
    pass


class Run:
    def __init__(self, url: str, timeout: float = 60.0):
        self.c = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        self.rows = []

    def check(self, name: str, fn):
        t0 = time.monotonic()
        try:
            note = fn() or ""
            status = "PASS"
        except Skip as e:
            status, note = "SKIP", str(e)
        except Warn as e:
            status, note = "WARN", str(e)
        except AssertionError as e:
            status, note = "FAIL", str(e) or "assertion failed"
        except Exception as e:                                         # noqa: BLE001
            status, note = "FAIL", f"{type(e).__name__}: {e}"
        dt = time.monotonic() - t0
        self.rows.append((status, name, dt, note))
        colour = {"PASS": "32", "WARN": "33", "SKIP": "36", "FAIL": "31"}[status]
        print(f"  \033[{colour}m{status:4s}\033[0m  {name:34s} {dt:6.1f} s  {note}", flush=True)

    def ok(self, r: httpx.Response, *codes) -> dict | bytes:
        codes = codes or (200,)
        assert r.status_code in codes, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json() if "json" in r.headers.get("content-type", "") else r.content


# ------------------------------------------------------------------ test data

def jpeg(img: Image.Image, long_edge: int = 640) -> bytes:
    img = img.convert("RGB")
    k = long_edge / max(img.size)
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def face_photo():
    """(label, jpeg) of an enrolment photo of someone in the gallery, or (None, None)."""
    root = settings.face_photo_dir
    for d in sorted(root.glob("*")) if root.exists() else []:
        if d.is_dir() and not d.name.startswith("_") and d.name != NAME:
            for p in sorted(d.glob("photo_*.png"))[:1]:
                return d.name, jpeg(Image.open(p))
    return None, None


def plant_photo():
    for p in sorted(settings.plant_dir.glob("map_*/plant_*/photo_1.png")) if settings.plant_dir.exists() else []:
        return jpeg(Image.open(p))
    return None


def recorded_frames(n: int):
    """The best stretch of n camera frames (with the robot's pose and head angles) from the newest debug
    session that has them: the one where the robot moved furthest, so monocular SLAM can initialise."""
    root = settings.debug_dir
    for sess in sorted((p for p in root.glob("2*") if p.is_dir()), reverse=True) if root.exists() else []:
        frames = []
        for js in sorted(sess.glob("[0-9]*.json")):
            try:
                d = json.loads(js.read_text())
            except ValueError:
                continue
            f, ex = d.get("frame") or {}, d.get("extras") or {}
            pose = f.get("pose") or ex.get("pose")
            img = sess / d["image"] if d.get("image") else None
            if not (img and img.exists() and pose and f.get("pan_deg") is not None and f.get("tilt_deg") is not None):
                continue
            real = "head_tilt_offset_deg" in (d.get("geometry") or {})
            frames.append(dict(t=f.get("t") or datetime.fromisoformat(d["ts"]).timestamp(), pose=pose,
                               pan=f["pan_deg"], tilt=f["tilt_deg"] + (0.0 if real else 21.0), img=img,
                               rows=(d.get("tof") or {}).get("rows")))
        if len(frames) < n:
            continue
        frames.sort(key=lambda f: f["t"])
        P = np.array([f["pose"][:2] for f in frames])
        step = np.r_[0, np.hypot(*np.diff(P, axis=0).T)].cumsum()
        i = int(np.argmax(step[n - 1:] - step[: len(frames) - n + 1]))
        return sess.name, frames[i:i + n], float(step[i + n - 1] - step[i])
    return None, [], 0.0


ROOM = (-1.5, 2.5, -1.5, 1.5)          # x0, x1, y0, y1: the synthetic room the SLAM tests scan


def room_rays(pose, step_deg: float = 2.0):
    """[[bearing_rad, range_m]] in the robot's frame from `pose`, to the walls of ROOM."""
    x, y, th = pose
    x0, x1, y0, y1 = ROOM
    out = []
    for b in np.radians(np.arange(0.0, 360.0, step_deg)):
        a = th + b
        c, s = math.cos(a), math.sin(a)
        ts = [t for t in (((x1 - x) / c if c > 1e-9 else None), ((x0 - x) / c if c < -1e-9 else None),
                          ((y1 - y) / s if s > 1e-9 else None), ((y0 - y) / s if s < -1e-9 else None)) if t]
        out.append([round(float(b), 4), round(min(ts), 3)])
    return out


def local_snapshot():
    """A RollingCostmap.snapshot (the Pi's) of open floor with a wall across the way at x = 1 m."""
    n, res, ox, oy = 80, 0.05, -2.0, -2.0
    cells = np.ones((n, n), np.uint8)                    # 1 seen free
    ix = int((1.0 - ox) / res)
    j0, j1 = int((-1.2 - oy) / res), int((0.8 - oy) / res)
    cells[ix:ix + 2, j0:j1] = 2                          # 2 occupied
    return {"ox": ox, "oy": oy, "res": res, "n": n, "cells": base64.b64encode(cells.tobytes()).decode()}


# ------------------------------------------------------------------ the tests

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('VS_PORT', '8080')}")
    ap.add_argument("--orb-frames", type=int, default=150, help="recorded frames to feed ORB-SLAM3")
    ap.add_argument("--keep", action="store_true", help="leave what the test added (selftest map, face, plant)")
    args = ap.parse_args()
    R = Run(args.url)
    ctx: dict = {"t0": time.time()}

    print(f"vision service self-test against {args.url}")
    try:
        R.c.get("/healthz", timeout=3)
    except httpx.HTTPError as e:
        print(f"  \033[31mFAIL\033[0m  the service does not answer ({e}): start it with `make start`")
        return 1
    label, face = face_photo()
    plant = plant_photo()
    scene = face or plant
    if scene is None:
        _, fr, _ = recorded_frames(1)
        scene = fr[0]["img"].read_bytes() if fr else jpeg(Image.fromarray(
            (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)))

    # ---- ops
    def health():
        h = R.ok(R.c.get("/healthz"))
        assert h["status"] == "ok", h
        ctx["vslam0"] = h.get("vslam") or {}
        return f"backend {h['backend']}, queue {h['queue_depth']}, up {h['uptime_s']:.0f} s, VLM {h.get('vlm') or 'none'}"
    R.check("healthz", health)

    def models():
        m = R.ok(R.c.get("/v1/models"))
        assert m["versions"], m
        return ", ".join(f"{k}={v}" for k, v in m["versions"].items())
    R.check("models", models)

    # ---- inference
    def detect():
        d = R.ok(R.c.post("/v1/detect", files={"image": ("f.jpg", scene, "image/jpeg")}, data={"seq": 7}))
        assert d["seq"] == 7 and isinstance(d["detections"], list), d
        classes = sorted({x["class"] for x in d["detections"]})
        if face and "person" not in classes:
            raise Warn(f"no person found in an enrolment photo of {label} ({classes})")
        return f"{len(d['detections'])} found {classes} in {d['inference_ms']:.0f} ms"
    R.check("detect", detect)

    def depth():
        w, h = Image.open(io.BytesIO(scene)).size
        d = R.ok(R.c.post("/v1/depth", files={"image": ("f.jpg", scene, "image/jpeg")},
                          data={"seq": 1, "points": json.dumps([[w // 2, h // 2], [w // 4, h // 2]])}))
        assert len(d["depths_m"]) == 2, d
        return f"{d['depths_m']} ({'metric' if d['metric'] else 'relative'}) in {d['inference_ms']:.0f} ms"
    R.check("depth", depth)

    def faces():
        if face is None:
            raise Skip("no enrolment photo in faces/ to test with")
        lst = R.ok(R.c.get("/v1/faces"))
        e = R.ok(R.c.post("/v1/faces/embed", files={"image": ("f.jpg", face, "image/jpeg")}, data={"seq": 1}))
        assert e["embedding"], e
        m = R.ok(R.c.post("/v1/faces/match", json={"embedding": e["embedding"]}))
        top = m["matches"][0] if m["matches"] else None
        assert top and top["label"] == label and top["similarity"] >= m["threshold"], \
            f"{label}'s own photo matched {top} (threshold {m['threshold']})"
        ctx["face_emb"] = e["embedding"]
        return f"{len(lst['labels'])} enrolled; {label}'s photo -> {top['label']} {top['similarity']:.2f} (quality {e['quality']:.2f})"
    R.check("faces: list, embed, match", faces)

    def face_enroll():
        if face is None:
            raise Skip("no enrolment photo in faces/")
        r = R.ok(R.c.post("/v1/faces/enroll", data={"label": NAME}, files=[("images", ("f.jpg", face, "image/jpeg"))]))
        assert r["embeddings_added"] >= 1, r
        d = R.ok(R.c.delete(f"/v1/faces/{NAME}"))
        assert d["embeddings_removed"] >= 1, d
        return f"enrolled and deleted '{NAME}' ({r['embeddings_added']} embedding)"
    R.check("faces: enroll, delete", face_enroll)

    # ---- follow
    def follow():
        img = face or scene
        R.ok(R.c.post("/v1/follow/start", json={"label": ""}))
        states = []
        try:
            for k in range(3):
                s = R.ok(R.c.post("/v1/follow/step", files={"image": ("f.jpg", img, "image/jpeg")},
                                  data={"seq": k + 1, "pose": json.dumps([0.0, 0.0, 0.0])}))
                assert "state" in s, s
                states.append(s["state"])
        finally:
            R.ok(R.c.post("/v1/follow/stop"))
        return f"states {states}"
    R.check("follow: start, step x3, stop", follow)

    # ---- nav (follow's global planner)
    def nav_plan():
        R.ok(R.c.post("/v1/nav/reset"))
        ctx["nav_dirty"] = True
        p = R.ok(R.c.post("/v1/nav/plan", json={"pose": [0.0, 0.0, 0.0], "goal": [1.7, 0.0], "target": [1.9, 0.0],
                                                "radius_m": 0.24, "local": local_snapshot()}))
        path = p["path"]
        assert len(path) >= 2, f"no path round the wall: {p}"
        # no leg may cross the wall (x 1.0-1.1 m, y -1.2-0.8 m in the snapshot)
        for a, b in zip(path, path[1:]):
            for k in np.linspace(0.0, 1.0, 40):
                x, y = a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])
                assert not (0.98 <= x <= 1.12 and -1.22 <= y <= 0.82), f"path crosses the wall: {path}"
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        st = R.ok(R.c.get("/v1/nav/state"))
        png = R.c.get("/v1/nav/map.png")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG", png.status_code
        R.ok(R.c.post("/v1/nav/reset"))                       # the test wall must never reach a real run's map
        ctx.pop("nav_dirty")
        return f"round the wall: {len(path)} waypoints, {length:.2f} m, {p['ms']} ms; {st['plans']} plans so far"
    R.check("nav: plan round a wall, state, map", nav_plan)

    # ---- ORB-SLAM3
    def orb_ready():
        v = ctx.get("vslam0") or {}
        if (v.get("reason") or "").startswith("disabled"):
            raise Skip("VS_VSLAM=0")
        assert v.get("backend") == "orbslam3" and v.get("ready"), \
            f"backend {v.get('backend')}, ready {v.get('ready')}: {v.get('reason')}"
        return f"orbslam3 ready, {v.get('frames_in', 0)} frames in, {v.get('frames_tracked', 0)} tracked so far"
    R.check("ORB-SLAM3: loaded", orb_ready)

    def orb_track():
        v0 = R.ok(R.c.get("/healthz"))["vslam"]
        if v0.get("backend") != "orbslam3" or not v0.get("ready"):
            raise Skip("ORB-SLAM3 not running")
        sess, frames, moved = recorded_frames(args.orb_frames)
        if not frames:
            raise Skip(f"no debug session with {args.orb_frames} camera frames to replay")
        tracked_any = False
        for f in frames:
            tof = {"rows": f["rows"], "age_s": 0.05} if f["rows"] else None
            r = R.ok(R.c.post("/v1/nav/frame", files={"image": ("f.jpg", f["img"].read_bytes(), "image/jpeg")},
                              data={"t": f["t"], "pose": json.dumps(f["pose"]), "pan_deg": f["pan"],
                                    "tilt_deg": f["tilt"], **({"tof": json.dumps(tof)} if tof else {})}))
            tracked_any |= bool(r.get("tracking"))
            if r.get("vslam") != "orbslam3":
                break
            time.sleep(0.1)                                   # about the Pi's rate; frames never queue anyway
        time.sleep(2.0)
        v = R.ok(R.c.get("/v1/nav/state"))["vslam"]
        R.ok(R.c.post("/v1/nav/reset"))                       # the recorded run's alignment must not shift the next real one
        assert v["backend"] == "orbslam3", f"ORB-SLAM3 stopped during the replay: {v.get('reason')}"
        gained = v["frames_tracked"] - v0.get("frames_tracked", 0)
        note = (f"{len(frames)} frames of {sess} ({moved:.1f} m driven): {gained} tracked, "
                f"tracking now {v['tracking']}, scale {v['scale_source'] or 'not fitted'}")
        if gained == 0 and not tracked_any:
            raise Warn("no crash, but it never initialised a map: " + note)
        return note
    R.check("ORB-SLAM3: track a recorded run", orb_track)

    # ---- explore SLAM, under its own map name
    def slam():
        r = R.ok(R.c.post("/v1/slam/reset", json={"name": NAME, "initial_pose": [0, 0, 0], "load": False}))
        assert r["name"] == NAME and r["keyframes"] == 0, r
        ctx["slam"] = True
        meta = {"odom": [0.0, 0.0, 0.0], "pan_deg": 0.0, "tilt_deg": 0.0, "settled": True, "pitch_deg": 0.0,
                "tof_rows": [[2.5] * 8 for _ in range(8)], "geom": {}}
        for pan in (-30.0, 0.0, 30.0):                       # 3-D scan frames for the first keyframe
            R.ok(R.c.post("/v1/recon/frame", files={"image": ("f.jpg", scene, "image/jpeg")},
                          data={"meta": json.dumps({**meta, "pan_deg": pan})}))
        k0 = R.ok(R.c.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": room_rays((0, 0, 0))}))
        k1 = R.ok(R.c.post("/v1/slam/keyframe", json={"odom": [0.5, 0.1, 0.2],
                                                     "rays": room_rays((0.5, 0.1, 0.2))}))
        assert k0["kf_id"] == 0 and k1["kf_id"] == 1, (k0, k1)
        st = R.ok(R.c.get("/v1/slam/state"))
        assert st["keyframes"] == 2, st
        walls = R.ok(R.c.get("/v1/slam/walls"))
        return (f"2 keyframes, {k1.get('explored_m2', 0):.1f} m² explored, match {k1['match']['score']:.2f}"
                f" ({'ok' if k1['match']['accepted'] else 'refused'}), {len(walls.get('walls', []))} walls")
    R.check("SLAM: reset, recon frames, keyframes, walls", slam)

    def slam_plan():
        if not ctx.get("slam"):
            raise Skip("SLAM reset failed")
        p = R.ok(R.c.post("/v1/slam/plan", json={"start": [0.0, 0.0], "goal": [1.5, 0.8]}))
        assert len(p["path"]) >= 2, p
        v = R.ok(R.c.post("/v1/slam/viewpoints", json={"start": [0.0, 0.0], "target": [1.2, -0.4], "radius": 0.7, "n": 8}))
        assert v["viewpoints"], f"no viewpoint round a plant in open floor: {v}"
        png = R.c.get("/v1/slam/map.png")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG", png.status_code
        s = R.ok(R.c.post("/v1/slam/save"))
        assert Path(s["saved"]).name == f"{NAME}.npz", s
        return f"path {len(p['path'])} waypoints, {len(v['viewpoints'])}/8 viewpoints, map saved"
    R.check("SLAM: plan, viewpoints, map, save", slam_plan)

    def recon_build():
        if not ctx.get("slam"):
            raise Skip("SLAM reset failed")
        r = R.c.post("/v1/recon/build", json={"name": NAME})
        assert r.status_code in (200, 409), f"HTTP {r.status_code}: {r.text[:200]}"
        deadline, job = time.monotonic() + 180, {}
        while time.monotonic() < deadline:
            job = R.ok(R.c.get("/v1/recon/status"))
            if job.get("state") in ("done", "failed"):
                break
            time.sleep(1.0)
        if job.get("state") == "failed":
            raise Warn(f"build ran but failed on the synthetic scan: {job.get('error')}")
        assert job.get("state") == "done", f"not done after 180 s: {job}"
        if not (job.get("report") or {}).get("faces"):
            raise Warn(f"built from {job.get('done')} frames, but the test scan has no surface to mesh")
        mesh = R.c.get("/v1/recon/mesh.ply")
        assert mesh.status_code == 200 and mesh.content[:3] == b"ply", mesh.status_code
        return f"{job.get('done')} frames fused, mesh {len(mesh.content) // 1024} KB"
    R.check("recon: build, status, mesh", recon_build)

    def plants():
        img = plant or scene
        e = R.ok(R.c.post("/v1/plants/embed", files={"image": ("p.jpg", img, "image/jpeg")}, data={"seq": 1}))
        assert e["embedding"], e
        m = R.ok(R.c.post("/v1/plants/match", json={"embedding": e["embedding"], "position": [1.2, -0.4]}))
        r = R.ok(R.c.post("/v1/plants/enroll", files=[("images", ("p.jpg", img, "image/jpeg"))],
                          data={"label": "", "meta": json.dumps({"position": [1.2, -0.4], "views": [{"pose": [0.5, -0.4, 0]}]})}))
        lab = r["label"]
        ctx["plant"] = lab
        assert lab in json.dumps(R.ok(R.c.get("/v1/plants"))["plants"]), f"{lab} enrolled but not listed"
        info = R.ok(R.c.get(f"/v1/plants/{lab}"))
        photo = None
        for _ in range(20):                                   # the PNG is written by a background task
            photo = R.c.get(f"/v1/plants/{lab}/photo_1.png")
            if photo.status_code == 200:
                break
            time.sleep(0.25)
        assert photo is not None and photo.status_code == 200, f"photo_1.png: HTTP {photo.status_code}"
        R.ok(R.c.delete(f"/v1/plants/{lab}"))
        ctx.pop("plant")
        return (f"embedding {len(e['embedding'])}, {len(m['matches'])} matches, enrolled {lab} "
                f"({info.get('n_views')} view), photo served, deleted")
    R.check("plants: embed, match, enroll, get, delete", plants)

    # ---- debug uploads
    def debug():
        sess = f"{NAME}-{datetime.now():%Y%m%d-%H%M%S}"
        ctx["debug"] = sess
        latest = settings.debug_dir / "latest.json"           # what Monitoring-Service shows: put back afterwards
        ctx["debug_latest"] = latest.read_text() if latest.exists() else None
        meta = {"session": sess, "n": 1, "ts": datetime.now().astimezone().isoformat(timespec="milliseconds")}
        R.ok(R.c.post("/v1/debug/frame", data={"meta": json.dumps(meta)}, files={"image": ("f.jpg", scene, "image/jpeg")}))
        latest = R.ok(R.c.get("/v1/debug/latest"))
        assert latest.get("session") == sess, latest
        return f"stored and read back session {sess}"
    R.check("debug: frame, latest", debug)

    # ---- image + text model
    def ask():
        r = R.c.post("/v1/ask", data={"question": "In one short sentence, what is in this photo?"},
                     files=[("images", ("f.jpg", scene, "image/jpeg"))], timeout=90)
        if r.status_code == 503:
            raise Skip(r.json().get("detail", "no VLM answered")[:120])
        a = R.ok(r)
        assert a.get("answer"), a
        return f"{a.get('provider')}: {a['answer'][:80]}"
    R.check("ask (VLM)", ask)

    # ---- at the end: still healthy, ORB-SLAM3 still alive
    def final():
        h = R.ok(R.c.get("/healthz"))
        v = h.get("vslam") or {}
        if (v.get("reason") or "").startswith("disabled"):
            return "healthy (visual SLAM disabled)"
        assert v.get("backend") == "orbslam3" and v.get("ready"), f"ORB-SLAM3 is no longer running: {v.get('reason')}"
        return f"healthy, ORB-SLAM3 still running ({v['frames_in']} frames in, {v['frames_tracked']} tracked)"
    R.check("still healthy at the end", final)

    if not args.keep:
        cleanup(R, ctx)
    fails = [r for r in R.rows if r[0] == "FAIL"]
    counts = {s: sum(1 for r in R.rows if r[0] == s) for s in ("PASS", "WARN", "SKIP", "FAIL")}
    print("\n" + "  ".join(f"{k} {v}" for k, v in counts.items()) +
          f"   ({sum(r[2] for r in R.rows):.0f} s)" + ("" if args.keep else "; test data removed"))
    return 1 if fails else 0


def cleanup(R: Run, ctx: dict) -> None:
    """Remove what the test added: the leftovers the API does not delete itself."""
    if ctx.get("nav_dirty"):                                  # the nav check failed half way: its wall is still in the map
        R.c.post("/v1/nav/reset")
    if ctx.get("plant"):
        R.c.delete(f"/v1/plants/{ctx['plant']}")
    R.c.delete(f"/v1/faces/{NAME}")                          # 404 if already gone: fine
    time.sleep(0.5)                                           # background photo writes
    for p in (settings.face_photo_dir / NAME, *settings.slam_dir.glob(f"{NAME}.*"),
              *(settings.slam_dir / "recon").glob(f"{NAME}-*"), *(settings.slam_dir / "recon").glob(f"{NAME}.*"),
              *settings.debug_dir.glob(f"{NAME}-*")):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    # a deleted plant's photos are moved to plants/_deleted/, not destroyed: the test plant's go for good
    for p in settings.plant_dir.glob("_deleted/plant_*"):
        if p.stat().st_mtime >= ctx["t0"] - 1:
            shutil.rmtree(p, ignore_errors=True)
    # the selftest map's plant index (the SLAM reset registered it): gone, so real maps keep their numbering
    idx_file = settings.plant_dir / "maps.json"
    try:
        known = json.loads(idx_file.read_text())
        idx = known.pop(NAME, None)
        if idx is not None:
            idx_file.write_text(json.dumps(known, indent=2))
            folder = settings.plant_dir / f"map_{idx}"
            if folder.is_dir() and not any(p.is_file() for p in folder.rglob("*")):
                shutil.rmtree(folder, ignore_errors=True)
    except (OSError, ValueError):
        pass
    if "debug_latest" in ctx:
        latest = settings.debug_dir / "latest.json"
        if ctx["debug_latest"] is None:
            latest.unlink(missing_ok=True)
        else:
            latest.write_text(ctx["debug_latest"])


if __name__ == "__main__":
    sys.exit(main())
