"""The plant register: identity embeddings plus a folder of photos per plant.

    plants/                         (VS_PLANT_DIR; default: next to the code)
        gallery.npz                 DINOv2 embeddings, one row per view
        maps.json                   SLAM map name -> map index
        map_1/
            plant_001/
                meta.json           position on the map, every visit, every view
                photo_1.png  bbox_1.txt        (written in the background, see photos.py)
                ...

Same shape as the face gallery on purpose — enroll = add views under a label,
match = best cosine similarity over that label's views. Several views per plant
is what makes a *rotated* plant still match: the side the robot sees today was
probably photographed from some angle last time.

Re-visiting a known plant appends a new visit (with its photos) rather than a
new plant, which is what growth tracking reads later.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import photos
from .gallery import Gallery

LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BadLabel(ValueError):
    pass


def combine_positions(positions: list) -> tuple:
    """One plant position from every visit's ([x, y] or [x, y, height]): the median of
    each coordinate (one bad ranging cannot drag it), and how far the visits' x, y spread
    from it (m). A plant measured once has spread 0."""
    import statistics
    n = min(len(p) for p in positions)
    mid = [round(statistics.median(p[i] for p in positions), 3) for i in range(n)]
    spread = max(((p[0] - mid[0]) ** 2 + (p[1] - mid[1]) ** 2) ** 0.5 for p in positions)
    return mid, round(spread, 3)


def foliage_health(rgb: np.ndarray, bbox=None) -> dict:
    """Fast, zero-LLM foliage health index from an RGB plant photo (or bbox crop).
    Computes healthy-green vs yellow/brown-stressed pixel shares using Excess-Green
    and HSV hue bands, returning {score (0..100 or None), green_pct, yellow_brown_pct}."""
    h, w = rgb.shape[:2]
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (max(0, int(round(bbox[0]))), max(0, int(round(bbox[1]))),
                          min(w, int(round(bbox[2]))), min(h, int(round(bbox[3]))))
        crop = rgb[y1:y2, x1:x2] if x2 > x1 + 4 and y2 > y1 + 4 else rgb
    else:
        my, mx = h // 5, w // 5
        crop = rgb[my:h - my, mx:w - mx] if h >= 10 and w >= 10 else rgb
    f = crop.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx_c, mn_c = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    sat = np.where(mx_c > 1e-4, (mx_c - mn_c) / np.maximum(mx_c, 1e-4), 0.0)
    val = mx_c
    # Healthy foliage: green dominant (Excess-Green > 0.06) with real saturation and brightness
    exg = 2.0 * g - r - b
    green = (exg > 0.06) & (g > r * 1.02) & (sat > 0.18) & (val > 0.12)
    # Stressed / chlorotic / dry foliage: warm yellow-brown tones (red >= green > blue, saturated)
    yel = (~green) & (r >= g * 0.95) & (g > b * 1.15) & (r > b * 1.35) & (sat > 0.22) & (val > 0.15) & (val < 0.92)
    total = max(1, int(crop.shape[0] * crop.shape[1]))
    g_cnt, y_cnt = int(green.sum()), int(yel.sum())
    g_pct, y_pct = round(100.0 * g_cnt / total, 1), round(100.0 * y_cnt / total, 1)
    foliage = g_cnt + y_cnt
    score = round(100.0 * g_cnt / foliage, 1) if (foliage / total) >= 0.02 else None
    return {"score": score, "green_pct": g_pct, "yellow_brown_pct": y_pct}


def check_label(label: str) -> str:
    # Labels become directory names: refuse anything that could escape the root.
    if not LABEL_RE.match(label or ""):
        raise BadLabel(f"label {label!r} must match {LABEL_RE.pattern}")
    return label


class PlantStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.gallery = Gallery(root / "gallery.npz", noun="plants")
        self._lock = threading.Lock()
        self.map_index = self._latest_map() or 1

    def _latest_map(self) -> int:
        return max((int(m.group(1)) for p in self.root.iterdir() if (m := re.match(r"^map_(\d+)$", p.name))),
                   default=0)

    def use_map(self, name: str, new: bool) -> int:
        """Point the store at a SLAM map. A fresh map gets the next index; a
        reloaded one gets the index it had (or the latest, if never recorded)."""
        f = self.root / "maps.json"
        with self._lock:
            known = json.loads(f.read_text()) if f.exists() else {}
            if new or name not in known:
                known[name] = (max([self._latest_map(), *known.values()]) + 1) if new else (self._latest_map() or 1)
                f.write_text(json.dumps(known, indent=2))
            self.map_index = known[name]
        return self.map_index

    def folder(self, label: str) -> Path:
        return self.root / f"map_{self.map_index}" / check_label(label)

    def next_label(self) -> str:
        with self._lock:
            nums = [int(m.group(1)) for p in self.root.glob("map_*/plant_*")
                    if (m := re.match(r"^plant_(\d+)$", p.name))]
            nums += [int(m.group(1)) for l in self.gallery.labels()
                     if (m := re.match(r"^plant_(\d+)$", l))]
            return f"plant_{(max(nums) + 1) if nums else 1:03d}"

    def meta(self, label: str) -> Optional[dict]:
        f = self.folder(label) / "meta.json"
        return json.loads(f.read_text()) if f.exists() else None

    def add_visit(self, label: str, jpegs: list[bytes], embeddings: list[np.ndarray],
                  meta: dict) -> dict:
        """Store one visit: embeddings in the gallery, the visit record (position,
        per-view poses, anything else the robot sent) in meta.json, and the photo
        file names reserved. The photos themselves are written afterwards by a
        background task (photos.write) — the caller gets `files` back for that."""
        folder = self.folder(label)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            first = (self.meta(label) or {}).get("n_views", 0) + 1
            files = [f"photo_{first + i}.png" for i in range(len(jpegs))]
            m = self.meta(label) or {"label": label, "created": stamp, "visits": []}
            views = meta.get("views") or [{} for _ in files]
            m["visits"].append({
                "t": stamp,
                "position": meta.get("position"),
                "views": [{"file": f, **(views[i] if i < len(views) else {})}
                          for i, f in enumerate(files)],
                **{k: v for k, v in meta.items() if k not in ("views", "position")},
            })
            if meta.get("position") is not None:
                m["position"], m["position_spread_m"] = combine_positions(
                    [v["position"] for v in m["visits"] if v.get("position")])
            m["updated"] = stamp
            m["n_views"] = sum(len(v["views"]) for v in m["visits"])
            tmp = folder / "meta.json.tmp"
            tmp.write_text(json.dumps(m, indent=2))
            tmp.replace(folder / "meta.json")
        added = self.gallery.add(label, embeddings)
        return {"label": label, "folder": str(folder), "files": files, "photos_saved": len(files),
                "embeddings_added": added, "views_total": m["n_views"], "visits": len(m["visits"])}

    def list(self) -> list[dict]:
        out = []
        for p in sorted((self.root / f"map_{self.map_index}").glob("*")):
            if p.is_dir() and LABEL_RE.match(p.name) and (p / "meta.json").exists():
                m = json.loads((p / "meta.json").read_text())
                out.append({k: m.get(k) for k in ("label", "position", "created", "updated", "n_views")}
                           | {"visits": len(m.get("visits", []))})
        return out

    def positions(self) -> list[tuple]:
        """(label, x, y, height) of every plant: x, y on the map, height of the pot's
        base above the floor (None if it was never measured)."""
        return [(m["label"], *m["position"][:2], m["position"][2] if len(m["position"]) > 2 else None)
                for m in self.list() if m.get("position")]

    def remove(self, label: str) -> int:
        """Forget a plant. Photos are moved to _deleted/, never destroyed."""
        check_label(label)
        removed = self.gallery.remove(label)
        folder = self.folder(label)
        if folder.exists():
            dest = self.root / "_deleted" / f"{label}_{time.strftime('%Y%m%dT%H%M%S')}"
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(folder), str(dest))
            removed = removed or 1
        return removed
