"""Raster orientation checks.

Three different conventions meet in this project and every one of them has already
been wrong once:
  * analysis rasters and baked textures  -> row 0 = SOUTH
  * WMS GetMap responses                 -> row 0 = NORTH
  * three.js textures                    -> flipY decides which of the two you get
A mirrored tile is not an obvious failure -- it renders as plausible imagery in the
wrong place (a street across a hilltop) -- so it needs a test, not an eyeball.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA = Path("data")
WMS = "https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest"
FAIL = []


def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def baked_crop(x0, y0, t, n=256):
    g = json.loads((DATA / "grid_meta.json").read_text())
    minx, miny = g["out_minx"], g["out_miny"]
    maxx, maxy = g["out_maxx"], g["out_maxy"]
    im = Image.open("viewer/data/ortho.jpg").convert("L")
    w, h = im.size
    box = (int((x0 - minx) / (maxx - minx) * w), int((y0 - miny) / (maxy - miny) * h),
           int((x0 + t - minx) / (maxx - minx) * w), int((y0 + t - miny) / (maxy - miny) * h))
    return np.asarray(im.crop(box).resize((n, n)), dtype=np.float32)


def wms_tile(x0, y0, t, n=256):
    url = (f"{WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=sn_dop_020&STYLES="
           f"&CRS=EPSG:25833&BBOX={x0},{y0},{x0+t},{y0+t}"
           f"&WIDTH={n}&HEIGHT={n}&FORMAT=image/jpeg")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("L"), dtype=np.float32)


def check(name, got, want, tol=0.0):
    ok = got >= want - tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {got:+.3f}")
    if not ok:
        FAIL.append(name)


def main():
    sites = [("Fockeberg", 316000, 5688000), ("centre", 316500, 5690500),
             ("Auensee", 313000, 5693000)]
    print("1. WMS tiles are NORTH-up; the baked mosaic is SOUTH-up")
    for name, x0, y0 in sites:
        b = baked_crop(x0, y0, 500)
        w = wms_tile(x0, y0, 500)
        asis, flip = corr(b, w), corr(b, w[::-1])
        print(f"  {name:12s} as-is {asis:+.3f}   flipped {flip:+.3f}")
        if flip <= asis:
            FAIL.append(f"{name}: WMS orientation")
        check(f"{name}: flipped matches the baked mosaic", flip, 0.7)

    print("\n2. viewer/app.js must set flipY=true on the streamed WMS texture")
    src = Path("viewer/app.js").read_text(encoding="utf-8")
    i = src.find("loader.setCrossOrigin")
    seg = src[i:i + 600]
    ok = "tex.flipY = true" in seg
    print(f"  {'PASS' if ok else 'FAIL'}  streamed texture uses flipY = true")
    if not ok:
        FAIL.append("app.js flipY")

    print("\n3. Baked detail height tiles are SOUTH-up (row 0 = miny)")
    man = json.loads(Path("viewer/data/detail/manifest.json").read_text())
    gx = int((316000 - man["minx"]) / man["tile_m"])
    gy = int((5688000 - man["miny"]) / man["tile_m"])
    a = np.asarray(Image.open(f"viewer/data/detail/{gx}_{gy}.png").convert("RGB"),
                   dtype=np.float32)
    hh = ((a[:, :, 0].astype(np.uint32) * 256 + a[:, :, 1]) / man["scale"] + man["h0"])
    g = json.loads((DATA / "grid_meta.json").read_text())
    dtm = np.load(DATA / "dsm2m.npy", mmap_mode="r")
    i0 = int((5688000 - g["miny"]) / g["res"])
    j0 = int((316000 - g["minx"]) / g["res"])
    ref = np.asarray(dtm[i0:i0 + 250, j0:j0 + 250], dtype=np.float32)
    small = hh[::2, ::2][:250, :250]
    asis, flip = corr(ref, small), corr(ref, small[::-1])
    print(f"  as-is {asis:+.3f}   flipped {flip:+.3f}")
    check("detail height tile is south-up", asis, 0.9)
    if flip > asis:
        FAIL.append("detail tile orientation")

    print()
    print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
