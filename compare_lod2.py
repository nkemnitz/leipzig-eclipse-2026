"""What is actually blocking the sun, and would a LoD2 building model contain it?

Leipzig's official 3D model is LoD2: per-building solids with roof shapes, plus a
separate 3D tree layer. Ours is DOM1, a 1 m laser-scanned surface that contains
whatever physically stood there. The practical question is not which is prettier,
it is whether the specific object that blocks each viewpoint is a *building*.

For each spot we march to the sun, find the cell that produces the horizon maximum,
and classify it against OSM building footprints and woodland polygons.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

from viewshed import EYE_H, R_EFF, grid_convergence_deg

DATA = Path("data")
to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
AZ_MAX, ALT_MAX = 289.75, 3.50


def rasterize_local(polys, shape, minx, miny, res, ox, oy):
    ny, nx = shape
    img = Image.new("1", (nx, ny), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        pts = [((x + ox - minx) / res, (y + oy - miny) / res) for x, y in p]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def main():
    g = json.loads((DATA / "grid_meta.json").read_text())
    res, minx, miny = g["res"], g["minx"], g["miny"]
    dsm = np.load(DATA / "dsm2m.npy", mmap_mode="r")
    dtm = np.load(DATA / "dtm2m.npy", mmap_mode="r")
    ny, nx = dsm.shape

    om = json.loads((DATA / "meta.json").read_text())
    ox, oy = om["origin"]
    print("rasterising OSM buildings / woods over the full grid ...")
    b = json.loads((DATA / "buildings.json").read_text())
    w = json.loads((DATA / "woods.json").read_text())
    build = rasterize_local([x["r"] for x in b], (ny, nx), minx, miny, res, ox, oy)
    wood = rasterize_local([x["r"] for x in w], (ny, nx), minx, miny, res, ox, oy)
    print(f"  building cells {build.sum()/1e6:.1f} M, wood cells {wood.sum()/1e6:.1f} M")

    conv = grid_convergence_deg((g["out_minx"] + g["out_maxx"]) / 2,
                                (g["out_miny"] + g["out_maxy"]) / 2)
    a = np.radians(AZ_MAX + conv)
    ue, un = np.sin(a), np.cos(a)
    d = np.concatenate([np.arange(res, 200, res), np.arange(200, 1000, res * 2),
                        np.arange(1000, 6000, res * 6)])
    drop = d * d / (2 * R_EFF)

    ranked = json.loads((DATA / "ranked_spots.json").read_text())
    spots = [r for r in ranked["landmarks"] if "utm_x" in r][:14]

    print(f"\n{'spot':26s} {'skyline':>8s} {'blocker':>9s} {'at':>7s} {'obj h':>7s}  what")
    tally = {}
    for r in spots:
        x, y = r["utm_x"], r["utm_y"]
        i0, j0 = (y - miny) / res, (x - minx) / res
        eye = float(dtm[int(i0), int(j0)]) + EYE_H
        ii = np.rint(i0 + d * un / res).astype(np.int64)
        jj = np.rint(j0 + d * ue / res).astype(np.int64)
        ok = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        h = dsm[ii[ok], jj[ok]]
        ang = np.degrees(np.arctan((h - drop[ok] - eye) / d[ok]))
        k = int(ang.argmax())
        bi, bj, bd = ii[ok][k], jj[ok][k], d[ok][k]
        objh = float(dsm[bi, bj] - dtm[bi, bj])
        if build[bi, bj]:
            what = "BUILDING (LoD2 has it)"
        elif wood[bi, bj] or objh > 4:
            what = "vegetation / unmodelled (LoD2 lacks it)"
        else:
            what = "bare terrain"
        tally[what] = tally.get(what, 0) + 1
        print(f"{r['landmark'][:26]:26s} {ang[k]:+8.2f} {'':>9s} {bd:6.0f}m "
              f"{objh:6.1f}m  {what}")

    print("\nblocker classification:")
    for k2, v in sorted(tally.items(), key=lambda t: -t[1]):
        print(f"  {v:2d}/{len(spots)}  {k2}")

    # DOM1 currency, straight from the per-tile description files
    print("\nDOM1 acquisition dates (from the tiles' own _akt.csv):")
    dates = set()
    for z in sorted((DATA / "geosn/dom1").glob("*.zip"))[:12]:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".csv"):
                    txt = zf.read(n).decode("latin-1", errors="replace")
                    dates.add(txt.strip().splitlines()[-1][:120])
    for t in sorted(dates)[:8]:
        print("  ", t)


if __name__ == "__main__":
    main()
