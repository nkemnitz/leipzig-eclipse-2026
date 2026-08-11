"""Turn Leipzig's LoD2 CityGML into tiled binary geometry for the viewer.

These buildings are for ORIENTATION ONLY. The sun/shadow answer keeps coming from
the 2023 DOM1 laser surface, which also contains the tree canopy that actually does
most of the blocking; LoD2 is a 2021 building-only model and would under-block badly
if it drove the physics.

Output per 1 km tile: two triangle soups (roof, wall) of int16-quantised vertices.
The de-quantisation is a per-axis scale+offset, applied as the mesh's own
scale/position in three.js, so the raw int16 buffer goes straight to the GPU.
Normals are derived in the fragment shader (dFdx/dFdy), so none are stored --
that halves the payload and suits LoD2's faceted look.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from mapbox_earcut import triangulate_float32

SRC = Path("data/lod2/leipzig_lod2_citygml.zip")
OUT = Path("viewer/data/lod2")

TILE = 1000.0          # metres
XY_SCALE = TILE / 32000.0     # ~0.031 m
Z0, Z_SCALE = 60.0, 0.01      # 1 cm, covers 60..715 m

# Study area (must match grid_meta out_*)
MINX, MAXX = 308000, 324000
MINY, MAXY = 5678000, 5696000

RE_SURF = re.compile(r"<bldg:(Wall|Roof|Ground)Surface\b")
RE_POS = re.compile(r"<gml:(exterior|interior)>\s*<gml:LinearRing[^>]*>\s*"
                    r"<gml:posList[^>]*>([^<]+)</gml:posList>")


def tri_from_polygon(rings):
    """rings: list of (kind, Nx3 array). Returns (M,3,3) triangles."""
    ext = [r for k, r in rings if k == "exterior"]
    if not ext:
        return None
    holes = [r for k, r in rings if k == "interior"]
    verts = [ext[0]] + holes
    counts = []
    acc = []
    for v in verts:
        if len(v) > 1 and np.allclose(v[0], v[-1]):
            v = v[:-1]
        if len(v) < 3:
            continue
        acc.append(v)
        counts.append(len(v))
    if not acc:
        return None
    pts = np.concatenate(acc, axis=0)

    # Newell normal -> project onto the plane by dropping its dominant axis
    p = acc[0]
    n = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ax = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != ax]
    flat = np.ascontiguousarray(pts[:, keep], dtype=np.float32)

    rings_idx = np.cumsum(counts).astype(np.uint32)
    try:
        idx = triangulate_float32(flat, rings_idx)
    except Exception:
        return None
    if len(idx) < 3:
        return None
    return pts[np.asarray(idx, dtype=np.int64)].reshape(-1, 3, 3)


def parse_one(name):
    with zipfile.ZipFile(SRC) as z:
        raw = z.read(name).decode("utf-8", errors="replace")

    out = {"roof": {}, "wall": {}}
    n_bld = 0
    for chunk in raw.split("<bldg:boundedBy>")[1:]:
        m = RE_SURF.search(chunk)
        if not m:
            continue
        kind = m.group(1)
        if kind == "Ground":       # never visible; ~10% of the triangles
            continue
        target = "roof" if kind == "Roof" else "wall"
        rings = []
        for which, txt in RE_POS.findall(chunk):
            a = np.fromstring(txt, sep=" ")
            if a.size < 9 or a.size % 3:
                continue
            rings.append((which, a.reshape(-1, 3)))
            if which == "exterior" and len(rings) > 1:
                # a new exterior starts a new polygon
                pass
        if not rings:
            continue
        # group into polygons: each 'exterior' opens one
        polys, cur = [], []
        for which, arr in rings:
            if which == "exterior":
                if cur:
                    polys.append(cur)
                cur = [(which, arr)]
            elif cur:
                cur.append((which, arr))
        if cur:
            polys.append(cur)

        for poly in polys:
            tris = tri_from_polygon(poly)
            if tris is None:
                continue
            if target == "wall":
                # Extend walls a few metres below their own base. The terrain mesh
                # is a decimated DGM1 and can sit above or below true ground by a
                # metre or two (doubled by vertical exaggeration); without a skirt
                # that shows up as buildings sunk into the hill with only polygonal
                # caps visible, or floating with a gap underneath.
                zmin = tris[:, :, 2].min()
                if tris[:, :, 2].max() - zmin > 3.0:
                    tris[:, :, 2] = np.where(tris[:, :, 2] <= zmin + 0.5,
                                             tris[:, :, 2] - 4.0, tris[:, :, 2])
            n_bld += 1
            c = tris.reshape(-1, 3).mean(axis=0)
            if not (MINX <= c[0] < MAXX and MINY <= c[1] < MAXY):
                continue
            tx = int((c[0] - MINX) // TILE)
            ty = int((c[1] - MINY) // TILE)
            out[target].setdefault((tx, ty), []).append(tris)

    return {k: {t: np.concatenate(v, axis=0) for t, v in d.items()}
            for k, d in out.items()}


def quantise(tris, tx, ty):
    """(easting, northing, height) -> int16 (x, HEIGHT, northing).

    Component order is swapped here, not in the viewer: three.js reads a position
    attribute as (x, y-up, z), so height must be component 1. The viewer then
    de-quantises purely with the mesh's own scale/position (with a negative Z
    scale to turn northing into world -Z), leaving the int16 buffer untouched.
    """
    ox, oy = MINX + tx * TILE, MINY + ty * TILE
    q = np.empty(tris.shape, dtype=np.int16)
    q[:, :, 0] = np.clip((tris[:, :, 0] - ox) / XY_SCALE, -32768, 32767)
    q[:, :, 1] = np.clip((tris[:, :, 2] - Z0) / Z_SCALE, -32768, 32767)
    q[:, :, 2] = np.clip((tris[:, :, 1] - oy) / XY_SCALE, -32768, 32767)
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SRC) as z:
        names = [n for n in z.namelist() if n.endswith(".gml")]
    print(f"parsing {len(names)} district files ...")

    merged = {"roof": {}, "wall": {}}
    done = 0
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(parse_one, names):
            done += 1
            for cls in ("roof", "wall"):
                for t, arr in res[cls].items():
                    merged[cls].setdefault(t, []).append(arr)
            if done % 10 == 0:
                print(f"  {done}/{len(names)}")

    manifest = {"tile": TILE, "minx": MINX, "miny": MINY,
                "xy_scale": XY_SCALE, "z0": Z0, "z_scale": Z_SCALE, "tiles": {}}
    total_tris = 0
    total_bytes = 0
    tiles = sorted(set(merged["roof"]) | set(merged["wall"]))
    for (tx, ty) in tiles:
        entry = {}
        for cls in ("roof", "wall"):
            parts = merged[cls].get((tx, ty))
            if not parts:
                continue
            tris = np.concatenate(parts, axis=0)
            q = quantise(tris, tx, ty)
            fn = f"{tx}_{ty}_{cls}.bin"
            (OUT / fn).write_bytes(q.tobytes())
            entry[cls] = {"file": fn, "tris": int(len(tris))}
            total_tris += len(tris)
            total_bytes += q.nbytes
        if entry:
            manifest["tiles"][f"{tx}_{ty}"] = entry

    (OUT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"\n{len(manifest['tiles'])} tiles, {total_tris/1e6:.2f} M triangles, "
          f"{total_bytes/1e6:.1f} MB")
    if manifest["tiles"]:
        per = total_bytes / len(manifest["tiles"]) / 1e3
        print(f"average {per:.0f} kB per 1 km tile")


if __name__ == "__main__":
    main()
