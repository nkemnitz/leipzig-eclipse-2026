"""Export a window of the voxel grid as a PLY that MeshLab (or CloudCompare) opens.

occ.npy is 90 M cells x 32 bits -- far past what a mesh viewer will take, and it
is a bit-packed array, not geometry. This slices a box out of it and writes real
points: one per occupied voxel, plus the bare-earth surface underneath so the
scene has a floor to read the trees against.

Coordinates are LOCAL METRES relative to the box's south-west corner, because
float32 cannot hold a UTM northing of 5,688,000 to better than half a metre --
writing absolute coordinates would quantise the whole cloud onto a 0.5 m lattice.
The absolute origin goes in the PLY header as a comment.

    python export_voxels.py --place fockeberg --size 1000
    python export_voxels.py --x 316500 --y 5688600 --size 1500 --cubes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VOX = Path("data/voxels")

PLACES = {
    "fockeberg":     (316430, 5688540),
    "clara-zetkin":  (313900, 5687900),
    "markt":         (314500, 5691500),
    "auensee":       (310900, 5695400),
    "bistumshoehe":  (312500, 5680500),
    "panorama":      (314270, 5691060),
    "voelkerschlacht": (318700, 5687600),
}

VERT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                 ("r", "u1"), ("g", "u1"), ("b", "u1")])

# 8 corners and 12 triangles of a unit cube, for --cubes
CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                   [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                   [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)


def canopy_colour(h):
    """Height above ground -> colour. Dark at the trunk, bright at the crown top."""
    t = np.clip(h / 30.0, 0.0, 1.0)[:, None]
    lo = np.array([46, 74, 40], dtype=np.float32)      # shaded understorey
    hi = np.array([150, 200, 96], dtype=np.float32)    # sunlit crown
    return (lo + (hi - lo) * t).astype(np.uint8)


def write_ply(path, verts, faces, comments):
    hdr = ["ply", "format binary_little_endian 1.0"]
    hdr += [f"comment {c}" for c in comments]
    hdr += [f"element vertex {len(verts)}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        hdr += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    hdr += ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(hdr).encode("ascii"))
        f.write(verts.tobytes())
        if faces is not None:
            rec = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", 3)])
            rec["n"] = 3
            rec["v"] = faces
            f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", choices=sorted(PLACES))
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--size", type=float, default=1000.0, help="box edge in metres")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cubes", action="store_true",
                    help="write real cubes instead of points (12x the file)")
    ap.add_argument("--no-ground", action="store_true")
    a = ap.parse_args()

    if a.place:
        cx, cy = PLACES[a.place]
    elif a.x is not None and a.y is not None:
        cx, cy = a.x, a.y
    else:
        ap.error("give --place or both --x and --y")
    name = a.place or f"{int(cx)}_{int(cy)}"
    out = Path(a.out or f"out/voxels_{name}_{int(a.size)}m.ply")
    out.parent.mkdir(parents=True, exist_ok=True)

    m = json.loads((VOX / "meta.json").read_text())
    res, vh, nb = m["res"], m["vh"], m["bands"]
    j0 = int((cx - a.size / 2 - m["minx"]) / res)
    i0 = int((cy - a.size / 2 - m["miny"]) / res)
    n = int(a.size / res)
    j0 = max(0, min(j0, m["nx"] - n))
    i0 = max(0, min(i0, m["ny"] - n))
    x0 = m["minx"] + j0 * res
    y0 = m["miny"] + i0 * res

    occ = np.load(VOX / "occ.npy", mmap_mode="r")[i0:i0 + n, j0:j0 + n]
    occ = np.ascontiguousarray(occ)
    sp = VOX / "solid.npy"
    solid = (np.ascontiguousarray(np.load(sp, mmap_mode="r")[i0:i0 + n, j0:j0 + n])
             if sp.exists() else np.zeros((n, n), bool))

    # bare earth, from the same DGM the voxel heights were measured against
    gm = json.loads(Path("data/grid_meta.json").read_text())
    gj = int((x0 - gm["minx"]) / gm["res"])
    gi = int((y0 - gm["miny"]) / gm["res"])
    dtm = np.ascontiguousarray(
        np.load("data/dtm2m.npy", mmap_mode="r")[gi:gi + n, gj:gj + n]).astype(np.float32)
    dtm = np.nan_to_num(dtm, nan=float(np.nanmedian(dtm)))

    # unpack the bit planes into (cell, band) pairs
    bit = ((occ[:, :, None] >> np.arange(nb, dtype=np.uint32)[None, None, :]) & 1).astype(bool)
    ii, jj, bb = np.nonzero(bit)
    zag = (bb + 0.5) * vh
    px = (jj + 0.5) * res
    py = (ii + 0.5) * res
    pz = dtm[ii, jj] + zag

    col = canopy_colour(zag)
    is_solid = solid[ii, jj]
    col[is_solid] = np.array([196, 176, 152], dtype=np.uint8)   # roof / wall

    parts = [(px, py, pz, col)]
    if not a.no_ground:
        gy, gx = np.mgrid[0:n, 0:n]
        shade = np.clip((dtm - dtm.min()) / max(float(np.ptp(dtm)), 1.0), 0, 1).ravel()[:, None]
        gcol = (np.array([58, 52, 44], np.float32)
                + np.array([70, 62, 50], np.float32) * shade).astype(np.uint8)
        parts.append(((gx.ravel() + 0.5) * res, (gy.ravel() + 0.5) * res,
                      dtm.ravel(), gcol))

    comments = [
        f"origin EPSG:25833 easting {x0:.1f} northing {y0:.1f}",
        f"local metres; add the origin above to get UTM33N",
        f"voxels {res:g} m horizontal, {vh:g} m vertical bands above ground",
        "green = vegetation (LSC returns), grey = building footprint (OSM), brown = DGM1 bare earth",
    ]

    if a.cubes:
        vx = np.concatenate([px, py, pz]).reshape(3, -1).T - np.array(
            [res / 2, res / 2, vh / 2], np.float32)
        scale = np.array([res, res, vh], np.float32)
        nv = len(vx)
        verts = np.empty(nv * 8, dtype=VERT)
        corners = (CUBE_V[None, :, :] * scale).reshape(1, 8, 3)
        pos = (vx[:, None, :] + corners).reshape(-1, 3)
        verts["x"], verts["y"], verts["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        c8 = np.repeat(col, 8, axis=0)
        verts["r"], verts["g"], verts["b"] = c8[:, 0], c8[:, 1], c8[:, 2]
        faces = (CUBE_F[None, :, :] + (np.arange(nv) * 8)[:, None, None]).reshape(-1, 3)
        write_ply(out, verts, faces, comments)
        print(f"{nv:,} voxel cubes -> {len(faces):,} triangles")
    else:
        xs = np.concatenate([p[0] for p in parts])
        ys = np.concatenate([p[1] for p in parts])
        zs = np.concatenate([p[2] for p in parts])
        cs = np.concatenate([p[3] for p in parts])
        verts = np.empty(len(xs), dtype=VERT)
        verts["x"], verts["y"], verts["z"] = xs, ys, zs
        verts["r"], verts["g"], verts["b"] = cs[:, 0], cs[:, 1], cs[:, 2]
        write_ply(out, verts, None, comments)
        print(f"{len(ii):,} occupied voxels"
              + ("" if a.no_ground else f" + {n*n:,} ground points"))

    print(f"  box {a.size:g} m at E {x0:.0f} N {y0:.0f}, {is_solid.sum():,} solid")
    print(f"  {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
