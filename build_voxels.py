"""Build a voxel occupancy grid from the LSC point cloud.

The DSM march treats a tree as an opaque column from the ground to the crown top.
At a 3.4 deg sun the sight line is only 4.7 m up at 50 m and 7.6 m at 100 m, while
the median crown base in Leipzig's woodland measured 10.7 m -- so the line passes
UNDER the crowns and the column model wrongly blocks it. Occupancy resolved in
height fixes that: only material AT the ray's height blocks.

Grid: 2 m horizontally (matching the analysis grid), 32 vertical bands of 2 m
measured ABOVE GROUND, packed one uint32 per cell. Anything above 64 m is folded
into the top band; that is only ever buildings, which are opaque regardless.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA = Path("data")
OUT = DATA / "voxels"

E_KM = range(306, 324, 2)
N_KM = range(5678, 5698, 2)
MINX, MINY = 306000, 5678000
RES = 2.0                 # horizontal cell size
VH = 2.0                  # vertical band height
NB = 32                   # bands -> one uint32 per cell
NX = int(sum(1 for _ in E_KM) * 2000 / RES)
NY = int(sum(1 for _ in N_KM) * 2000 / RES)
TILE = int(2000 / RES)    # 1000 cells per 2 km tile
MIN_RETURNS = 2           # a voxel counts as material at >= this many returns
CHUNK = 8_000_000


def one_tile(args):
    e, n = args
    laz = DATA / f"lsc/lsc_33{e:03d}_{n:04d}_2_sn_laz.zip"
    dgm = DATA / f"geosn/dgm1/dgm1_33{e:03d}_{n:04d}_2_sn_tiff.zip"
    if not laz.exists() or not dgm.exists():
        return e, n, None, 0

    import laspy
    with zipfile.ZipFile(dgm) as zf:
        t = next(x for x in zf.namelist() if x.endswith(".tif"))
        w = [float(v) for v in zf.read(next(x for x in zf.namelist()
                                            if x.endswith(".tfw"))).decode().split()]
        g1 = np.asarray(Image.open(io.BytesIO(zf.read(t))), dtype=np.float32).copy()
    g1[g1 <= -1000.0] = np.nan
    gx0 = w[4] - w[0] / 2.0
    gytop = w[5] - w[3] / 2.0
    g1 = g1[::-1]                                    # north-up -> south-up
    gy0 = gytop - g1.shape[0]
    med = np.nanmedian(g1)
    g1 = np.nan_to_num(g1, nan=med)

    counts = np.zeros(TILE * TILE * NB, dtype=np.uint16)
    npts = 0
    with zipfile.ZipFile(laz) as zf:
        nm = next(x for x in zf.namelist() if x.endswith(".laz"))
        raw = zf.read(nm)
    with laspy.open(io.BytesIO(raw)) as fh:
        for pts in fh.chunk_iterator(CHUNK):
            x = np.asarray(pts.x); y = np.asarray(pts.y)
            z = np.asarray(pts.z, dtype=np.float32)
            cls = np.asarray(pts.classification)
            npts += len(x)
            gi = np.clip((y - gy0).astype(np.int64), 0, g1.shape[0] - 1)
            gj = np.clip((x - gx0).astype(np.int64), 0, g1.shape[1] - 1)
            hag = z - g1[gi, gj]
            keep = (cls != 2) & (hag > 0.75)         # non-ground, above ground fuzz
            if not keep.any():
                continue
            # tile-local cell indices; row 0 = south, as everywhere else here
            ci = np.clip((y[keep] - gy0) / RES, 0, TILE - 1).astype(np.int64)
            cj = np.clip((x[keep] - gx0) / RES, 0, TILE - 1).astype(np.int64)
            cb = np.clip(hag[keep] / VH, 0, NB - 1).astype(np.int64)
            flat = (ci * TILE + cj) * NB + cb
            c = np.bincount(flat, minlength=TILE * TILE * NB)
            np.minimum(c, 60000, out=c)
            counts += c.astype(np.uint16)
            del c

    occ = (counts.reshape(TILE, TILE, NB) >= MIN_RETURNS)
    bits = np.zeros((TILE, TILE), dtype=np.uint32)
    for b in range(NB):
        bits |= occ[:, :, b].astype(np.uint32) << np.uint32(b)
    return e, n, bits, npts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((NY, NX), dtype=np.uint32)
    jobs = [(e, n) for e in E_KM for n in N_KM]
    print(f"{len(jobs)} tiles -> voxel grid {NY} x {NX} x {NB} bands "
          f"({grid.nbytes/1e6:.0f} MB)")

    done = 0
    total_pts = 0
    with cf.ProcessPoolExecutor(max_workers=7) as ex:
        for e, n, bits, npts in ex.map(one_tile, jobs):
            done += 1
            total_pts += npts
            if bits is None:
                print(f"  [{done}/{len(jobs)}] {e}_{n} MISSING")
                continue
            j0 = int((e * 1000 - MINX) / RES)
            i0 = int((n * 1000 - MINY) / RES)
            grid[i0:i0 + TILE, j0:j0 + TILE] = bits
            if done % 5 == 0:
                occ = np.count_nonzero(grid) / grid.size
                print(f"  [{done}/{len(jobs)}] {total_pts/1e6:.0f} M pts, "
                      f"{100*occ:.1f}% of cells have material", flush=True)

    np.save(OUT / "occ.npy", grid)
    (OUT / "meta.json").write_text(json.dumps({
        "minx": MINX, "miny": MINY, "res": RES, "nx": NX, "ny": NY,
        "vh": VH, "bands": NB, "min_returns": MIN_RETURNS,
        "note": "bit b set = material between b*VH and (b+1)*VH metres above ground",
    }, indent=2))
    nz = np.count_nonzero(grid)
    print(f"\n{total_pts/1e6:.0f} M points -> {nz:,} cells with material "
          f"({100*nz/grid.size:.1f}%), {grid.nbytes/1e6:.0f} MB")


if __name__ == "__main__":
    main()
