"""Bake 1 m detail height tiles -- DOM1's native resolution -- for close-range view.

The baked-in-one-piece display surface is 8 m, which is why canopy reads as a
smooth blanket. These tiles are the same display rule (bare earth under buildings,
DOM1 everywhere else) at 1 m, cut into 500 m tiles so the viewer can stream only
what is near the camera, the way the orthophoto does.

Encoded RGB: R,G = height hi/lo bytes (as elsewhere), B = vegetation flag.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           label, maximum_filter)

Image.MAX_IMAGE_PIXELS = None
DATA = Path("data")
OUT = Path("viewer/data/detail")

TILE_M = 500          # tile edge in metres -> 500x500 px at 1 m
MESH_M = 2            # MUST match the viewer's detail mesh spacing (seg = TILE_M/MESH_M)
MIN_STAND_M2 = 300    # smallest tree stand drawn as a canopy SURFACE
CANOPY_MIN_H = 2.0    # metres above ground that counts as canopy
CANOPY_CLOSE = 1      # gentle: bridges ~2 m, too small to swallow a footpath
CANOPY_FILL_M2 = 800  # enclosed gaps up to this area are filled
H0, HSCALE = 80.0, 50.0

MINX, MAXX = 308000, 324000
MINY, MAXY = 5678000, 5696000


def load_tif(path: Path):
    with zipfile.ZipFile(path) as zf:
        tif = next(n for n in zf.namelist() if n.endswith(".tif"))
        tfw = next(n for n in zf.namelist() if n.endswith(".tfw"))
        w = [float(v) for v in zf.read(tfw).decode().split()]
        a = np.asarray(Image.open(io.BytesIO(zf.read(tif))), dtype=np.float32).copy()
    a[a <= -1000.0] = np.nan
    x0 = w[4] - w[0] / 2.0
    ytop = w[5] - w[3] / 2.0
    return a, x0, ytop



def fill_enclosed(v, max_m2):
    """Fill background pockets fully enclosed by canopy, up to max_m2.

    Deliberately NOT a big morphological closing. A closing wide enough to shut
    the gaps between crowns is also wide enough to bridge a footpath -- the
    Fockeberg's spiral path was swallowed and lifted to canopy height that way.
    A path reaches the edge of the stand, so it is not enclosed and survives here;
    the gaps between crowns are enclosed, and they get filled.
    """
    lab, n = label(~v)
    if not n:
        return v
    sizes = np.bincount(lab.ravel())
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = sizes <= max_m2
    fill[0] = False
    fill[edge[edge > 0]] = False          # touches open ground -> a path, keep it
    return v | fill[lab]


def one_zip(args):
    e, n, bmask_pack = args
    dom = DATA / f"geosn/dom1/dom1_33{e:03d}_{n:04d}_2_sn_tiff.zip"
    dgm = DATA / f"geosn/dgm1/dgm1_33{e:03d}_{n:04d}_2_sn_tiff.zip"
    if not dom.exists() or not dgm.exists():
        return 0
    try:
        dsm, x0, ytop = load_tif(dom)
        dtm, _, _ = load_tif(dgm)
    except Exception:
        return 0
    if dsm.shape != dtm.shape:
        return 0

    # north-up -> south-up so rows run the same way as everything else here
    dsm = dsm[::-1]
    dtm = dtm[::-1]
    ybot = ytop - dsm.shape[0]

    bmask, bminx, bminy, bres = bmask_pack
    ny, nx = dsm.shape
    jj = ((x0 - bminx) / bres + np.arange(nx) / bres).astype(np.int64)
    ii = ((ybot - bminy) / bres + np.arange(ny) / bres).astype(np.int64)
    jj = np.clip(jj, 0, bmask.shape[1] - 1)
    ii = np.clip(ii, 0, bmask.shape[0] - 1)
    bl = bmask[np.ix_(ii, jj)]

    dtm = np.where(np.isnan(dtm), dsm, dtm)
    dsm = np.where(np.isnan(dsm), dtm, dsm)
    med = np.nanmedian(dtm)
    dsm = np.nan_to_num(dsm, nan=med)
    dtm = np.nan_to_num(dtm, nan=med)

    # TERRAIN and CANOPY are kept as separate surfaces, exactly as in the coarse
    # layer. Baking one combined surface makes the ground itself rise to treetop
    # height under woodland, and at 1 m every individual crown becomes a spike.
    #   R,G = bare-earth height        B = canopy height ABOVE ground, in metres
    # The canopy height is dilated past its mask and the mask is then eroded, so no
    # drawn quad ever slopes from crown down to bare earth (that made floating slabs).
    # B packs BOTH, because the height must extend further than the drawn area:
    #   bit 7      = eroded canopy mask (where fragments are actually drawn)
    #   bits 0..6  = canopy height above ground in metres, DILATED past that mask
    # Dilated height + eroded mask means every drawn fragment sits on the flat part
    # of the blanket, so no quad slopes from crown down to bare earth.
    cab = np.where(bl, 0.0, np.maximum(dsm - dtm, 0.0))
    cab_dil = np.clip(maximum_filter(cab, size=7), 0, 127)
    # Canopy mask. Four steps in a specific order, each fixing a real artefact:
    #
    # 1. Gentle CLOSE plus enclosed-hole fill. At 1 m the gaps between crowns
    #    resolve, so an unclosed mask is a lace of holes that reads as scattered
    #    debris -- but a closing big enough to shut them also bridges footpaths.
    # 2. Reduce to MESH-CELL resolution by majority -- mask and height must be
    #    constant per mesh cell, or a vertex landing on a crown lifts a whole quad
    #    whose fragments mostly fail the mask (the "floating pieces").
    # 3. ERODE by one whole cell, in CELL space. A vertex sits at a cell CORNER, so
    #    a quad spans two cells; without this a masked cell next to an unmasked one
    #    is drawn as a near-vertical sliver running down to bare earth.
    # 4. Drop small components LAST, so whatever the erosion shattered goes with them.
    #
    # Measured on the Fockeberg source tile: 58.8% cover in 51 components with ZERO
    # under 40 m2, versus the coarse blanket's 58.4% over the same ground. The
    # previous settings gave 8.4% -- seven times too sparse, which is exactly why
    # the detailed canopy looked like scattered quads.
    m = MESH_M
    h2, w2 = (ny // m) * m, (nx // m) * m
    v = binary_closing(cab >= CANOPY_MIN_H, iterations=CANOPY_CLOSE)
    v = fill_enclosed(v, CANOPY_FILL_M2)
    cm = v[:h2, :w2].reshape(h2 // m, m, w2 // m, m).mean(axis=(1, 3)) > 0.5
    cm = binary_erosion(cm, iterations=1)
    lab, ncomp = label(cm)
    if ncomp:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        cm = (sizes >= MIN_STAND_M2 / (m * m))[lab]
    ch = maximum_filter(cab_dil[:h2, :w2].reshape(h2 // m, m, w2 // m, m).max(axis=(1, 3)),
                        size=5)

    def up(a):
        return np.repeat(np.repeat(a, m, axis=0), m, axis=1)

    cmask_c = np.zeros(cab.shape, dtype=bool)
    cab_c = np.zeros(cab.shape, dtype=cab_dil.dtype)
    cmask_c[:h2, :w2] = up(cm)
    cab_c[:h2, :w2] = up(ch)
    cab_out = cab_c.astype(np.uint8) | (cmask_c.astype(np.uint8) * 128)

    written = 0
    per = TILE_M
    for ty in range(ny // per):
        for tx in range(nx // per):
            sub = dtm[ty * per:(ty + 1) * per, tx * per:(tx + 1) * per]
            vsub = cab_out[ty * per:(ty + 1) * per, tx * per:(tx + 1) * per]
            gx = int((x0 + tx * per - MINX) // TILE_M)
            gy = int((ybot + ty * per - MINY) // TILE_M)
            if not (0 <= gx < (MAXX - MINX) // TILE_M and 0 <= gy < (MAXY - MINY) // TILE_M):
                continue
            enc = np.clip((sub - H0) * HSCALE, 0, 65535).astype(np.uint16)
            rgb = np.zeros((per, per, 3), dtype=np.uint8)
            rgb[:, :, 0] = (enc >> 8).astype(np.uint8)
            rgb[:, :, 1] = (enc & 0xFF).astype(np.uint8)
            rgb[:, :, 2] = vsub
            Image.fromarray(rgb, "RGB").save(OUT / f"{gx}_{gy}.png", optimize=True)
            written += 1
    return written


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = json.loads((DATA / "grid_meta.json").read_text())
    res = g["res"]

    # Building mask once, at the 2 m analysis grid, reused by every worker.
    from rank_spots import rasterize, load_json
    om = json.loads((DATA / "meta.json").read_text())
    ox, oy = om["origin"]
    ny = int((g["maxy"] - g["miny"]) / res)
    nx = int((g["maxx"] - g["minx"]) / res)
    print("rasterising building mask ...")
    bmask = rasterize([[[x + ox, y + oy] for x, y in p["r"]]
                       for p in load_json("buildings.json")],
                      (ny, nx), g["minx"], g["miny"], res)
    bmask = binary_dilation(bmask, iterations=4)
    pack = (bmask, g["minx"], g["miny"], res)

    jobs = [(e, n, pack) for e in range(MINX // 1000, MAXX // 1000, 2)
            for n in range(MINY // 1000, MAXY // 1000, 2)]
    print(f"{len(jobs)} source tiles -> {TILE_M} m detail tiles at 1 m")

    total = 0
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for k, w in enumerate(ex.map(one_zip, jobs), 1):
            total += w
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)}  {total} tiles")

    size = sum(p.stat().st_size for p in OUT.glob("*.png"))
    (OUT / "manifest.json").write_text(json.dumps({
        "tile_m": TILE_M, "px": TILE_M, "minx": MINX, "miny": MINY,
        "maxx": MAXX, "maxy": MAXY, "h0": H0, "scale": HSCALE,
        "tiles": sorted(p.stem for p in OUT.glob("*.png")),
    }, separators=(",", ":")))
    print(f"\n{total} tiles, {size/1e6:.0f} MB, {size/max(total,1)/1e3:.0f} kB each")


if __name__ == "__main__":
    main()
