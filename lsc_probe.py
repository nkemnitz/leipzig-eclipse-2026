"""Does deriving the surface ourselves from the raw LSC point cloud beat DOM1?

DOM1 is a rasterised first-return surface: one height per square metre. The cloud
it came from has every return, so for each cell we can take the true maximum over
all returns instead of one resampled value -- which should close most of the
leaf-off "lace" without any morphological fudging -- and, uniquely, measure canopy
POROSITY: what fraction of pulses got through to the ground. That is the quantity
an opaque-surface model cannot express, and it is exactly the leaf-off question.

Scope: the single 2 km tile containing the Fockeberg.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw

TILE = "lsc_33316_5688.zip"
X0, Y0, N = 316000, 5688000, 2000        # tile origin (UTM33N) and 1 m size
GROUND_CLASS = 2
VEG_CLASSES = (3, 4, 5)


def cell_max(idx, val, n, init=-9999.0):
    """Max of val per cell index, via sort + reduceat (ufunc.at is far too slow)."""
    order = np.argsort(idx, kind="stable")
    i_s, v_s = idx[order], val[order]
    uniq, starts = np.unique(i_s, return_index=True)
    out = np.full(n, init, dtype=np.float32)
    out[uniq] = np.maximum.reduceat(v_s, starts)
    return out


def cell_min(idx, val, n, init=9999.0):
    order = np.argsort(idx, kind="stable")
    i_s, v_s = idx[order], val[order]
    uniq, starts = np.unique(i_s, return_index=True)
    out = np.full(n, init, dtype=np.float32)
    out[uniq] = np.minimum.reduceat(v_s, starts)
    return out


def main():
    import laspy

    print("reading the point cloud ...")
    with zipfile.ZipFile(f"data/lsc/{TILE}") as z:
        name = next(n for n in z.namelist() if n.endswith(".laz"))
        las = laspy.read(io.BytesIO(z.read(name)))
    x = np.asarray(las.x); y = np.asarray(las.y); z_ = np.asarray(las.z, dtype=np.float32)
    cls = np.asarray(las.classification)
    nret = np.asarray(las.number_of_returns)
    print(f"  {len(x):,} points over 4 km2  =  {len(x)/4e6:.1f} pts/m2")
    import collections
    names = {1: "unclassified", 2: "ground", 3: "low veg", 4: "med veg", 5: "high veg",
             6: "building", 7: "noise", 9: "water"}
    for c, k in sorted(collections.Counter(cls).items(), key=lambda kv: -kv[1])[:8]:
        print(f"    class {c:2d} {names.get(c,'other'):13s} {k:11,}  {100*k/len(x):5.1f}%")
    print(f"  multi-return pulses: {100*(nret>1).mean():.1f}%  (max {nret.max()} returns)")

    ix = np.clip(((x - X0)).astype(np.int64), 0, N - 1)
    iy = np.clip(((y - Y0)).astype(np.int64), 0, N - 1)
    idx = iy * N + ix                                   # row 0 = south, like everything else

    print("\nrasterising at 1 m ...")
    surf = cell_max(idx, z_, N * N).reshape(N, N)       # true max over ALL returns
    g = cls == GROUND_CLASS
    grnd = cell_min(idx[g], z_[g], N * N).reshape(N, N)

    # fill ground gaps from the neighbourhood so height-above-ground is defined
    from scipy.ndimage import generic_filter, minimum_filter
    miss = grnd > 9000
    print(f"  ground returns missing in {100*miss.mean():.1f}% of cells -> filling")
    filled = grnd.copy(); filled[miss] = np.nan
    for size in (5, 11, 25, 51):
        if not np.isnan(filled).any():
            break
        sm = minimum_filter(np.nan_to_num(filled, nan=9999.0), size=size)
        filled = np.where(np.isnan(filled), np.where(sm > 9000, np.nan, sm), filled)
    filled = np.nan_to_num(filled, nan=float(np.nanmedian(filled)))

    lsc_cab = np.maximum(surf - filled, 0)

    # POROSITY: of all pulses over this cell, what fraction reached the ground?
    tot = np.bincount(idx, minlength=N * N).reshape(N, N).astype(np.float32)
    gnd_cnt = np.bincount(idx[g], minlength=N * N).reshape(N, N).astype(np.float32)
    veg = np.isin(cls, VEG_CLASSES)
    veg_cnt = np.bincount(idx[veg], minlength=N * N).reshape(N, N).astype(np.float32)

    # ---- compare against DOM1/DGM1 for the same tile ----
    def rd(p):
        with zipfile.ZipFile(p) as zf:
            t = next(n for n in zf.namelist() if n.endswith(".tif"))
            return np.asarray(Image.open(io.BytesIO(zf.read(t))), dtype=np.float32)[::-1]
    dom = rd("data/geosn/dom1/dom1_33316_5688_2_sn_tiff.zip")
    dgm = rd("data/geosn/dgm1/dgm1_33316_5688_2_sn_tiff.zip")
    dom_cab = np.maximum(dom - dgm, 0)

    # OSM woodland as an independent "this is forest" statement
    om = json.load(open("data/meta.json")); ox, oy = om["origin"]
    img = Image.new("1", (N, N), 0); d = ImageDraw.Draw(img)
    for w in json.load(open("data/woods.json")):
        pts = [((px + ox - X0), (py + oy - Y0)) for px, py in w["r"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    wood = np.asarray(img, dtype=bool)

    print(f"\n=== inside OSM woodland ({wood.sum():,} m2) ===")
    print(f"{'':22s} {'DOM1':>8s} {'LSC':>8s}")
    print(f"{'cells reading <2 m':22s} {100*(dom_cab[wood]<2).mean():7.1f}% "
          f"{100*(lsc_cab[wood]<2).mean():7.1f}%   <-- the leaf-off lace")
    print(f"{'median height >2 m':22s} {np.median(dom_cab[wood][dom_cab[wood]>=2]):7.1f}m "
          f"{np.median(lsc_cab[wood][lsc_cab[wood]>=2]):7.1f}m")
    print(f"{'mean height':22s} {dom_cab[wood].mean():7.1f}m {lsc_cab[wood].mean():7.1f}m")
    diff = lsc_cab[wood] - dom_cab[wood]
    print(f"\nLSC surface is higher than DOM1 by mean {diff.mean():+.2f} m, "
          f"p90 {np.percentile(diff,90):+.2f} m")

    with np.errstate(invalid="ignore", divide="ignore"):
        por = np.where(tot > 0, gnd_cnt / tot, np.nan)
    pw = por[wood & (tot > 0)]
    print(f"\n=== POROSITY (fraction of returns that are GROUND) ===")
    print(f"  inside woodland : {np.nanmean(pw)*100:5.1f}%  "
          f"(p10 {np.nanpercentile(pw,10)*100:.0f}%, p90 {np.nanpercentile(pw,90)*100:.0f}%)")
    op = por[(~wood) & (tot > 0)]
    print(f"  outside woodland: {np.nanmean(op)*100:5.1f}%")
    print(f"  => in leaf-off January, ~{np.nanmean(pw)*100:.0f}% of pulses reached the")
    print(f"     ground through the canopy. In August that number would be far lower,")
    print(f"     which is the bias an opaque-surface model cannot see.")

    np.savez_compressed("data/lsc_fockeberg.npz", lsc_cab=lsc_cab.astype(np.float32),
                        lsc_surf=surf.astype(np.float32), lsc_ground=filled.astype(np.float32),
                        porosity=por.astype(np.float32), wood=wood)
    print("\nsaved data/lsc_fockeberg.npz")


if __name__ == "__main__":
    main()
