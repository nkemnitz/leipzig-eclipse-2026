"""Mosaic the GeoSN DOM1/DGM1 tiles into a single 2 m grid over the study area.

DOM1 (surface, incl. roofs and canopy) is downsampled with MAX: for a
"can I see the sun from here" question, over-blocking is the safe error --
it makes us reject a marginal spot rather than send someone to a shadowed one.
DGM1 (bare earth) is downsampled with MEAN, which is the right estimator for a
smooth terrain surface.

Output grid: EPSG:25833, row 0 = SOUTH edge, col 0 = WEST edge.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

RES = 2.0
# Full grid (incl. western/northern ray-march context)
MINX, MINY = 302000, 5676000
MAXX, MAXY = 324000, 5698000
NX = int((MAXX - MINX) / RES)  # 11000
NY = int((MAXY - MINY) / RES)  # 11000

# Sub-region results are actually reported for. Extends south to 5678000 to cover
# the Neuseenland lakes (Cospudener, Zwenkauer, Markkleeberger), whose open water
# to the west is the most promising low-horizon terrain anywhere near Leipzig.
OUT_MINX, OUT_MINY = 308000, 5678000
OUT_MAXX, OUT_MAXY = 324000, 5696000

SRC = Path("data/geosn")
OUT = Path("data")


def load_tile(path: Path):
    """Return (arr_1m_north_up, easting0, northing_top) or None."""
    try:
        with zipfile.ZipFile(path) as zf:
            tif = next(n for n in zf.namelist() if n.endswith(".tif"))
            tfw = next(n for n in zf.namelist() if n.endswith(".tfw"))
            w = [float(v) for v in zf.read(tfw).decode().split()]
            arr = np.asarray(Image.open(io.BytesIO(zf.read(tif))), dtype=np.float32).copy()
        # Tiles straddling the Sachsen-Anhalt border carry -9999 outside coverage.
        arr[arr <= -1000.0] = np.nan
        # world file: px_x, rot, rot, px_y(neg), x_of_centre_of_topleft, y_of_centre
        x0 = w[4] - w[0] / 2.0
        ytop = w[5] - w[3] / 2.0  # w[3] is negative
        return arr, x0, ytop
    except Exception as exc:
        print(f"  !! {path.name}: {type(exc).__name__} {exc}")
        return None


def downsample(a: np.ndarray, how: str) -> np.ndarray:
    f = int(RES)
    h, w = a.shape
    a = a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f)
    with warnings.catch_warnings():  # all-NaN blocks are expected at the border
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=(1, 3)) if how == "max" else np.nanmean(a, axis=(1, 3))


def fill_nearest_along_rows(a: np.ndarray) -> np.ndarray:
    """Fill NaNs with the nearest valid value in the same row.

    The gap here is a north-south strip on the western edge (outside Saxony), so
    a 1-D fill along x propagates real neighbouring terrain into it.
    """
    out = a.copy()
    valid = ~np.isnan(out)
    idx = np.where(valid, np.arange(out.shape[1])[None, :], -1)
    fwd = np.maximum.accumulate(idx, axis=1)
    idx_b = np.where(valid, np.arange(out.shape[1])[None, :], out.shape[1])
    bwd = np.minimum.accumulate(idx_b[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(out.shape[0])[:, None]
    cols = np.arange(out.shape[1])[None, :]
    df = np.where(fwd >= 0, cols - fwd, 1 << 30)
    db = np.where(bwd < out.shape[1], bwd - cols, 1 << 30)
    take = np.where(df <= db, np.clip(fwd, 0, out.shape[1] - 1),
                    np.clip(bwd, 0, out.shape[1] - 1))
    filled = out[rows, take]
    return np.where(valid, out, filled)


def build(prod: str, how: str) -> np.ndarray:
    files = sorted((SRC / prod).glob("*.zip"))
    print(f"{prod}: {len(files)} tiles, downsample={how}")
    mosaic = np.full((NY, NX), np.nan, dtype=np.float32)

    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        for path, res in zip(files, ex.map(load_tile, files)):
            if res is None:
                continue
            arr, x0, ytop = res
            small = downsample(arr, how).astype(np.float32)
            n = small.shape[0]
            # tile bottom-left in grid coords; flip so row 0 = south
            ybot = ytop - arr.shape[0]
            j0 = int(round((x0 - MINX) / RES))
            i0 = int(round((ybot - MINY) / RES))
            small = small[::-1]  # north-up -> south-up
            i1, j1 = i0 + n, j0 + small.shape[1]
            si0, sj0 = max(0, -i0), max(0, -j0)
            i0c, j0c = max(i0, 0), max(j0, 0)
            i1c, j1c = min(i1, NY), min(j1, NX)
            if i1c <= i0c or j1c <= j0c:
                continue
            mosaic[i0c:i1c, j0c:j1c] = small[si0:si0 + (i1c - i0c), sj0:sj0 + (j1c - j0c)]

    miss = int(np.isnan(mosaic).sum())
    print(f"  filled {(1 - miss / mosaic.size) * 100:.2f}%  range {np.nanmin(mosaic):.1f}..{np.nanmax(mosaic):.1f} m")
    return mosaic


def main():
    dsm = build("dom1", "max")
    dtm = build("dgm1", "mean")

    # Gaps are the out-of-Saxony strip on the far western edge. It sits 3.5-6 km
    # west of the reported area and is only ever distant ray-march context: to
    # block a 3.3 deg sun from 5 km it would have to stand 260 m above the
    # observer, which flat farmland does not. Fill from the nearest real terrain.
    nodata = np.isnan(dsm) | np.isnan(dtm)
    dtm = np.where(np.isnan(dtm), dsm, dtm)
    dsm = np.where(np.isnan(dsm), dtm, dsm)
    if nodata.any():
        print(f"  filling {int(nodata.sum()):,} nodata cells ({100*nodata.mean():.2f}%) "
              f"from nearest valid terrain in-row")
        dsm = fill_nearest_along_rows(dsm)
        dtm = fill_nearest_along_rows(dtm)
    med = float(np.nanmedian(dtm))
    dsm = np.nan_to_num(dsm, nan=med).astype(np.float32)
    dtm = np.nan_to_num(dtm, nan=med).astype(np.float32)
    np.save(OUT / "nodata_mask.npy", nodata)

    # Object height = surface minus bare earth. Clamp tiny negatives from
    # independent-acquisition noise between the two models.
    obj = np.maximum(dsm - dtm, 0.0).astype(np.float32)

    np.save(OUT / "dsm2m.npy", dsm)
    np.save(OUT / "dtm2m.npy", dtm)
    np.save(OUT / "obj2m.npy", obj)
    meta = {
        "res": RES, "minx": MINX, "miny": MINY, "nx": NX, "ny": NY,
        "maxx": MAXX, "maxy": MAXY,
        "out_minx": OUT_MINX, "out_maxx": OUT_MAXX,
        "out_miny": OUT_MINY, "out_maxy": OUT_MAXY,
        "crs": "EPSG:25833", "vertical": "DHHN2016",
        "row0": "south", "col0": "west",
        "source": "GeoSN DOM1/DGM1 1 m, dl-de/by-2-0",
        "dsm_downsample": "max", "dtm_downsample": "mean",
    }
    (OUT / "grid_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ngrid {NY} x {NX} @ {RES} m")
    print(f"  terrain  {dtm.min():7.1f} .. {dtm.max():7.1f} m")
    print(f"  surface  {dsm.min():7.1f} .. {dsm.max():7.1f} m")
    print(f"  objects  {obj.min():7.1f} .. {obj.max():7.1f} m")
    for pct in (50, 90, 99, 99.9):
        print(f"  object height p{pct:<5} {np.percentile(obj, pct):6.1f} m")


if __name__ == "__main__":
    main()
