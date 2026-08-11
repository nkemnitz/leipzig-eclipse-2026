"""Fetch terrain elevation for the study area from AWS 'terrarium' tiles.

These are open (no key, no account). Encoding: h = R*256 + G + B/256 - 32768 metres.
Resampled onto the same local UTM33N metric grid used everywhere else.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

Z = 13  # ~12 m/px at this latitude; source data is ~30 m so this is already oversampled
BBOX = dict(south=51.25, north=51.42, west=12.22, east=12.55)
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE = Path("data/dem_tiles")
OUT = Path("data")

ORIGIN_X, ORIGIN_Y = 315000.0, 5690000.0
GRID_RES = 5.0  # metres -- terrain only; buildings are rasterised finer later

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)


def deg2tile(lat, lon, z):
    n = 2.0**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{z}_{x}_{y}.png"
    if not p.exists():
        r = requests.get(URL.format(z=z, x=x, y=y), timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    img = Image.open(io.BytesIO(p.read_bytes())).convert("RGB")
    a = np.asarray(img, dtype=np.float64)
    return a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0 - 32768.0


def main():
    x0f, y1f = deg2tile(BBOX["north"], BBOX["west"], Z)
    x1f, y0f = deg2tile(BBOX["south"], BBOX["east"], Z)
    tx0, tx1 = int(math.floor(x0f)), int(math.floor(x1f))
    ty0, ty1 = int(math.floor(y1f)), int(math.floor(y0f))
    print(f"tiles x {tx0}..{tx1}  y {ty0}..{ty1}  ({(tx1-tx0+1)*(ty1-ty0+1)} tiles)")

    ts = 256
    mosaic = np.zeros(((ty1 - ty0 + 1) * ts, (tx1 - tx0 + 1) * ts), dtype=np.float64)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mosaic[(ty - ty0) * ts:(ty - ty0 + 1) * ts,
                   (tx - tx0) * ts:(tx - tx0 + 1) * ts] = fetch_tile(Z, tx, ty)
    print(f"mosaic {mosaic.shape}  elev {mosaic.min():.1f}..{mosaic.max():.1f} m")

    # Target local metric grid
    cx0, cy0 = to_utm.transform(BBOX["west"], BBOX["south"])
    cx1, cy1 = to_utm.transform(BBOX["east"], BBOX["north"])
    cx2, cy2 = to_utm.transform(BBOX["west"], BBOX["north"])
    cx3, cy3 = to_utm.transform(BBOX["east"], BBOX["south"])
    minx, maxx = min(cx0, cx2), max(cx1, cx3)
    miny, maxy = min(cy0, cy3), max(cy1, cy2)
    minx = math.floor((minx - ORIGIN_X) / GRID_RES) * GRID_RES
    miny = math.floor((miny - ORIGIN_Y) / GRID_RES) * GRID_RES
    maxx = math.ceil((maxx - ORIGIN_X) / GRID_RES) * GRID_RES
    maxy = math.ceil((maxy - ORIGIN_Y) / GRID_RES) * GRID_RES

    nx = int((maxx - minx) / GRID_RES)
    ny = int((maxy - miny) / GRID_RES)
    print(f"grid {nx} x {ny} @ {GRID_RES} m")

    gx = minx + (np.arange(nx) + 0.5) * GRID_RES
    gy = miny + (np.arange(ny) + 0.5) * GRID_RES
    gxx, gyy = np.meshgrid(gx, gy)
    lon, lat = to_wgs.transform(gxx + ORIGIN_X, gyy + ORIGIN_Y)

    # Web-mercator pixel coords of every grid cell, then bilinear sample
    n = 2.0**Z
    px = (lon + 180.0) / 360.0 * n
    py = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    fx = (px - tx0) * ts
    fy = (py - ty0) * ts

    h, w = mosaic.shape
    fx = np.clip(fx, 0, w - 1.001)
    fy = np.clip(fy, 0, h - 1.001)
    ix, iy = np.floor(fx).astype(np.int32), np.floor(fy).astype(np.int32)
    dx, dy = fx - ix, fy - iy
    dem = (
        mosaic[iy, ix] * (1 - dx) * (1 - dy)
        + mosaic[iy, ix + 1] * dx * (1 - dy)
        + mosaic[iy + 1, ix] * (1 - dx) * dy
        + mosaic[iy + 1, ix + 1] * dx * dy
    ).astype(np.float32)

    np.save(OUT / "dem.npy", dem)
    (OUT / "dem_meta.json").write_text(json.dumps({
        "res": GRID_RES, "minx": minx, "miny": miny, "nx": nx, "ny": ny,
        "origin": [ORIGIN_X, ORIGIN_Y], "crs": "EPSG:25833", "zoom": Z,
        "source": "AWS terrarium (elevation-tiles-prod), public, no auth",
        "note": "row 0 = miny (south); y increases northward",
    }, indent=2))
    print(f"dem.npy  {dem.shape}  {dem.min():.1f}..{dem.max():.1f} m")
    # Sanity: Leipzig centre should be ~110-120 m
    cxm, cym = to_utm.transform(12.3731, 51.3397)
    i = int((cym - ORIGIN_Y - miny) / GRID_RES)
    j = int((cxm - ORIGIN_X - minx) / GRID_RES)
    print(f"Leipzig Markt elevation = {dem[i, j]:.1f} m (expect ~113 m)")


if __name__ == "__main__":
    main()
