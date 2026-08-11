"""Where can you actually see the eclipsed sun from?

For every 2 m ground cell we march a ray toward the sun and ask whether anything --
roof, tree canopy, or terrain -- rises above the line of sight. This is a real
line-of-sight computation against a 1 m laser-scanned surface model, not a
rendering trick.

At maximum eclipse the sun is only ~3.3 deg up, so two effects that are normally
ignorable are included:
  * Earth curvature, which drops a target by d^2/(2R) -- ~2 m at 5 km, i.e. more
    than eye height.
  * Atmospheric refraction, which lifts the apparent sun and effectively flattens
    the earth to R_eff = 7/6 R. Refraction is applied to the sun altitude via
    solar.apparent_altitude; the curvature term uses R_eff to match.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

R_EFF = 7.0 / 6.0 * 6371000.0  # refraction-corrected earth radius
EYE_H = 1.7                     # standing observer
DMAX = 6000.0                   # max sight-line distance considered

DATA = Path("data")


def grid_convergence_deg(x, y, crs="EPSG:25833"):
    """Angle from grid north to true north at (x, y), in degrees.

    Solar azimuths are TRUE azimuths, but the ray march steps through UTM grid
    columns and rows. Leipzig sits ~2.6 deg west of the zone-33 central meridian,
    so grid north is rotated ~2.05 deg from true north. Marching a true azimuth as
    if it were a grid bearing walks the sight line 180 m sideways over 5 km --
    at a 3.4 deg sun that is the difference between a gap and a wall.

        grid_bearing = true_azimuth + grid_convergence_deg(...)
    """
    from pyproj import Geod, Transformer
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = to_wgs.transform(x, y)
    lon2, lat2, _ = Geod(ellps="GRS80").fwd(lon, lat, 0.0, 1000.0)  # 1 km true north
    x2, y2 = to_grid.transform(lon2, lat2)
    return float(np.degrees(np.arctan2(x2 - x, y2 - y)))


def load_grid():
    meta = json.loads((DATA / "grid_meta.json").read_text())
    dsm = np.load(DATA / "dsm2m.npy")
    dtm = np.load(DATA / "dtm2m.npy")
    return dsm, dtm, meta


# Multi-resolution schedule: (pool factor, from_dist, to_dist).
# Marching far away with a coarse step is what makes this tractable, but a coarse
# step on the raw grid steps OVER narrow obstructions -- a mast or a treeline seen
# edge-on simply vanishes. So each band samples a MAX-POOLED copy whose cell size
# equals its step: every cell between two samples is still represented by the
# pooled maximum, so an obstruction can never be skipped, only (slightly)
# over-stated. Over-stating blocks a marginal spot rather than falsely clearing it,
# which is the error we want.
BANDS = ((1, 0.0, 200.0), (4, 200.0, 800.0), (8, 800.0, 2400.0), (16, 2400.0, DMAX))


def maxpool_upsample(a: np.ndarray, f: int) -> np.ndarray:
    """Max-pool by f, then expand back to the original shape."""
    if f == 1:
        return a
    ny, nx = a.shape
    py, px = (-ny) % f, (-nx) % f
    if py or px:
        a = np.pad(a, ((0, py), (0, px)), mode="edge")
    h, w = a.shape
    pooled = a.reshape(h // f, f, w // f, f).max(axis=(1, 3))
    up = np.repeat(np.repeat(pooled, f, axis=0), f, axis=1)
    return up[:ny, :nx]


def build_pyramid(dsm: np.ndarray):
    """One max-pooled level per distance band, reused for every timestamp."""
    levels = []
    for f, d0, d1 in BANDS:
        levels.append((maxpool_upsample(dsm, f), f, d0, d1))
    return levels


def ray_offsets(az_deg: float, res: float, dmax: float = DMAX,
                step: float | None = None, dmin: float = 0.0):
    """Integer (di, dj) pixel offsets along the direction toward the sun.

    Duplicates from rounding are dropped, and the true distance implied by the
    integer offset is returned so nothing is lost to quantisation.
    """
    a = np.radians(az_deg)
    ue, un = np.sin(a), np.cos(a)  # east, north components
    step = step or res
    ds = np.arange(max(dmin, step), dmax, step)
    dj = np.rint(ds * ue / res).astype(np.int32)
    di = np.rint(ds * un / res).astype(np.int32)
    keep = np.concatenate([[True], (np.diff(di) != 0) | (np.diff(dj) != 0)])
    di, dj = di[keep], dj[keep]
    dist = np.hypot(di.astype(np.float64) * res, dj.astype(np.float64) * res)
    ok = dist > 0
    return di[ok], dj[ok], dist[ok]


def shifted_view(src: np.ndarray, di: int, dj: int):
    """Views (dst_slice, src_slice) such that dst[s] = src[s shifted by (di,dj)]."""
    ny, nx = src.shape
    i0d, i1d = max(0, -di), min(ny, ny - di)
    j0d, j1d = max(0, -dj), min(nx, nx - dj)
    if i0d >= i1d or j0d >= j1d:
        return None, None
    i0s, i1s = i0d + di, i1d + di
    j0s, j1s = j0d + dj, j1d + dj
    return (slice(i0d, i1d), slice(j0d, j1d)), (slice(i0s, i1s), slice(j0s, j1s))


def shadow_reference(dsm, az_deg, alt_deg, res, dmax=DMAX, pyramid=None):
    """Per-cell threshold height S: an observer at p sees the sun iff eye(p) >= S(p).

        shadowed(p)  <=>  exists d: DSM(p+d.u) - drop(d) - eye(p) > d*tan(alt)
                     <=>  max_d [ DSM(p+d.u) - drop(d) - d*tan(alt) ] > eye(p)

    The bracket does not mention the observer, so a single march answers the
    question for ANY height at p: standing on the ground (DTM+1.7), standing on
    the surface itself (DSM, which is what the 3-D view must shade), or on a
    balcony 20 m up. That is why this returns S instead of a boolean.
    """
    if pyramid is None:
        pyramid = build_pyramid(dsm)
    tan_alt = np.float32(np.tan(np.radians(alt_deg)))
    best = np.full(dsm.shape, -np.inf, dtype=np.float32)

    for level, f, d0, d1 in pyramid:
        if d0 >= dmax:
            break
        di, dj, dist = ray_offsets(az_deg, res, min(d1, dmax), step=f * res, dmin=d0)
        for k in range(len(di)):
            dslice, sslice = shifted_view(level, int(di[k]), int(dj[k]))
            if dslice is None:
                continue
            d = np.float32(dist[k])
            bias = np.float32(d * d / (2.0 * R_EFF) + dist[k] * tan_alt)
            np.maximum(best[dslice], level[sslice] - bias, out=best[dslice])
    return best


def horizon_tangent(dsm, eye_z, az_deg, res, dmax=DMAX, pyramid=None):
    """Max tan(elevation angle) of any obstruction toward `az_deg`, per cell.

    Cells whose sight line leaves the grid keep whatever they accumulated; use
    `valid_mask` to know where that is a real answer.
    """
    if pyramid is None:
        pyramid = build_pyramid(dsm)
    best = np.full(dsm.shape, -np.inf, dtype=np.float32)

    for level, f, d0, d1 in pyramid:
        if d0 >= dmax:
            break
        step = f * res
        di, dj, dist = ray_offsets(az_deg, res, min(d1, dmax), step=step, dmin=d0)
        drop = (dist ** 2) / (2.0 * R_EFF)  # earth curvature
        for k in range(len(di)):
            dslice, sslice = shifted_view(level, int(di[k]), int(dj[k]))
            if dslice is None:
                continue
            # apparent height of the obstruction above the observer's eye
            rel = level[sslice] - np.float32(drop[k]) - eye_z[dslice]
            np.maximum(best[dslice], rel * np.float32(1.0 / dist[k]), out=best[dslice])
    return best


def valid_mask(shape, az_deg, res, dmax=DMAX):
    """True where a full-length sight line stays inside the grid."""
    a = np.radians(az_deg)
    ny, nx = shape
    dj = int(np.rint(dmax * np.sin(a) / res))
    di = int(np.rint(dmax * np.cos(a) / res))
    m = np.zeros(shape, dtype=bool)
    i0, i1 = max(0, -di), min(ny, ny - di)
    j0, j1 = max(0, -dj), min(nx, nx - dj)
    if i0 < i1 and j0 < j1:
        m[i0:i1, j0:j1] = True
    return m


CEST = _dt.timezone(_dt.timedelta(hours=2))
_G = {}  # fork-shared state for the worker pool


def _worker(job):
    idx, az, alt = job
    s = shadow_reference(_G["dsm"], az + _G["conv"], alt, _G["res"], pyramid=_G["pyr"])
    m = _G["meta"]
    i0, i1, j0, j1 = m["oi0"], m["oi1"], m["oj0"], m["oj1"]
    sub = s[i0:i1, j0:j1]
    ground = _G["eye"][i0:i1, j0:j1] >= sub    # person standing on the ground
    surface = _G["dsm"][i0:i1, j0:j1] >= sub   # the surface itself, for 3-D shading
    return (idx, np.packbits(ground, axis=None), np.packbits(surface, axis=None),
            float(ground.mean()))


def main():
    import multiprocessing as mp

    from eclipse import circumstances
    from solar import LEIPZIG_LAT, LEIPZIG_LON, sun_position

    dsm, dtm, meta = load_grid()
    res = meta["res"]
    print(f"grid {dsm.shape} @ {res} m")

    # Report only over the real Leipzig study area; the rest is ray-march context.
    oj0 = int((meta["out_minx"] - meta["minx"]) / res)
    oj1 = int((meta["out_maxx"] - meta["minx"]) / res)
    oi0 = int((meta["out_miny"] - meta["miny"]) / res)
    oi1 = int((meta["out_maxy"] - meta["miny"]) / res)
    meta.update(oi0=oi0, oi1=oi1, oj0=oj0, oj1=oj1)
    print(f"output region {oi1-oi0} x {oj1-oj0} cells")

    times = [_dt.datetime(2026, 8, 12, 19, 20, tzinfo=CEST) + _dt.timedelta(minutes=5 * k)
             for k in range(17)]
    sp = sun_position(times, LEIPZIG_LAT, LEIPZIG_LON)

    # Obscuration at each timestamp, from the ephemeris eclipse model.
    circ = circumstances()
    by_min = {t.astimezone(CEST).strftime("%H:%M"): o for t, o, *_ in circ["samples"]}
    obsc = np.array([by_min.get(t.strftime("%H:%M"), 0.0) for t in times], dtype=np.float32)

    conv = grid_convergence_deg((meta["out_minx"] + meta["out_maxx"]) / 2,
                                (meta["out_miny"] + meta["out_maxy"]) / 2)
    print(f"grid convergence at study-area centre: {conv:+.3f} deg "
          f"(true azimuth -> grid bearing)")

    print("building max-pool pyramid ...", end="", flush=True)
    _G.update(dsm=dsm, eye=(dtm + EYE_H).astype(np.float32), res=res, meta=meta,
              conv=conv, pyr=build_pyramid(dsm))
    print(" done")

    jobs = [(k, float(sp["azimuth"][k]), float(sp["apparent_altitude"][k]))
            for k in range(len(times))]
    ground, surface = {}, {}
    with mp.get_context("fork").Pool(5) as pool:
        for idx, g, s, frac in pool.imap_unordered(_worker, jobs):
            ground[idx], surface[idx] = g, s
            print(f"  {times[idx]:%H:%M}  az {jobs[idx][1]:6.2f}  alt {jobs[idx][2]:5.2f}  "
                  f"obsc {obsc[idx]*100:4.1f}%  ground sunlit {frac*100:5.1f}%")

    ny_o, nx_o = oi1 - oi0, oj1 - oj0
    np.savez_compressed(
        DATA / "sunlit.npz",
        packed=np.stack([ground[k] for k in range(len(times))]),
        packed_surface=np.stack([surface[k] for k in range(len(times))]),
        shape=np.array([ny_o, nx_o]),
        times=np.array([t.strftime("%H:%M") for t in times]),
        az=sp["azimuth"].astype(np.float32),
        alt=sp["apparent_altitude"].astype(np.float32),
        obsc=obsc,
        extent=np.array([meta["out_minx"], meta["out_maxx"],
                         meta["out_miny"], meta["out_maxy"]], dtype=np.int64),
        res=np.float32(res),
    )
    print(f"saved {len(ground)} ground + surface masks for {ny_o}x{nx_o} grid")


if __name__ == "__main__":
    main()
