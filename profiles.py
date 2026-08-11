"""Skyline profile at a single point: obstruction elevation angle vs azimuth.

This is the view from one spot -- the shape of the horizon you would actually see,
which can be plotted directly against the sun's track. Cheap enough to run per
point (unlike the raster march), so it is done at full 1 m... 2 m grid resolution
with fine distance steps.
"""

from __future__ import annotations

import numpy as np

R_EFF = 7.0 / 6.0 * 6371000.0
EYE_H = 1.7


def profile(dsm, meta, x, y, az_from=180.0, az_to=360.0, n_az=181,
            dmax=8000.0, eye_h=EYE_H, ground_z=None, convergence=None):
    """Return (true_azimuths, elevation_deg) of the skyline seen from (x, y).

    Azimuths in and out are TRUE azimuths so the result can be plotted directly
    against the sun's track; `convergence` rotates them into UTM grid bearings for
    the actual marching (see viewshed.grid_convergence_deg).
    """
    res = meta["res"]
    minx, miny = meta["minx"], meta["miny"]
    ny, nx = dsm.shape

    j = (x - minx) / res
    i = (y - miny) / res
    if not (0 <= i < ny and 0 <= j < nx):
        raise ValueError("point outside grid")
    if ground_z is None:
        ground_z = float(dsm[int(i), int(j)])
    eye = ground_z + eye_h

    # Distance samples: fine near, coarse far -- angular resolution is what matters.
    d = np.concatenate([
        np.arange(res, 100.0, res),
        np.arange(100.0, 500.0, res * 2),
        np.arange(500.0, 2000.0, res * 4),
        np.arange(2000.0, dmax, res * 10),
    ])
    drop = d * d / (2.0 * R_EFF)

    if convergence is None:
        from viewshed import grid_convergence_deg
        convergence = grid_convergence_deg(x, y)

    az = np.linspace(az_from, az_to, n_az)
    out = np.full(n_az, -90.0)
    for k, a in enumerate(az):
        ar = np.radians(a + convergence)   # true azimuth -> grid bearing
        jj = np.rint(j + d * np.sin(ar) / res).astype(np.int64)
        ii = np.rint(i + d * np.cos(ar) / res).astype(np.int64)
        ok = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        if not ok.any():
            continue
        h = dsm[ii[ok], jj[ok]]
        ang = np.degrees(np.arctan((h - drop[ok] - eye) / d[ok]))
        out[k] = float(ang.max())
    return az, out


def sun_track(times, lat, lon):
    """Sun azimuth/altitude track, for overlaying on a skyline profile."""
    from solar import sun_position
    sp = sun_position(times, lat, lon)
    return sp["azimuth"], sp["apparent_altitude"]
