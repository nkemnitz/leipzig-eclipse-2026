"""Correctness checks for the ray-march geometry, on synthetic scenes."""

import numpy as np

from viewshed import EYE_H, R_EFF, horizon_tangent, ray_offsets

RES = 2.0
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} got {got:9.5f}  want {want:9.5f}")
    if not ok:
        FAIL.append(name)


def flat(ny, nx, h=0.0):
    return np.full((ny, nx), h, dtype=np.float32)


print("1. Wall 100 m WEST of observer, sun due west (az 270)")
dsm = flat(21, 201)
dsm[:, 50] = 10.0                       # wall at col 50
eye = flat(21, 201) + EYE_H
ht = horizon_tangent(dsm, eye, 270.0, RES)
d = 100.0
want = (10.0 - d * d / (2 * R_EFF) - EYE_H) / d
check("blocking wall to the west", float(ht[10, 100]), want, 1e-4)

print("2. Same wall, but sun due EAST (az 90) -- must not block")
# Over open ground the horizon angle is slightly NEGATIVE: you look marginally
# down at ground receding under the curve. The max over d of (-eye - d^2/2R)/d is
# attained at the far end of the available grid (100 cells = 200 m here).
ht_e = horizon_tangent(dsm, eye, 90.0, RES)
d_end = 200.0
check("wall behind observer is irrelevant",
      float(ht_e[10, 100]), (-EYE_H - d_end**2 / (2 * R_EFF)) / d_end, 1e-4)

print("3. Sun due north (az 0): wall to the north blocks")
dsm2 = flat(201, 21)
dsm2[150, :] = 10.0                     # wall 100 m north of row 100
eye2 = flat(201, 21) + EYE_H
ht_n = horizon_tangent(dsm2, eye2, 0.0, RES)
check("blocking wall to the north", float(ht_n[100, 10]), want, 1e-4)

print("4. Azimuth 279.277 deg (eclipse C1) resolves E/N components correctly")
di, dj, dist = ray_offsets(279.277, RES, dmax=1000.0)
k = np.argmin(np.abs(dist - 500.0))
east = dj[k] * RES
north = di[k] * RES
check("east component at ~500 m  (sin az)", east, 500.0 * np.sin(np.radians(279.277)), 3.0)
check("north component at ~500 m (cos az)", north, 500.0 * np.cos(np.radians(279.277)), 3.0)

print("5. Observer standing ON a 30 m roof sees over a 10 m wall")
dsm3 = flat(21, 201)
dsm3[:, 50] = 10.0
dsm3[:, 100] = 30.0
eye3 = flat(21, 201) + EYE_H
eye3[:, 100] = 30.0 + EYE_H             # observer on the roof
ht_r = horizon_tangent(dsm3, eye3, 270.0, RES)
print(f"  {'PASS' if ht_r[10, 100] < 0 else 'FAIL'}  elevated observer clears the wall"
      f"{'':22s} tan={float(ht_r[10, 100]):+.4f} (<0)")
if not ht_r[10, 100] < 0:
    FAIL.append("elevated observer")

print("6. Narrow distant target must NOT be stepped over (max-pool pyramid)")
dsm4 = flat(21, 3001)
dsm4[:, 500] = 50.0                     # single 2 m wide wall, 4000 m west of col 2500
eye4 = flat(21, 3001) + EYE_H
ht_c = horizon_tangent(dsm4, eye4, 270.0, RES, dmax=5000.0)
d4 = 4000.0
want4 = (50.0 - d4 * d4 / (2 * R_EFF) - EYE_H) / d4
# Pooled sampling may place the wall up to one 32 m cell nearer/farther.
check("narrow 4 km wall is still seen", float(ht_c[10, 2500]), want4, 2e-4)
drop = d4 * d4 / (2 * R_EFF)
print(f"        (curvature drop at 4 km = {drop:.2f} m, i.e. {drop/EYE_H:.1f}x eye height)")

print("7. Sunlit test reproduces a known geometry")
# 20 m obstruction, sun at 3.3 deg -> blocks out to 20/tan(3.3) = 347 m
for dist_m, expect in ((300.0, False), (400.0, True)):
    n = int(dist_m / RES)
    dsm5 = flat(9, n + 60)
    dsm5[:, 20] = 20.0
    eye5 = flat(9, n + 60) + EYE_H
    h = horizon_tangent(dsm5, eye5, 270.0, RES, dmax=2000.0)
    lit = bool(h[4, 20 + n] < np.tan(np.radians(3.3)))
    print(f"  {'PASS' if lit == expect else 'FAIL'}  20 m wall at {dist_m:.0f} m, sun 3.3 deg "
          f"-> {'sunlit' if lit else 'shadowed'} (expect {'sunlit' if expect else 'shadowed'})")
    if lit != expect:
        FAIL.append(f"20m wall @ {dist_m}")

print("8. Meridian convergence: true azimuth is NOT a UTM grid bearing")
from viewshed import grid_convergence_deg
from pyproj import Geod, Transformer
_g = Geod(ellps="GRS80")
_tw = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
_tg = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
for px, py_ in ((316000.0, 5690000.0), (310000.0, 5682000.0)):
    conv = grid_convergence_deg(px, py_)
    lon, lat = _tw.transform(px, py_)
    true_az = 289.862
    lon2, lat2, _ = _g.fwd(lon, lat, true_az, 5000.0)
    x2, y2 = _tg.transform(lon2, lat2)
    measured = np.degrees(np.arctan2(x2 - px, y2 - py_)) % 360
    check(f"grid bearing at E{px:.0f}", true_az + conv, measured, 0.02)
    lat_err = 5000 * abs(np.sin(np.radians(conv)))
    print(f"        (convergence {conv:+.3f} deg = {lat_err:.0f} m sideways at 5 km)")

print()
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
