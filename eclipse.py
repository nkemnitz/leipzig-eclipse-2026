"""Local circumstances of the 2026-08-12 solar eclipse for a given site.

Computed from topocentric Sun/Moon ephemeris (pyephem/libastro), not scraped:
obscuration is the true circle-circle overlap area ratio, magnitude is the
fraction of the solar diameter covered.
"""

from __future__ import annotations

import datetime as _dt
import math

import ephem
import numpy as np

from solar import LEIPZIG_LAT, LEIPZIG_LON, LEIPZIG_ELEV_M, sun_position

CEST = _dt.timezone(_dt.timedelta(hours=2))


def _sep_and_radii(t_utc, lat, lon, elev):
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), elev
    obs.date = t_utc.replace(tzinfo=None)
    obs.pressure = 0  # geometric positions; refraction handled separately
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)
    sep = float(ephem.separation((sun.az, sun.alt), (moon.az, moon.alt)))
    # ephem .size is apparent diameter in arcseconds
    rs = math.radians(sun.size / 3600.0 / 2.0)
    rm = math.radians(moon.size / 3600.0 / 2.0)
    return sep, rs, rm, math.degrees(float(sun.alt)), math.degrees(float(sun.az))


def obscuration(sep, rs, rm):
    """Fraction of the solar disk area covered by the moon."""
    if sep >= rs + rm:
        return 0.0
    if sep <= abs(rm - rs):
        return 1.0 if rm >= rs else (rm / rs) ** 2
    d, r, R = sep, min(rs, rm), max(rs, rm)
    a1 = r * r * math.acos((d * d + r * r - R * R) / (2 * d * r))
    a2 = R * R * math.acos((d * d + R * R - r * r) / (2 * d * R))
    a3 = 0.5 * math.sqrt(max(0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    return (a1 + a2 - a3) / (math.pi * rs * rs)


def magnitude(sep, rs, rm):
    return max(0.0, min((rs + rm - sep) / (2 * rs), 1.0))


def circumstances(lat=LEIPZIG_LAT, lon=LEIPZIG_LON, elev=LEIPZIG_ELEV_M,
                  date=_dt.date(2026, 8, 12)):
    t0 = _dt.datetime(date.year, date.month, date.day, 14, 0, tzinfo=_dt.timezone.utc)
    samples = []
    for k in range(0, 8 * 60 + 1):  # 14:00 -> 22:00 UTC, 1 min
        t = t0 + _dt.timedelta(minutes=k)
        sep, rs, rm, alt, az = _sep_and_radii(t, lat, lon, elev)
        samples.append((t, obscuration(sep, rs, rm), magnitude(sep, rs, rm), alt, az))

    obs_arr = np.array([s[1] for s in samples])
    imax = int(obs_arr.argmax())

    def refine(lo, hi, target_positive):
        for _ in range(50):
            mid = lo + (hi - lo) / 2
            sep, rs, rm, _, _ = _sep_and_radii(mid, lat, lon, elev)
            o = obscuration(sep, rs, rm)
            if (o > 0) == target_positive:
                hi = mid
            else:
                lo = mid
            if (hi - lo).total_seconds() < 0.5:
                break
        return lo + (hi - lo) / 2

    c1 = c4 = None
    for i in range(len(samples) - 1):
        if obs_arr[i] == 0 and obs_arr[i + 1] > 0:
            c1 = refine(samples[i][0], samples[i + 1][0], True)
        if obs_arr[i] > 0 and obs_arr[i + 1] == 0:
            c4 = refine(samples[i + 1][0], samples[i][0], True)

    # Refine maximum by golden-section on obscuration
    lo, hi = samples[max(imax - 2, 0)][0], samples[min(imax + 2, len(samples) - 1)][0]
    for _ in range(60):
        a = lo + (hi - lo) * 0.382
        b = lo + (hi - lo) * 0.618
        sa = _sep_and_radii(a, lat, lon, elev)
        sb = _sep_and_radii(b, lat, lon, elev)
        if obscuration(*sa[:3]) > obscuration(*sb[:3]):
            hi = b
        else:
            lo = a
        if (hi - lo).total_seconds() < 1:
            break
    tmax = lo + (hi - lo) / 2
    sep, rs, rm, _, _ = _sep_and_radii(tmax, lat, lon, elev)

    return {
        "c1": c1, "max": tmax, "c4": c4,
        "obscuration": obscuration(sep, rs, rm),
        "magnitude": magnitude(sep, rs, rm),
        "samples": samples,
    }


def fmt(t):
    if t is None:
        return "n/a"
    return f"{t.astimezone(CEST):%H:%M:%S} CEST ({t:%H:%M:%S} UTC)"


if __name__ == "__main__":
    from solar import sunrise_sunset

    r = circumstances()
    _, sset = sunrise_sunset(_dt.date(2026, 8, 12), LEIPZIG_LAT, LEIPZIG_LON)
    sset = sset.replace(tzinfo=_dt.timezone.utc)

    print("Solar eclipse 2026-08-12, Leipzig (51.3397 N, 12.3731 E, 113 m)\n")
    for lbl, t in (("C1 first contact", r["c1"]), ("MAXIMUM", r["max"]), ("C4 last contact", r["c4"])):
        if t is None:
            print(f"  {lbl:18s} n/a")
            continue
        s = sun_position([t], LEIPZIG_LAT, LEIPZIG_LON)
        print(f"  {lbl:18s} {fmt(t)}   sun az {s['azimuth'][0]:6.2f}  alt {s['apparent_altitude'][0]:5.2f}")
    print(f"  {'SUNSET':18s} {fmt(sset)}")
    print(f"\n  max obscuration  {r['obscuration']*100:.1f} %  (magnitude {r['magnitude']:.3f})")
    if r["c4"] and r["c4"] > sset:
        print("  NOTE: the sun sets while still eclipsed")

    print("\n  time     obsc%   az     alt")
    for t, o, m, alt, az in r["samples"]:
        if t.astimezone(CEST).minute % 15 == 0 and o > 0:
            s = sun_position([t], LEIPZIG_LAT, LEIPZIG_LON)
            print(f"  {t.astimezone(CEST):%H:%M}  {o*100:6.1f}  {s['azimuth'][0]:6.2f}  {s['apparent_altitude'][0]:5.2f}")
