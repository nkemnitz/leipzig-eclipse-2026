"""Solar position (NOAA / Meeus) in pure numpy.

Azimuth convention: degrees clockwise from true north (0=N, 90=E, 180=S, 270=W).
Altitude: degrees above the true horizon. `apparent` altitude includes atmospheric
refraction, which is worth ~0.09 deg at 10 deg altitude and ~0.48 deg at the horizon --
non-negligible when the whole question is whether a rooftop clears the sun.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

# Leipzig, Markt (city centre)
LEIPZIG_LAT = 51.3397
LEIPZIG_LON = 12.3731
LEIPZIG_ELEV_M = 113.0


def to_julian_day(times_utc) -> np.ndarray:
    """UTC datetimes -> Julian Day (float). Accepts a scalar or an iterable."""
    if isinstance(times_utc, _dt.datetime):
        times_utc = [times_utc]
    out = []
    for t in times_utc:
        if t.tzinfo is not None:
            t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        y, m = t.year, t.month
        day = (
            t.day
            + (t.hour + (t.minute + (t.second + t.microsecond / 1e6) / 60.0) / 60.0) / 24.0
        )
        if m <= 2:
            y -= 1
            m += 12
        a = y // 100
        b = 2 - a + a // 4  # Gregorian calendar
        jd = (
            np.floor(365.25 * (y + 4716))
            + np.floor(30.6001 * (m + 1))
            + day
            + b
            - 1524.5
        )
        out.append(jd)
    return np.asarray(out, dtype=float)


def _refraction_deg(elev_deg: np.ndarray) -> np.ndarray:
    """NOAA atmospheric refraction correction, in degrees. Input: true elevation."""
    e = np.asarray(elev_deg, dtype=float)
    te = np.tan(np.radians(np.clip(e, -5.0, 90.0)))
    # arcseconds
    r = np.where(
        e > 85.0,
        0.0,
        np.where(
            e > 5.0,
            58.1 / te - 0.07 / te**3 + 0.000086 / te**5,
            np.where(
                e > -0.575,
                1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))),
                -20.772 / te,
            ),
        ),
    )
    return r / 3600.0


def sun_position(times_utc, lat_deg: float, lon_deg: float):
    """Return dict with azimuth, altitude (true), apparent_altitude, declination, eot.

    lon_deg is EAST-positive.
    """
    jd = to_julian_day(times_utc)
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude / anomaly of the sun
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    mr = np.radians(m)
    c = (
        np.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mr) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mr) * 0.000289
    )
    true_long = l0 + c

    # Apparent longitude (nutation + aberration)
    omega = 125.04 - 1934.136 * t
    lam = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    # Obliquity of the ecliptic
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))

    epsr, lamr = np.radians(eps), np.radians(lam)
    decl = np.degrees(np.arcsin(np.sin(epsr) * np.sin(lamr)))

    # Equation of time (minutes)
    y = np.tan(epsr / 2.0) ** 2
    l0r = np.radians(l0)
    eot = 4.0 * np.degrees(
        y * np.sin(2 * l0r)
        - 2 * e * np.sin(mr)
        + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r)
        - 1.25 * e * e * np.sin(2 * mr)
    )

    # Hour angle. Minutes-of-day in UTC, derived from the JD itself to stay consistent.
    minutes_utc = (jd + 0.5 - np.floor(jd + 0.5)) * 1440.0
    true_solar_time = (minutes_utc + eot + 4.0 * lon_deg) % 1440.0
    ha = true_solar_time / 4.0 - 180.0
    ha = np.where(ha < -180.0, ha + 360.0, ha)

    latr, declr, har = np.radians(lat_deg), np.radians(decl), np.radians(ha)
    cos_zen = np.clip(
        np.sin(latr) * np.sin(declr) + np.cos(latr) * np.cos(declr) * np.cos(har), -1.0, 1.0
    )
    zenith = np.degrees(np.arccos(cos_zen))
    elev = 90.0 - zenith

    # Azimuth, clockwise from north
    sin_zen = np.sin(np.radians(zenith))
    denom = np.cos(latr) * sin_zen
    az_arg = np.where(
        np.abs(denom) < 1e-12,
        0.0,
        np.clip((np.sin(latr) * cos_zen - np.sin(declr)) / np.where(np.abs(denom) < 1e-12, 1.0, denom), -1.0, 1.0),
    )
    az = np.degrees(np.arccos(az_arg))
    az = np.where(ha > 0.0, (az + 180.0) % 360.0, (540.0 - az) % 360.0)

    return {
        "julian_day": jd,
        "azimuth": az,
        "altitude": elev,
        "apparent_altitude": elev + _refraction_deg(elev),
        "declination": decl,
        "eot_minutes": eot,
        "hour_angle": ha,
    }


def sun_vector(az_deg, alt_deg):
    """Unit vector pointing FROM the observer TOWARD the sun, in a local ENU frame
    (x=east, y=north, z=up)."""
    a, h = np.radians(az_deg), np.radians(alt_deg)
    return np.stack(
        [np.cos(h) * np.sin(a), np.cos(h) * np.cos(a), np.sin(h)], axis=-1
    )


def _bisect_altitude(target, lo, hi, lat, lon, rising, tol_s=0.5):
    """Find the UTC time between lo and hi where apparent altitude crosses `target`."""
    for _ in range(60):
        mid = lo + (hi - lo) / 2
        a = sun_position([mid], lat, lon)["apparent_altitude"][0]
        above = a > target
        if above == rising:
            hi = mid
        else:
            lo = mid
        if (hi - lo).total_seconds() < tol_s:
            break
    return lo + (hi - lo) / 2


def sunrise_sunset(date, lat_deg, lon_deg, horizon=-0.833):
    """Sunrise/sunset UTC for a date (upper limb, standard -0.833 deg horizon)."""
    day = _dt.datetime(date.year, date.month, date.day)
    times = [day + _dt.timedelta(minutes=i) for i in range(0, 1441, 10)]
    alt = sun_position(times, lat_deg, lon_deg)["apparent_altitude"]
    above = alt > horizon
    rise = sset = None
    for i in range(len(times) - 1):
        if not above[i] and above[i + 1]:
            rise = _bisect_altitude(horizon, times[i], times[i + 1], lat_deg, lon_deg, True)
        if above[i] and not above[i + 1]:
            sset = _bisect_altitude(horizon, times[i], times[i + 1], lat_deg, lon_deg, False)
    return rise, sset


if __name__ == "__main__":
    CEST = _dt.timezone(_dt.timedelta(hours=2))
    peak = _dt.datetime(2026, 8, 12, 19, 15, tzinfo=CEST)
    r = sun_position([peak], LEIPZIG_LAT, LEIPZIG_LON)
    print(f"Leipzig {peak:%Y-%m-%d %H:%M %Z}")
    print(f"  azimuth            {r['azimuth'][0]:8.3f} deg")
    print(f"  altitude (true)    {r['altitude'][0]:8.3f} deg")
    print(f"  altitude (apparent){r['apparent_altitude'][0]:8.3f} deg")
    rise, sset = sunrise_sunset(peak.date(), LEIPZIG_LAT, LEIPZIG_LON)
    print(f"  sunrise {rise.replace(tzinfo=_dt.timezone.utc).astimezone(CEST):%H:%M:%S}")
    print(f"  sunset  {sset.replace(tzinfo=_dt.timezone.utc).astimezone(CEST):%H:%M:%S}")
