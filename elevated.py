"""Public viewpoints that are NOT at ground level.

The raster answer assumes you stand on the ground at 1.7 m. Leipzig has a handful
of publicly accessible high platforms, and 100 m of height buys an enormous amount
at a 3.4 deg sun -- it is the one way to beat the skyline from inside the centre.
Evaluated with the same skyline march, just started higher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

import profiles
from solar import LEIPZIG_LAT, LEIPZIG_LON, sun_position

DATA = Path("data")
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

# (name, lat, lon, height of the platform above local ground in metres, note)
VIEWPOINTS = [
    ("Panorama Tower (City-Hochhaus)", 51.33862, 12.38137, 111.0,
     "29th-floor viewing platform / restaurant, paid lift, opens to ~23:00"),
    ("Neues Rathaus tower", 51.33800, 12.37000, 90.0,
     "guided tower tours only - check opening times for the evening"),
    ("Völkerschlachtdenkmal platform", 51.31230, 12.41330, 91.0,
     "top viewing platform, 500 steps, normally closes 18:00 - verify"),
    ("Fockeberg summit", 51.31712, 12.36222, 1.7,
     "free, always open, walk up in 10 min - ground level reference"),
    ("Rosentalhügel tower", 51.35896, 12.35264, 20.0,
     "Aussichtsturm Rosental, free, ~20 m steel tower"),
]


def main():
    gmeta = json.loads((DATA / "grid_meta.json").read_text())
    dsm = np.load(DATA / "dsm2m.npy", mmap_mode="r")
    dtm = np.load(DATA / "dtm2m.npy", mmap_mode="r")
    gm = {"res": gmeta["res"], "minx": gmeta["minx"], "miny": gmeta["miny"]}

    z = np.load(DATA / "sunlit.npz")
    times = [str(t) for t in z["times"]]
    sp = {t: (float(a), float(e)) for t, a, e in zip(times, z["az"], z["alt"])}

    print(f"{'viewpoint':34s} {'platform':>9s} {'skyline':>8s} {'margin':>8s} "
          f"{'visible':>9s}")
    print(f"{'':34s} {'m ASL':>9s} {'@290deg':>8s} {'@20:10':>8s} {'until':>9s}")
    rows = []
    for name, lat, lon, up, note in VIEWPOINTS:
        x, y = to_utm.transform(lon, lat)
        i = int((y - gmeta["miny"]) / gmeta["res"])
        j = int((x - gmeta["minx"]) / gmeta["res"])
        ground = float(dtm[i, j])
        az, elev = profiles.profile(dsm, gm, x, y, az_from=270.0, az_to=300.0,
                                    n_az=61, ground_z=ground + up, eye_h=0.0)

        # From a high platform the true horizon is far outside the 22 km grid
        # (46 km from 141 m up), so the march stops while the ground is still
        # dropping and reports a skyline BELOW the real horizon. Clamp to the
        # analytic spherical-earth horizon depression, -sqrt(2h/R_eff).
        rr = int(5000 / gmeta["res"])
        around = dtm[max(0, i - rr):i + rr:25, max(0, j - rr):j + rr:25]
        h_above = max(0.0, (ground + up) - float(np.median(around)))
        horizon_floor = -np.degrees(np.sqrt(2.0 * h_above / profiles.R_EFF))
        elev = np.maximum(elev, horizon_floor)

        def skyline_at(a, _az=az, _el=elev):
            return float(np.interp(a, _az, _el))

        hz = skyline_at(289.75)
        margin = sp["20:10"][1] - hz
        last = None
        for t in times:
            a, e = sp[t]
            if e > skyline_at(a):
                last = t
        rows.append((name, ground + up, hz, margin, last, note))
        print(f"{name:34s} {ground+up:9.1f} {hz:8.2f} {margin:+8.2f} {str(last):>9s}")

    print()
    for name, _, _, _, _, note in rows:
        print(f"  {name}: {note}")

    (DATA / "elevated.json").write_text(json.dumps([
        {"name": n, "asl_m": round(a, 1), "skyline_deg": round(h, 2),
         "margin_2010_deg": round(m, 2), "visible_until": l, "note": nt}
        for n, a, h, m, l, nt in rows], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
