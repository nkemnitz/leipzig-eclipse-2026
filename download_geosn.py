"""Download open GeoSN (Saxony) raster tiles for the Leipzig study area.

Products used:
  DOM1 - Digitales Oberflaechenmodell, 1 m laser-scanned SURFACE model
         (terrain + buildings + tree canopy). This is what actually blocks the sun.
  DGM1 - Digitales Gelaendemodell, 1 m bare-earth TERRAIN model.
         DOM1 - DGM1 gives true object heights.

License: Datenlizenz Deutschland Namensnennung 2.0 (dl-de/by-2-0), source: GeoSN.
Free, no account. Tiles are 2 km x 2 km, ETRS89/UTM33N (EPSG:25833), heights DHHN2016.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import requests

BASE = "https://geocloud.landesvermessung.sachsen.de/index.php/s/{share}/download?path=%2F&files={fn}"
SHARES = {"dom1": "S6wwnFwX7882sZm", "dgm1": "JCcXyifaNdLDnxZ"}
FILENAME = "{prod}_33{e:03d}_{n:04d}_2_sn_tiff.zip"

# Study area, 2 km tile grid (SW corner km, UTM33N).
# Results are REPORTED for 308-324 km E x 5684-5696 km N (16 x 12 km centred on
# Leipzig: Kulkwitzer See W, Cospudener See S, Voelkerschlachtdenkmal SE, Rosental N).
# The grid is extended west to 302 km and north to 5698 km purely as ray-march
# context: at maximum eclipse the sun sits at ~3.3 deg altitude on azimuth ~290 deg,
# so sight lines run up to ~5 km WNW and must not fall off the edge of the data.
E_KM = range(302, 324, 2)
N_KM = range(5676, 5698, 2)

OUT = Path("data/geosn")
HEADERS = {"User-Agent": "Mozilla/5.0 (eclipse-viewshed-study)"}


def fetch(prod: str, e: int, n: int) -> tuple[str, int, str]:
    fn = FILENAME.format(prod=prod, e=e, n=n)
    dest = OUT / prod / fn
    if dest.exists() and dest.stat().st_size > 10000:
        return fn, dest.stat().st_size, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(share=SHARES[prod], fn=fn)
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            if r.status_code != 200:
                return fn, 0, f"HTTP {r.status_code}"
            tmp = dest.with_suffix(".part")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    size += len(chunk)
            if size < 10000:
                tmp.unlink(missing_ok=True)
                return fn, size, "too small"
            tmp.rename(dest)
            return fn, size, "ok"
    except Exception as exc:
        return fn, 0, f"error {type(exc).__name__}"


def main():
    prods = sys.argv[1:] or ["dom1", "dgm1"]
    jobs = [(p, e, n) for p in prods for e in E_KM for n in N_KM]
    print(f"{len(jobs)} tiles ({len(prods)} products x {len(E_KM)}x{len(N_KM)} grid)")

    total = 0
    fails = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            fn, size, status = fut.result()
            total += size
            if status not in ("ok", "cached"):
                fails.append((fn, status))
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i:3d}/{len(jobs)}] {total/1e6:7.1f} MB  last={fn} {status}")

    print(f"\ndone: {total/1e6:.1f} MB, {len(fails)} failures")
    for fn, st in fails[:20]:
        print(f"  FAIL {fn}: {st}")
    (OUT / "manifest.json").write_text(json.dumps({
        "products": prods, "e_km": list(E_KM), "n_km": list(N_KM),
        "tile_km": 2, "crs": "EPSG:25833", "vertical": "DHHN2016",
        "license": "dl-de/by-2-0", "attribution": "GeoSN",
        "failures": fails,
    }, indent=2))


if __name__ == "__main__":
    main()
