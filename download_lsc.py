"""Download GeoSN Laserscandaten (LSC) point-cloud tiles.

Only the near-field is needed as voxels: at a 3.4 deg sun, vegetation stops
mattering past roughly 1.5 km (a 25 m crown blocks from 420 m; beyond ~1.5 km only
buildings and terrain are tall enough, and the DSM march handles those exactly).
So the voxel model covers the reported area plus a 1.5-2 km margin, not the whole
ray-march context -- 90 tiles instead of 121.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from pathlib import Path

import requests

BASE = "https://geocloud.landesvermessung.sachsen.de/index.php/s/{share}/download?path=%2F&files={fn}"
SHARE = "EpkzyJHScGb5ndd"
FILENAME = "lsc_33{e:03d}_{n:04d}_2_sn_laz.zip"

E_KM = range(306, 324, 2)        # 9 columns  (reported area 308-324 + 2 km west)
N_KM = range(5678, 5698, 2)      # 10 rows    (reported area 5678-5696 + 2 km north)

OUT = Path("data/lsc")
HEADERS = {"User-Agent": "Mozilla/5.0 (eclipse-viewshed-study)"}


def fetch(e: int, n: int):
    fn = FILENAME.format(e=e, n=n)
    dest = OUT / fn
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return fn, dest.stat().st_size, "cached"
    try:
        with requests.get(BASE.format(share=SHARE, fn=fn), headers=HEADERS,
                          stream=True, timeout=900) as r:
            if r.status_code != 200:
                return fn, 0, f"HTTP {r.status_code}"
            tmp = dest.with_suffix(".part")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(4 << 20):
                    f.write(chunk)
                    size += len(chunk)
            if size < 1_000_000:
                tmp.unlink(missing_ok=True)
                return fn, size, "too small"
            tmp.rename(dest)
            return fn, size, "ok"
    except Exception as exc:
        return fn, 0, f"error {type(exc).__name__}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(e, n) for e in E_KM for n in N_KM]
    print(f"{len(jobs)} LSC tiles (~{len(jobs)*0.29:.0f} GB)")
    total, fails = 0, []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            fn, size, status = fut.result()
            total += size
            if status not in ("ok", "cached"):
                fails.append((fn, status))
            if i % 5 == 0 or i == len(jobs):
                print(f"  [{i:3d}/{len(jobs)}] {total/1e9:6.2f} GB  last={fn} {status}",
                      flush=True)
    print(f"\ndone: {total/1e9:.2f} GB, {len(fails)} failures")
    for fn, st in fails[:20]:
        print(f"  FAIL {fn}: {st}")
    (OUT / "manifest.json").write_text(json.dumps({
        "e_km": list(E_KM), "n_km": list(N_KM), "tile_km": 2,
        "crs": "EPSG:25833", "license": "dl-de/by-2-0", "attribution": "GeoSN",
        "failures": fails}, indent=2))


if __name__ == "__main__":
    main()
