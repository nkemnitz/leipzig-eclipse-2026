"""Bake the Leipzig Baumkataster into a compact binary for the viewer.

182k trees with a MEASURED height and crown diameter (95% carry a height), which
is far better than anything OSM has. Drawn for orientation only -- the occlusion
model uses the laser surface, which already contains these trees and everything
the cadastre misses (it covers only ~8% of the canopy area).

Layout: Float32 [x, y, height, crown_radius] per tree, in local metres.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

SRC = Path("data/baumkataster.csv")
OUT = Path("viewer/data")
MINX, MAXX = 308000, 324000
MINY, MAXY = 5678000, 5696000


def main():
    xs, ys, hs, rs = [], [], [], []
    skipped = 0
    with open(SRC, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            m = re.match(r"POINT \(([-\d.]+) ([-\d.]+)\)", row.get("geom", "") or "")
            if not m:
                continue
            x, y = float(m.group(1)), float(m.group(2))
            if not (MINX <= x < MAXX and MINY <= y < MAXY):
                continue
            try:
                h = float(row["baumhoehe"])
            except (TypeError, ValueError):
                h = 0.0
            try:
                kr = float(row["kr_durchm"])
            except (TypeError, ValueError):
                kr = 0.0
            # The cadastre contains a few absurd values (one tree claims 9002 m).
            if not (1.0 <= h <= 60.0):
                if h > 0:
                    skipped += 1
                h = 11.0                      # median of the catalogue
            if not (0.5 <= kr <= 30.0):
                kr = max(2.0, h * 0.45)
            xs.append(x); ys.append(y); hs.append(h); rs.append(kr / 2.0)

    arr = np.empty((len(xs), 4), dtype=np.float32)
    arr[:, 0] = xs; arr[:, 1] = ys; arr[:, 2] = hs; arr[:, 3] = rs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trees.bin").write_bytes(arr.tobytes())
    (OUT / "trees_meta.json").write_text(json.dumps({
        "count": len(xs), "stride": 4, "fields": ["x", "y", "height", "crown_radius"],
        "extent": [MINX, MAXX, MINY, MAXY],
        "source": "Stadt Leipzig Baumkataster (dl-de/by-2-0)",
    }))
    print(f"{len(xs):,} trees in the study area  ({skipped} implausible heights replaced)")
    print(f"  height  p50 {np.median(arr[:,2]):.1f} m  p95 {np.percentile(arr[:,2],95):.1f} m")
    print(f"  crown r p50 {np.median(arr[:,3]):.1f} m")
    print(f"  trees.bin {(OUT/'trees.bin').stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
