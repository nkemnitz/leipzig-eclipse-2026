"""Horizon elevation angle toward the sun, per ground cell.

This is the discriminating quantity. "Sunlit / not sunlit" saturates -- huge parts
of the outskirts are simply lit for the whole eclipse and tie on any binary score.
The angle the western skyline subtends is continuous: a spot whose horizon sits at
0.5 deg is far more robust than one at 3.4 deg, even though both are "sunlit" while
the sun is at 3.5 deg.

Margin = sun altitude - horizon angle. Positive means the sun is visible; the size
tells you how much slack you have against tree growth, haze and my own error bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from viewshed import EYE_H, build_pyramid, grid_convergence_deg, horizon_tangent, load_grid

DATA = Path("data")

# (label, azimuth, sun apparent altitude) at the moments that matter.
KEY = [
    ("20:10", 289.75, 3.50),   # maximum eclipse, 86% obscured
    ("20:30", 293.61, 0.80),   # late, 52% obscured
    ("19:45", 284.97, 7.13),   # early, 40% obscured
]


def main():
    dsm, dtm, meta = load_grid()
    res = meta["res"]
    eye = (dtm + EYE_H).astype(np.float32)
    print("building pyramid ...", end="", flush=True)
    pyr = build_pyramid(dsm)
    print(" done")

    j0 = int((meta["out_minx"] - meta["minx"]) / res)
    j1 = int((meta["out_maxx"] - meta["minx"]) / res)
    i0 = int((meta["out_miny"] - meta["miny"]) / res)
    i1 = int((meta["out_maxy"] - meta["miny"]) / res)

    conv = grid_convergence_deg((meta["out_minx"] + meta["out_maxx"]) / 2,
                                (meta["out_miny"] + meta["out_maxy"]) / 2)
    print(f"grid convergence {conv:+.3f} deg (true azimuth -> UTM grid bearing)")

    out = {}
    for label, az, alt in KEY:
        print(f"  horizon toward az {az:6.2f} ({label}) ...", end="", flush=True)
        ht = horizon_tangent(dsm, eye, az + conv, res, pyramid=pyr)
        deg = np.degrees(np.arctan(ht[i0:i1, j0:j1])).astype(np.float32)
        out[label] = deg
        margin = alt - deg
        print(f" horizon median {np.median(deg):5.2f} deg, "
              f"{100*(margin > 0).mean():5.1f}% have the sun above the skyline")

    np.savez_compressed(DATA / "horizon.npz", **out,
                        labels=np.array([k for k, _, _ in KEY]),
                        az=np.array([a for _, a, _ in KEY], dtype=np.float32),
                        alt=np.array([e for _, _, e in KEY], dtype=np.float32))
    # Extent must come from grid_meta, never be restated here -- a hardcoded copy
    # silently desynced from the real output region once the area moved south.
    (DATA / "horizon_meta.json").write_text(json.dumps(
        {"extent": [meta["out_minx"], meta["out_maxx"],
                    meta["out_miny"], meta["out_maxy"]],
         "res": res, "grid_convergence_deg": conv,
         "key": [{"t": k, "az": a, "alt": e} for k, a, e in KEY]}, indent=2))
    print("saved data/horizon.npz")


if __name__ == "__main__":
    main()
