"""Checks on the voxel march that do not depend on believing the voxel march.

The strong one is a containment invariant against the old DSM model. Voxel
occupancy is a subset of the DSM column by construction: if a voxel exists at
band b of cell q, the DSM there is at least gnd(q) + b*vh. So anything the voxel
march calls BLOCKED must also be blocked in the DSM march -- the new model may
only ever OPEN cells the old one closed, never the reverse. A violation would mean
the near/far handover, the curvature term, the grid convergence, or the band
indexing is wrong, and no amount of plausible-looking output would show it.

The slack is 2 m of vertical quantisation: the march treats material as filling
its whole band, so a ray passing 1.9 m above a roof lands in the same band and
blocks in the voxel model while the DSM lets it through. That is not speculation
-- measured over 667,875 occupied cells, the voxel column top sits at or above
the DSM in 99.4% of them (mean +1.85 m). The direction of that error is safe: it
over-blocks. So the test bounds the violation rate rather than demanding zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path("data")
OUT = DATA / "voxout"


def old_blocked(label):
    """The binary DSM model's answer for a standing observer, same region."""
    z = np.load(DATA / "sunlit.npz")
    times = [str(t) for t in z["times"]]
    i = times.index(label)
    ny, nx = z["shape"]
    lit = np.unpackbits(z["packed"][i])[: ny * nx].reshape(ny, nx).astype(bool)
    return ~lit


def main():
    label = "20:10"
    v = np.load(OUT / f"vox_{label.replace(':','')}.npz")
    cls = v["cls"]
    veg = v["vegpath"].astype(np.float32)
    ob = old_blocked(label)
    assert cls.shape == ob.shape, (cls.shape, ob.shape)
    n = cls.size
    ok = True

    print(f"=== {label}: voxel model vs the DSM model it replaces ===")
    print(f"  DSM says blocked      {100*ob.mean():5.1f}%")
    for c, nm in ((0, "clear"), (1, "blocked"), (2, "through canopy")):
        print(f"  voxel {nm:14s} {100*(cls==c).mean():5.1f}%")

    # (1) containment: voxel-blocked must be DSM-blocked
    viol = (cls == 1) & ~ob
    rate = 100 * viol.sum() / max((cls == 1).sum(), 1)
    print(f"\n  voxel-BLOCKED but DSM-clear : {viol.sum():,} cells "
          f"({rate:.3f}% of blocked)   [band quantisation; bounded at 2%]")
    if rate > 2.0:
        ok = False
        print("  FAIL: the voxel march blocks rays the DSM lets through")

    # (2) the reverse direction is the whole point, and must be large
    opened = ob & (cls != 1)
    print(f"  DSM-blocked but voxel not   : {opened.sum():,} cells "
          f"({100*opened.mean():.1f}% of the map)  <- the opaque-column error")
    if opened.mean() < 0.05:
        ok = False
        print("  FAIL: the voxel model changes almost nothing; is it running?")

    # (3) clear cells must carry no canopy path, canopy cells must carry some
    bad = ((cls == 0) & (veg > 0)).sum() + ((cls == 2) & (veg <= 0)).sum()
    print(f"  class/path disagreement     : {bad:,} cells   [must be 0]")
    if bad:
        ok = False

    # (4) old-sunlit implies not-blocked, up to band quantisation
    s = (~ob) & (cls == 1)
    print(f"  DSM-clear but voxel-blocked : {100*s.mean():.3f}% of the map "
          f"(2 m band quantisation)")

    # (5) canopy path lengths must be physically sane
    p = veg[cls == 2]
    print(f"\n  canopy path  median {np.median(p):5.1f} m  p90 {np.percentile(p,90):5.1f} m"
          f"  max {p.max():5.1f} m")
    if not (0 < np.median(p) < 200):
        ok = False
        print("  FAIL: implausible canopy path lengths")

    m = json.loads((OUT / "vox_meta.json").read_text())
    k = m["k_august"]
    t = np.exp(-k * p)
    print(f"  transmittance at k={k:.4f}/m: >50% for {100*(t>0.5).mean():.1f}% of them, "
          f"<10% for {100*(t<0.1).mean():.1f}%")

    print("\n" + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
