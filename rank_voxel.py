"""Rank viewing spots on the voxel model, with a metric that cannot be gamed.

Two things change against rank_spots.py.

First, the model: a sight line that only crosses canopy is no longer "blocked",
it carries a transmittance. At a 3.4 deg sun most park sight lines run UNDER the
crowns, which is why the old ranking put the Fockeberg -- a hill built out of
war rubble, specifically to stand on -- at 1% open.

Second, the statistic. The old ranking scored a landmark by the BEST cell within
400 m, a maximum over ~40,000 cells. Any 2 m raster has single-cell noise, so the
maximum is a measurement of the noise, not of the place: it says a spot exists,
not that you could find it. This scores the FRACTION of standable ground within
200 m that can see the sun, which degrades gracefully and answers the question a
person actually has -- "if I walk there, will I get a view?"

Reported at both extinction coefficients. K_LEAFOFF is the measured January value
and therefore an upper bound on how much sun gets through; K_AUGUST is the
leaf-on estimate. Where the two disagree, the honest answer is "it depends how
dense the leaves are", and the table shows it rather than hiding it.

Two statistics, because they answer different questions and a single number
cannot do both:

  ODDS  -- the mean transmittance over standable ground. Beer-Lambert
           transmittance is the probability that one sight line misses every
           leaf, so this is the share of sight lines from the area that reach
           the sun. It is what you want when you can wander and wait.
  PATCH -- the area of the largest CONNECTED piece of ground with a reliable
           (T >= 0.5) view. A hill fails the ODDS test unfairly, because a 200 m
           disc around a summit is mostly the summit's own flanks; a connected
           patch of a few hundred square metres on top is the real answer. It
           also cannot be faked by single-cell raster noise, which is what sank
           the old best-cell-within-400 m ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rank_spots import ALLOWED_KINDS, KIND_RANK, LANDMARKS, load_json, rasterize
from voxel_march import K_AUGUST, K_LEAFOFF

DATA = Path("data")
OUT = DATA / "voxout"
RADIUS = 200.0        # metres of walking around the named point
OPEN = 0.5            # transmittance at which you can plainly see the sun
ANY = 0.1             # ... and at which you can still tell it is there


def standable_mask(shape, minx, miny, res):
    ny, nx = shape
    gmeta = json.loads((DATA / "grid_meta.json").read_text())
    j0 = int((minx - gmeta["minx"]) / res)
    i0 = int((miny - gmeta["miny"]) / res)
    obj = np.load(DATA / "obj2m.npy", mmap_mode="r")[i0:i0 + ny, j0:j0 + nx]
    water = rasterize(load_json("places_water.json"), shape, minx, miny, res)
    public = rasterize(load_json("places_public.json"), shape, minx, miny, res)
    private = rasterize(load_json("places_private.json"), shape, minx, miny, res)
    paths = rasterize(load_json("places_paths.json"), shape, minx, miny, res,
                      lines=True, width=5)
    return (np.asarray(obj) < 1.0) & ~water & (public | paths) & ~(private & ~public & ~paths)


SPOT_R = 11          # cells; a ~22 m across standing area, ~380 m2


def local_odds(vis, stand):
    """Odds averaged over a ~380 m2 standing area, so a single cell cannot carry it.

    This is deliberately a maximum again -- but of a field already averaged over
    ~95 standable cells, so raster noise is divided by 95 before it can win. That
    is what the old best-cell-in-400 m ranking got wrong: it maximised the noise
    itself. Averaging first is also the only way to treat a hill fairly, whose
    summit is real but small, without also rewarding a one-cell gap in a wood.
    """
    from scipy.ndimage import uniform_filter
    w = 2 * SPOT_R + 1
    num = uniform_filter(np.where(stand, vis, 0.0).astype(np.float32), w, mode="constant")
    den = uniform_filter(stand.astype(np.float32), w, mode="constant")
    out = np.where(den > 0.15, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    out[~stand] = 0.0
    return out


def main():
    meta = json.loads((OUT / "vox_meta.json").read_text())
    minx, maxx, miny, maxy = meta["extent"]
    res = meta["res"]

    v = np.load(OUT / "vox_2010.npz")
    cls, veg = v["cls"], v["vegpath"].astype(np.float32)
    ny, nx = cls.shape
    late = np.load(OUT / "vox_2030.npz")

    def vis_of(cls_, veg_, k):
        return np.where(cls_ == 0, 1.0,
                        np.where(cls_ == 2, np.exp(-k * veg_), 0.0)).astype(np.float32)

    vis_a = vis_of(cls, veg, K_AUGUST)
    vis_l = vis_of(cls, veg, K_LEAFOFF)
    vis_late = vis_of(late["cls"], late["vegpath"].astype(np.float32), K_AUGUST)

    stand = standable_mask((ny, nx), minx, miny, res)
    print(f"standable: {stand.sum():,} cells ({100*stand.mean():.1f}% of the area)")
    print(f"of standable ground at 20:10 (k={K_AUGUST}/m): "
          f"{100*(vis_a[stand] >= OPEN).mean():.1f}% open, "
          f"{100*(cls[stand] == 0).mean():.1f}% wholly unobstructed, "
          f"{100*(cls[stand] == 1).mean():.1f}% hard-blocked")

    named = [p for p in load_json("places_named.json") if p["k"] in ALLOWED_KINDS]

    def centre(needle):
        """The feature the NAME means, not the nearest thing that contains it.

        Tie-breaking on distance to the Markt silently resolved "Fockeberg" to
        "Spielplatz Am Fockeberg" -- the playground at the foot of the hill --
        and then reported the hill as 74% walled, which is true of its base and
        false of its summit. Exact name first, then the kind that makes a
        destination (a peak or a park beats a playground), then distance.
        """
        hits = [p for p in named if needle.lower() in p["n"].lower()]
        if not hits:
            return None
        return min(hits, key=lambda p: (
            p["n"].strip().lower() != needle.strip().lower(),
            next((r for r, s in enumerate(KIND_RANK) if p["k"] in s), 3),
            abs(p["x"] - 317035) + abs(p["y"] - 5690878)))

    loc_a, loc_l, loc_late = (local_odds(v, stand) for v in (vis_a, vis_l, vis_late))

    rows, skipped = [], []
    for label, needle in LANDMARKS:
        p = centre(needle)
        if p is None:
            skipped.append((label, "no such named feature in OSM"))
            continue
        # Grow the radius until there is real ground to stand on. A lake's named
        # point is its centroid, so a fixed 200 m disc around "Auensee" is open
        # water and scores the lake, not the shore you would watch from.
        for rad in (RADIUS, 300.0, 400.0, 600.0):
            r = int(rad / res)
            j = int((p["x"] - minx) / res)
            i = int((p["y"] - miny) / res)
            if not (r <= i < ny - r and r <= j < nx - r):
                r = None
                break
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disc = (yy ** 2 + xx ** 2) <= r ** 2
            sl = (slice(i - r, i + r + 1), slice(j - r, j + r + 1))
            m = stand[sl] & disc
            n = int(m.sum())
            if n * res * res >= 20000:            # 2 ha of standable ground
                break
        if r is None:
            skipped.append((label, "too close to the edge of the study area"))
            continue
        if n * res * res < 4000:                  # under 0.4 ha even at 600 m
            skipped.append((label, f"only {n*res*res/1e4:.2f} ha standable within 600 m"))
            continue
        sub = loc_a[sl] * disc
        k = int(np.argmax(sub))
        bi, bj = divmod(k, sub.shape[1])
        pct = lambda v: max(0.0, 100 * float(v))      # uniform_filter can emit -0.0
        rows.append(dict(
            name=label, n=n, radius_m=rad,
            clear=pct((cls[sl][m] == 0).mean()),
            blocked=pct((cls[sl][m] == 1).mean()),
            odds_a=pct(vis_a[sl][m].mean()),
            odds_l=pct(vis_l[sl][m].mean()),
            best_a=pct(sub[bi, bj]),
            best_l=pct((loc_l[sl] * disc)[bi, bj]),
            best_late=pct((loc_late[sl] * disc)[bi, bj]),
            walk=float(np.hypot(bi - r, bj - r) * res),
            bx=minx + (j - r + bj) * res, by=miny + (i - r + bi) * res,
        ))

    rows.sort(key=lambda d: -d["best_a"])
    print("\n=== NAMED SPOTS at 20:10 -- share of sight lines that reach the sun ===")
    print(f"{'':24s} {'area':>7s} {'r':>5s} {'anywhere':>9s} {'best spot':>10s} "
          f"{'leafoff':>8s} {'walk':>6s} {'wall':>6s} {'20:30':>6s}")
    for d in rows:
        print(f"{d['name']:24s} {d['n']*4/1e4:6.1f}ha {d['radius_m']:4.0f}m "
              f"{d['odds_a']:8.1f}% {d['best_a']:9.1f}% {d['best_l']:7.1f}% "
              f"{d['walk']:5.0f}m {d['blocked']:5.1f}% {d['best_late']:5.1f}%")
    for label, why in skipped:
        print(f"{label:24s}  -- not scored: {why}")

    # Citywide sweep, over the same ~380 m2 standing areas, so it is directly
    # comparable with the "best spot" column. Ranked by distance from the Markt,
    # not by score: out in the Neuseenland dozens of places score a flat 100% and
    # a score-ranked list would just be twelve ties. The useful question there is
    # "what is the nearest one".
    from scipy.ndimage import uniform_filter
    rs = int(RADIUS / res)
    w = 2 * rs + 1
    tot = uniform_filter(stand.astype(np.float32), w, mode="constant") * w * w * res * res
    cand = np.where(tot >= 20000, loc_a, 0.0)

    print(f"\n=== NEAREST WHOLLY OPEN STANDING AREAS (>= 90% of sight lines, "
          f">= 2 ha standable within {RADIUS:.0f} m) ===")
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    dist_markt = np.hypot(minx + jj * res - 317035.0, miny + ii * res - 5690878.0)
    del jj, ii
    pick = np.where(cand >= 0.90, dist_markt, np.inf)
    shown = 0
    while shown < 10:
        k = int(np.argmin(pick))
        i, j = divmod(k, nx)
        if not np.isfinite(pick[i, j]):
            break
        x, y = minx + j * res, miny + i * res
        d = sorted(((np.hypot(p["x"] - x, p["y"] - y), p) for p in named),
                   key=lambda t: t[0])
        nm = d[0][1]["n"] if d else "?"
        print(f"  {100*cand[i,j]:5.1f}%   {dist_markt[i,j]/1000:4.1f} km from the Markt   "
              f"E {x:.0f} N {y:.0f}   near {nm} ({d[0][0]:.0f} m)")
        i0, i1 = max(0, i - 3 * rs), min(ny, i + 3 * rs)
        j0, j1 = max(0, j - 3 * rs), min(nx, j + 3 * rs)
        pick[i0:i1, j0:j1] = np.inf       # suppress the neighbourhood, not just the cell
        shown += 1

    (OUT / "ranked_voxel.json").write_text(json.dumps(
        {"radius_m": RADIUS, "open_threshold": OPEN, "k_august": K_AUGUST,
         "k_leafoff": K_LEAFOFF, "at": "20:10", "rows": rows,
         "skipped": [{"name": a, "why": b} for a, b in skipped]}, indent=2))
    print(f"\nsaved {OUT}/ranked_voxel.json")


if __name__ == "__main__":
    main()
