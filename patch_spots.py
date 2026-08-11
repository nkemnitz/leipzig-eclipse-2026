"""Rebuild the viewer's spot list: named landmarks first, scored on the voxel model.

Two things were wrong with the old list.

The RANKING was margin in degrees against the opaque-column DSM, which next to the
canopy layer read as a contradiction -- the Fockeberg listed at +3.4 deg while the
map beside it showed the same hill attenuated.

The CONTENT was whatever the automatic search found, which is honest but useless:
the top of the list was "Apelstein 47" and "DR V 15 Innenbesichtigung", Napoleonic
memorial stones and a locomotive shed out in the Neuseenland. Nobody looking for
somewhere to watch an eclipse in Leipzig is helped by that. So the list now leads
with the places a person would actually think of -- the ones any "sunset spots in
Leipzig" search returns -- each anchored on its own landmark and scored honestly,
including when the honest score is bad. The automatic finds follow underneath,
which is where a surprise belongs.

Rewrites viewer/data/meta.json in place; run after voxel_march.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from rank_spots import ALLOWED_KINDS, KIND_RANK, LANDMARKS, load_json
from rank_voxel import local_odds, standable_mask
from voxel_march import K_AUGUST

META = Path("viewer/data/meta.json")
OUT = Path("data/voxout")
MARKT = (317035.0, 5690878.0)
to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

# Points pinned by measurement rather than by an OSM centroid. Only two, and both
# are derived: the Fockeberg summit is the DTM maximum inside the hill, and the
# Rosentalhügel tower is the location the automatic search already returned. I am
# deliberately NOT typing in coordinates for the other well-known spots from
# memory -- a plausible-looking wrong coordinate would be scored, ranked and
# published exactly like a right one, and nothing downstream would notice.
EXTRA = [
    ("Fockeberg (Gipfel)", 316183, 5688395),
    ("Rosentalhügel (Aussichtsturm)", 315703, 5693033),
]


def main():
    vm = json.loads((OUT / "vox_meta.json").read_text())
    minx, maxx, miny, maxy = vm["extent"]
    res = vm["res"]

    def vis(tag):
        z = np.load(OUT / f"vox_{tag}.npz")
        c, v = z["cls"], z["vegpath"].astype(np.float32)
        return np.where(c == 0, 1.0, np.where(c == 2, np.exp(-K_AUGUST * v), 0.0)
                        ).astype(np.float32), c

    v10, c10 = vis("2010")
    v30, _ = vis("2030")
    ny, nx = v10.shape
    stand = standable_mask((ny, nx), minx, miny, res)
    loc10, loc30 = local_odds(v10, stand), local_odds(v30, stand)

    meta = json.loads(META.read_text())
    named = [p for p in load_json("places_named.json") if p["k"] in ALLOWED_KINDS]

    def centre(needle):
        hits = [p for p in named if needle.lower() in p["n"].lower()]
        if not hits:
            return None
        return min(hits, key=lambda p: (
            p["n"].strip().lower() != needle.strip().lower(),
            next((r for r, s in enumerate(KIND_RANK) if p["k"] in s), 3),
            abs(p["x"] - MARKT[0]) + abs(p["y"] - MARKT[1])))

    def score(x, y, label):
        """Grow the radius until there is real ground to stand on, then report it.

        A lake or a park is named at its centroid, which is water or woodland; a
        fixed disc there measures the middle of the lake and calls the place
        hopeless. Growing until the disc holds standable ground is what makes the
        number mean "if you go to X, this is what you get".
        """
        j, i = int((x - minx) / res), int((y - miny) / res)
        for radius in (250.0, 500.0, 700.0):
            r = int(radius / res)
            if not (r <= i < ny - r and r <= j < nx - r):
                return None
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disc = (yy ** 2 + xx ** 2) <= r ** 2
            sl = (slice(i - r, i + r + 1), slice(j - r, j + r + 1))
            m = stand[sl] & disc
            if m.sum() * res * res >= 20000:         # 2 ha to stand on
                break
        if m.sum() * res * res < 1000:
            return None
        sub = loc10[sl] * disc
        if sub.max() <= 0:
            return None                              # nowhere near it works at all
        k = int(np.argmax(sub))
        bi, bj = divmod(k, sub.shape[1])
        bx = minx + (j - r + bj) * res
        by = miny + (i - r + bi) * res
        lon, lat = to_wgs.transform(x, y)
        pct = lambda v: round(max(0.0, 100 * float(v)), 1)
        return dict(
            label=label, name=label, utm_x=float(x), utm_y=float(y),
            lat=round(lat, 6), lon=round(lon, 6),
            km_from_markt=round(float(np.hypot(x - MARKT[0], y - MARKT[1]) / 1000), 2),
            landmark=True,
            vox=dict(here=pct(vis_a_mean(m, sl)), best=pct(sub[bi, bj]),
                     best_2030=pct((loc30[sl] * disc)[bi, bj]),
                     walk_m=round(float(np.hypot(bi - r, bj - r) * res)),
                     radius_m=radius, area_ha=round(float(m.sum()) * res * res / 1e4, 2),
                     best_x=float(bx), best_y=float(by),
                     wall=pct((c10[sl][disc] == 1).mean())))

    def vis_a_mean(m, sl):
        """Odds a RANDOM standable spot there works, vs the best one (`best`)."""
        return float(v10[sl][m].mean()) if m.any() else 0.0

    spots, seen = [], []
    for label, needle in LANDMARKS:
        p = centre(needle)
        if p is None:
            continue
        d = score(p["x"], p["y"], label)
        if d:
            spots.append(d)
            seen.append((p["x"], p["y"]))
    for label, x, y in EXTRA:
        if any(np.hypot(x - a, y - b) < 250 for a, b in seen):
            continue
        d = score(x, y, label)
        if d:
            spots.append(d)
            seen.append((x, y))

    # Ranked on the best standing area near the landmark, with the walk to it
    # carried alongside, because "0% at the centroid" is true of every lake and
    # useless to anyone. `here` keeps the honest average for the tooltip.
    spots.sort(key=lambda s: -s["vox"]["best"])
    named_n = len(spots)

    # Automatic finds that are not already covered by a landmark, best first.
    auto = []
    for s in meta["spots"]:
        x, y = s["utm_x"], s["utm_y"]
        if any(np.hypot(x - a, y - b) < 500 for a, b in seen):
            continue
        d = score(x, y, s.get("label") or s.get("name") or "?")
        if not d:
            continue
        d["landmark"] = False
        d["profile"] = s.get("profile")
        d["profile_az0"] = s.get("profile_az0")
        d["profile_az1"] = s.get("profile_az1")
        auto.append(d)
        seen.append((x, y))
    auto.sort(key=lambda s: -s["vox"]["best"])

    # Deduplication is landmark-first and never cross-anchor. An automatic find is
    # labelled with its NEAREST named feature, so one 572 m east of Augustusplatz
    # is also called "Augustusplatz" -- and merging by name let it outrank the
    # actual square with a score measured somewhere else entirely. A landmark entry
    # therefore always wins over any find that merely borrowed its name.
    known = {d["label"].split("(")[0].strip().lower() for d in spots}
    keep = [d for d in auto if d["label"].split("(")[0].strip().lower() not in known]
    seen_names = set()
    uniq = []
    for d in keep:
        k = d["label"].split("(")[0].strip().lower()
        if k in seen_names:
            continue
        seen_names.add(k)
        uniq.append(d)
    uniq.sort(key=lambda s: -s["vox"]["best"])
    meta["spots"] = spots + uniq[:14]
    meta["vox"] = {"k_august": K_AUGUST, "k_leafoff": vm["k_leafoff"], "at": "20:10"}
    META.write_text(json.dumps(meta))

    print(f"{named_n} named landmarks + {len(auto)} automatic finds "
          f"-> {len(meta['spots'])} after merging duplicates by name")
    print(f"{'':32s} {'best':>6s} {'avg':>6s} {'walk':>6s} {'area':>7s} {'20:30':>6s}")
    for s in meta["spots"][:22]:
        v = s["vox"]
        tag = "" if s["landmark"] else "  (auto)"
        print(f"{s['label'][:30]:32s} {v['best']:5.1f}% {v['here']:5.1f}% "
              f"{v['walk_m']:4.0f} m {v['area_ha']:5.1f}ha {v['best_2030']:5.1f}%{tag}")


if __name__ == "__main__":
    main()
