"""Mark which voxel columns are SOLID (a building) rather than vegetation.

The Saxon LSC delivery classifies only ground vs "everything else", so the point
cloud alone cannot tell a roof from a crown. OSM footprints can: 145k buildings
for the study area, and a roof is opaque wherever it is. Everything else that is
not ground is treated as vegetation, which is the conservative reading -- a
misclassified building only ever makes a spot look BETTER than it is, so it shows
up as a "clear" claim we can check against the aerial image, not as a silent loss.

Output: data/voxels/solid.npy, a bool over the voxel grid (2 m, row 0 = south).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

Image.MAX_IMAGE_PIXELS = None
OUT = Path("data/voxels")


def main():
    m = json.loads((OUT / "meta.json").read_text())
    nx, ny, res = m["nx"], m["ny"], m["res"]
    minx, miny = m["minx"], m["miny"]

    om = json.loads(Path("data/meta.json").read_text())
    ox, oy = om["origin"]

    img = Image.new("1", (nx, ny), 0)
    d = ImageDraw.Draw(img)
    n = 0
    for b in json.loads(Path("data/buildings.json").read_text()):
        ring = b["r"]
        if len(ring) < 3:
            continue
        pts = [((px + ox - minx) / res, (py + oy - miny) / res) for px, py in ring]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < 0 or min(xs) >= nx or max(ys) < 0 or min(ys) >= ny:
            continue
        d.polygon(pts, fill=1)
        n += 1

    solid = np.asarray(img, dtype=bool)          # row 0 = south already
    # One cell of growth so eaves and the odd metre of footprint error still read
    # as roof rather than as porous crown.
    solid = binary_dilation(solid, np.ones((3, 3), bool))
    np.save(OUT / "solid.npy", solid)
    print(f"{n:,} building footprints -> {solid.sum():,} solid cells "
          f"({100*solid.mean():.1f}% of the grid), {solid.nbytes/1e6:.0f} MB")


if __name__ == "__main__":
    main()
