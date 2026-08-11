"""Bake the three-class voxel result into textures the viewer can sample.

Two images, because the answer has two independent parts that must not be
averaged together:

  vox_wall.png   packed bitplanes, plane k set where the 8 m cell is mostly
                 HARD blocked at timestamp k -- terrain or masonry, no amount of
                 waiting or shuffling helps.
  vox_trans.png  R, G, B = mean transmittance at 19:45, 20:10 and 20:30. Three
                 channels is not a compromise: transmittance changes smoothly and
                 monotonically as the sun sinks, so the viewer interpolates
                 between the keys for the other fourteen timestamps and the error
                 is far below what the 8 m grid can express anyway.

Row 0 is south and the PNG is written unflipped, matching pack_bitplanes and the
flipY=false the shader relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_viewer import pack_bitplanes

SRC = Path("data/voxout/vox_timeline.npz")
OUT = Path("viewer/data")
KEYS = ("19:45", "20:10", "20:30")
WALL_AT = 0.5          # an 8 m cell counts as wall if most of it is


def main():
    z = np.load(SRC)
    wall, trans = z["wall"], z["trans"]
    times = [str(t) for t in z["times"]]
    ny, nx = wall.shape[1:]
    print(f"{len(times)} timestamps at {ny} x {nx} ({float(z['res']):g} m)")

    pack_bitplanes(wall >= WALL_AT * 255).save(OUT / "vox_wall.png", optimize=True)

    idx = [times.index(k) for k in KEYS]
    rgb = np.stack([trans[i] for i in idx], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(OUT / "vox_trans.png", optimize=True)

    meta = json.loads(Path("data/voxout/vox_meta.json").read_text())
    (OUT / "vox_meta.json").write_text(json.dumps({
        "times": times, "keys": list(KEYS), "key_index": idx,
        "res": float(z["res"]), "extent": [int(v) for v in z["extent"]],
        "k_august": float(z["k_august"]), "k_leafoff": float(z["k_leafoff"]),
        "rows": meta["rows"],
    }))

    for k, i in zip(KEYS, idx):
        w = (wall[i] >= WALL_AT * 255).mean()
        t = trans[i] / 255.0
        print(f"  {k}: wall {100*w:5.1f}%   mean transmittance {100*t.mean():5.1f}%   "
              f"cells over 50% {100*(t > 0.5).mean():5.1f}%")
    for n in ("vox_wall.png", "vox_trans.png"):
        print(f"  {n}  {(OUT/n).stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
