"""Fast close-up check: do the streamed detail tiles actually render?

The full verify_viewer.py run costs ~12 minutes under SwiftShader. This does only
the part that broke: fly onto the Fockeberg until detail tiles stream in, then
screenshot each surface mode and report any console error. A GLSL compile failure
in the detail material shows up here as a console error and a black tile, which is
invisible from the wide view because that uses a different material.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from verify_viewer import FLAGS, PORT, serve

SHOTS = Path("out/shots")
FOCKEBERG = (316183, 5688395)


def main():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    serve()
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=FLAGS)
        page = browser.new_page(viewport={"width": 1400, "height": 850})
        page.set_default_timeout(180000)
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))

        page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="load", timeout=120000)
        page.wait_for_selector("#load", state="detached", timeout=180000)

        page.evaluate("""(xy) => {
            const d = window.__dbg, m = d.meta;
            const [MINX, MAXX, MINY, MAXY] = m.extent;
            const W = MAXX - MINX, D = MAXY - MINY;
            const wx = xy[0] - MINX - W / 2, wz = -(xy[1] - MINY - D / 2);
            d.controls.target.set(wx, 300, wz);
            d.camera.position.set(wx + 300, 420, wz + 340);
            d.controls.update();
            if (d.updateDetail) d.updateDetail();
        }""", list(FOCKEBERG))
        page.wait_for_timeout(9000)

        n = page.evaluate("() => window.__dbg.detail ? window.__dbg.detail.size : -1")
        print(f"detail tiles at the Fockeberg: {n}")
        if n <= 0:
            errors.append("no detail tiles streamed in")

        for mode, name in ((0, "30_detail_aerial"), (1, "31_detail_visible")):
            page.click(f'#modes button[data-mode="{mode}"]')
            page.wait_for_timeout(2500)
            page.screenshot(path=str(SHOTS / f"{name}.png"))
            print(f"  shot {name}.png")

        # A black frame is the signature of a failed material. Measure it on the
        # SCREENSHOT, not with readPixels: the drawing buffer is not preserved
        # after compositing, so reading it outside the render call returns zeros
        # and reports every healthy frame as black.
        import io

        from PIL import Image
        px = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("L"),
                        dtype=np.float32)
        h, w = px.shape
        centre = px[h // 2 - 120:h // 2 + 120, w // 2 - 120:w // 2 + 120]
        print(f"mean centre luminance: {centre.mean():.1f}  (a failed material reads ~0)")
        if centre.mean() < 8:
            errors.append(f"centre of the view is black ({centre.mean():.1f})")

        browser.close()

    print(f"\nconsole errors: {len(errors)}")
    for e in errors[:8]:
        print(f"  {e[:300]}")
    print("PASS" if not errors else "FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
