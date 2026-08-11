"""Every uniform a shader declares must actually be bound.

WebGL does not error on an unbound sampler2D. It silently resolves to texture
unit 0 -- whatever was bound last -- so the shader reads a real texture, just the
wrong one, and draws something plausible. That is exactly how the detail tiles
came to decode their own elevation bytes as the sunlit bitplanes and paint
terrain contours across "Can I see the sun?" at high zoom: no console error, no
failed request, and the wide view looked perfect because it uses a different
material.

A visual check cannot be trusted to catch this, so check it statically: collect
every `uniform <type> <name>;` in every shader source and confirm each name is
supplied by the material's uniforms object (directly, or via the `shared` /
`uniforms` spreads). Cheap, and it fails loudly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

APP = Path("viewer/app.js")

# three.js injects these itself for ShaderMaterial.
BUILTIN = {
    "modelMatrix", "modelViewMatrix", "projectionMatrix", "viewMatrix",
    "normalMatrix", "cameraPosition", "isOrthographic", "position", "normal", "uv",
    "instanceMatrix", "instanceColor", "logDepthBufFC", "uvTransform",
}

UNIFORM_RE = re.compile(r"\buniform\s+\w+\s+([^;]+);")
DECL_RE = re.compile(r"^\s*(?:const\s+)?(\w+)\s*=\s*\{", re.M)


def names_in(decl: str):
    """`uMap, uPacked` or `uKeyW` -> the bare identifiers."""
    for part in decl.split(","):
        yield part.strip().split("[")[0].strip()


def key_names(block: str):
    """Keys of a JS object literal, including `...spread` markers."""
    keys = set(re.findall(r"(?:^|[,{\s])(\w+)\s*:", block))
    keys |= {f"...{m}" for m in re.findall(r"\.\.\.(\w+)", block)}
    return keys


def brace_block(src: str, start: int) -> str:
    """The {...} beginning at `start`, balanced."""
    depth, i = 0, start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    return src[start:]


def main():
    src = APP.read_text(encoding="utf-8")

    # the two shared uniform dictionaries, by name
    pools = {}
    for name in ("uniforms", "shared"):
        m = re.search(rf"\b{name}\s*=\s*\{{", src)
        if not m:
            print(f"FAIL: could not find the `{name}` uniform dictionary")
            return 1
        pools[name] = key_names(brace_block(src, m.end() - 1))

    bad = []
    checked = 0
    for m in re.finditer(r"uniforms:\s*\{", src):
        block = brace_block(src, m.end() - 1)
        supplied = key_names(block)
        for spread in [s for s in supplied if s.startswith("...")]:
            supplied |= pools.get(spread[3:], set())

        # the shader sources belonging to this material: search forward to the
        # next `uniforms: {` or end of the ShaderMaterial call
        tail = src[m.end():m.end() + 6000]
        declared = set()
        for sh in re.findall(r"(?:vertex|fragment)Shader:\s*(?:/\* glsl \*/)?`(.*?)`",
                             tail, re.S):
            for d in UNIFORM_RE.findall(sh):
                declared |= set(names_in(d))
        if not declared:
            continue
        checked += 1
        missing = {d for d in declared if d not in supplied and d not in BUILTIN}
        if missing:
            bad.append((m.start(), sorted(missing)))

        # And the mirror image, which is worse: a uniform USED but never declared
        # in that shader is a GLSL compile error, so the material renders black and
        # -- where the cover mask has already discarded the coarse terrain beneath
        # it -- punches a hole straight through the map. Binding it on the JS side
        # does not declare it in the source.
        used = set()
        for sh in re.findall(r"(?:vertex|fragment)Shader:\s*(?:/\* glsl \*/)?`(.*?)`",
                             tail, re.S):
            body = UNIFORM_RE.sub("", sh)
            used |= set(re.findall(r"\bu[A-Z]\w*", body))
        undeclared = {u for u in used if u not in declared and u not in BUILTIN}
        if undeclared:
            bad.append((m.start(), [f"{u} (used, never declared)"
                                    for u in sorted(undeclared)]))

    # GLSL chunks pasted into several shaders (LIT_GLSL and friends) declare
    # uniforms outside any material, so fold those in as a second pass.
    for name, chunk in re.findall(r"const (\w+_GLSL)\s*=\s*(?:/\* glsl \*/)?`(.*?)`",
                                  src, re.S):
        for d in UNIFORM_RE.findall(chunk):
            for n in names_in(d):
                if n not in pools["uniforms"] and n not in pools["shared"]:
                    bad.append((0, [f"{n} (declared in {name})"]))

    # A GLSL chunk interpolated into a shader ABOVE its own `const` declaration is
    # a temporal-dead-zone ReferenceError at module load: the whole page stops at
    # "Loading terrain…" with one console line and no other clue.
    decl = {m.group(1): src[:m.start()].count("\n") + 1
            for m in re.finditer(r"const (\w+_GLSL)\s*=", src)}
    for m in re.finditer(r"\$\{(\w+_GLSL)\}", src):
        use = src[:m.start()].count("\n") + 1
        at = decl.get(m.group(1))
        if at is None or at >= use:
            bad.append((m.start(), [f"{m.group(1)} used at line {use} but declared "
                                    f"at {at} (temporal dead zone)"]))

    print(f"checked {checked} shader materials")
    for pos, missing in bad:
        line = src[:pos].count("\n") + 1
        print(f"  FAIL app.js:{line}  {', '.join(missing)}")
    print("PASS" if not bad else f"FAIL ({len(bad)} materials)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
