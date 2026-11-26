#!/usr/bin/env python3
"""Compare the sm_86 cubins of a CUDA_ARCHS=86 build against the released image's sm_86 cubins.

Upstream ships sm_86;sm_89;sm_120 for the CMake engines, so whole-file hashes cannot match a
86-only build. The sm_86 SASS must match if the source, flags and toolchain are the same.

Usage: check-cubins.py <built-lib> <upstream-lib> <label>
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

CUOBJDUMP = "/opt/cuda/bin/cuobjdump"


def sm86_cubins(lib: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([CUOBJDUMP, "--extract-elf", "all", str(lib)], cwd=td, check=True,
                       capture_output=True)
        out = {}
        for p in Path(td).glob("*.cubin"):
            if ".sm_86." in p.name:
                out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out


def main() -> int:
    built, upstream, label = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    b = sm86_cubins(built)
    u = sm86_cubins(upstream)
    missing = sorted(set(u) - set(b))
    extra = sorted(set(b) - set(u))
    diff = sorted(k for k in set(u) & set(b) if u[k] != b[k])
    lines = [
        f"# {label}: {built.name} vs released {upstream.name}",
        f"sm_86 cubins upstream: {len(u)}, built: {len(b)}",
        f"missing in build: {missing}",
        f"extra in build: {extra}",
        f"differing: {diff}",
        "verdict: " + ("sm_86 cubins identical" if not (missing or extra or diff) else "DIFFERS"),
    ]
    print("\n".join(lines))
    return 0 if not (missing or extra or diff) else 1


if __name__ == "__main__":
    sys.exit(main())
