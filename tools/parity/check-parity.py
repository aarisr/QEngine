#!/usr/bin/env python3
"""Compare a built QEngine image's /app tree against the released 1.5.0 image.

Usage: check-parity.py <built-image-ref> <label>

Writes parity/<label>-manifest.txt, parity/<label>-hashes.txt and
parity/<label>-compare.txt; prints a short verdict.
"""
import hashlib
import subprocess
import argparse
from pathlib import Path

OUT = Path(__file__).resolve().parent
# Files whose bytes must be identical for a layout/behaviour-parity claim.
MUST_MATCH = ("server/", "vectors/", "data/", "tools/", "LICENSE", "VERSION")
# Files expected to differ, with the reason.
ARCH_FILES = {
    "build/libqengine.so": "CUDA arch set (sm_86 vs sm_86;sm_89;sm_120)",
    "build/libqflash.so": "CUDA arch set (sm_86 vs sm_86;sm_89;sm_120)",
    "build/libqglm.so": "CUDA arch set (sm_86 vs sm_86;sm_89;sm_120)",
    "build/flash_run": "CUDA arch set (sm_86 vs sm_86;sm_89;sm_120)",
}
BUILDID = {"build/qe_build.txt": "records the al0 commit, not d091eaee"}


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def snapshot(image: str, label: str) -> dict[str, tuple[int, str]]:
    manifest = sh(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            image,
            "-c",
            'cd /app && find . -type f -printf "%p %s\\n" | sort && '
            'find . -type f | sort | xargs -r sha256sum',
        ]
    )
    entries: dict[str, tuple[int, str]] = {}
    for line in manifest.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("./") and parts[1].isdigit():
            entries[parts[0][2:]] = (int(parts[1]), "")
        elif len(parts) == 2 and parts[1].startswith("./"):
            path = parts[1][2:]
            if path in entries:
                entries[path] = (entries[path][0], parts[0])
    if not entries:
        raise SystemExit(f"no /app files captured from {image}")
    (OUT / f"{label}-manifest.txt").write_text(
        "".join(f"./{p} {size}\n" for p, (size, _) in sorted(entries.items()))
    )
    (OUT / f"{label}-hashes.txt").write_text(
        "".join(f"{digest}  ./{p}\n" for p, (_, digest) in sorted(entries.items()))
    )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("built_image")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expect-identical", action="store_true")
    args = parser.parse_args()
    image, upstream_image, label = args.built_image, args.upstream, args.label
    expected_diff = {} if args.expect_identical else {**ARCH_FILES, **BUILDID}
    ours = snapshot(image, label)
    upstream = snapshot(upstream_image, "upstream")

    lines = [f"# parity: {image} vs {upstream_image}", ""]
    only_up = sorted(set(upstream) - set(ours))
    only_ours = sorted(set(ours) - set(upstream))
    lines += [f"files upstream only: {only_up}", f"files built only: {only_ours}", ""]
    bad = []
    expected = []
    for p in sorted(set(upstream) & set(ours)):
        us, uh = upstream[p]
        bs, bh = ours[p]
        if us == bs and uh == bh:
            continue
        reason = expected_diff.get(p)
        row = f"{p}: upstream {us}B {uh[:16]} | built {bs}B {bh[:16]}"
        if reason:
            expected.append(f"{row}  # expected: {reason}")
        else:
            bad.append(row)
    lines += ["## must-match files that differ", *([] if not bad else bad), ""]
    lines += ["## expected differences", *([] if not expected else expected), ""]

    entry_up = ""
    entry_ours = ""
    try:
        entry_up = sh(
            ["docker", "run", "--rm", "--entrypoint", "bash", upstream_image, "-c",
             "sha256sum /usr/local/bin/qengine-entrypoint"]
        ).split()[0]
        entry_ours = sh(
            ["docker", "run", "--rm", "--entrypoint", "bash", image, "-c",
             "sha256sum /usr/local/bin/qengine-entrypoint"]
        ).split()[0]
    except subprocess.CalledProcessError as exc:
        lines.append(f"entrypoint compare failed: {exc}")
    lines += [f"entrypoint upstream {entry_up}", f"entrypoint built    {entry_ours}",
              f"entrypoint identical: {entry_up == entry_ours}", ""]

    report = OUT / f"{label}-compare.txt"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {report}")
    return 0 if not bad and entry_up == entry_ours else 1


if __name__ == "__main__":
    sys.exit(main())
