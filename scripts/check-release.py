#!/usr/bin/env python3
"""Release artifact consistency gate.

This is intentionally narrow: it checks the files that are packaged or directly
drive release packaging. Historical reports may mention older versions; the
packaged README and workflow comments must not drift.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_STRING_RE = re.compile(r"\bv?\d+\.\d+\.\d+-alpha\b")


def fail(msg):
    print(f"release-check: {msg}", file=sys.stderr)
    return False


def binary_version(binary):
    proc = subprocess.run([str(Path(binary).resolve()), "--version"], text=True,
                          capture_output=True, timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"{binary} --version failed")
    return proc.stdout.strip()


def stale_release_strings(text, version, tag):
    stale = set()
    for s in RELEASE_STRING_RE.findall(text):
        if s not in {version, tag}:
            stale.add(s)
    return sorted(stale)


def read(path):
    return Path(path).read_text(encoding="utf-8")


def check(args):
    ok = True
    version = args.tag[1:] if args.tag.startswith("v") else args.tag
    expected_binary = f"runner {version}"

    got = binary_version(args.binary)
    if got != expected_binary:
        ok &= fail(f"binary version {got!r} does not match tag {args.tag!r}")

    readme = read(args.readme)
    if f"Public alpha (`{version}`)" not in readme:
        ok &= fail(f"README does not identify public alpha {version}")
    if f"./runner --version   # -> runner {version}" not in readme:
        ok &= fail("README version-output example is not in sync")
    for stale in stale_release_strings(readme, version, args.tag):
        ok &= fail(f"README contains stale release string {stale!r}")

    changelog = read(args.changelog)
    if not re.search(rf"^## v?{re.escape(version)}\b", changelog, re.M):
        ok &= fail(f"CHANGELOG has no section for {version}")

    build_info = read(args.build_info)
    if expected_binary not in build_info.splitlines()[:1]:
        ok &= fail("BUILD-INFO first line does not match binary version")
    if f"tag:        {args.tag}" not in build_info:
        ok &= fail("BUILD-INFO tag line is inconsistent")

    release_workflow = ROOT / ".github/workflows/release.yml"
    if release_workflow.exists():
        workflow = release_workflow.read_text(encoding="utf-8")
        for stale in stale_release_strings(workflow, version, args.tag):
            ok &= fail(f"release workflow contains stale release string {stale!r}")

    return ok


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    parser.add_argument("--build-info", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if check(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
