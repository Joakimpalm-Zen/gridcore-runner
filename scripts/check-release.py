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


def python_version_for_release(version):
    """Translate Runner's alpha spelling to its PEP 440 package spelling."""
    if version.endswith("-alpha"):
        return version[:-len("-alpha")] + "a0"
    return version


def pyproject_version(path):
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', read(path), re.M)
    if not match:
        raise ValueError(f"no project version found in {path}")
    return match.group(1)


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
    build_lines = build_info.splitlines()
    if build_lines[:1] != [expected_binary]:
        ok &= fail("BUILD-INFO first line does not match binary version")
    if f"tag:        {args.tag}" not in build_lines:
        ok &= fail("BUILD-INFO tag line is inconsistent")
    if f"commit:     {args.commit}" not in build_lines:
        ok &= fail("BUILD-INFO commit line is inconsistent")

    expected_python = python_version_for_release(version)
    try:
        got_python = pyproject_version(args.python_pyproject)
    except (OSError, ValueError) as exc:
        ok &= fail(f"cannot read Python package version: {exc}")
    else:
        if got_python != expected_python:
            ok &= fail(
                f"Python package version {got_python!r} does not match "
                f"release {version!r} ({expected_python!r})"
            )

    release_workflow = args.release_workflow
    if release_workflow.exists():
        workflow = release_workflow.read_text(encoding="utf-8")
        for stale in stale_release_strings(workflow, version, args.tag):
            ok &= fail(f"release workflow contains stale release string {stale!r}")

    for path in args.current_docs:
        for stale in stale_release_strings(read(path), version, args.tag):
            ok &= fail(f"current document {path} contains stale release string {stale!r}")

    # A compatibility ledger nobody publishes is a private spreadsheet. The
    # reports have existed since 2026-08-13; this makes producing one part of
    # cutting a release instead of part of remembering to. The gate only
    # asks that a dated report for THIS version exists — what it contains is
    # the manifest's business, and a report whose checks did not run says so
    # itself (RNR-007).
    if args.compat_reports and args.compat_reports.is_dir():
        stem = f"{version}-"
        if not any(p.name.startswith(stem) and p.suffix == ".json"
                   for p in args.compat_reports.iterdir()):
            ok &= fail(
                f"no compat report for {version} in {args.compat_reports} "
                f"(expected {stem}<date>.json — run scripts/compat_matrix.py "
                f"on a box that has the pinned models and commit the report)"
            )

    return ok


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--release-workflow", type=Path,
                        default=ROOT / ".github/workflows/release.yml")
    parser.add_argument("--python-pyproject", type=Path,
                        default=ROOT / "python/pyproject.toml")
    parser.add_argument("--compat-reports", type=Path,
                        default=ROOT / "docs" / "compat-reports",
                        help="directory of dated per-release compatibility "
                             "reports; the release must ship one for its own "
                             "version")
    parser.add_argument("--commit", required=True,
                        help="commit SHA expected in BUILD-INFO.txt")
    parser.add_argument(
        "--current-doc", dest="current_docs", type=Path, action="append",
        default=[ROOT / "SECURITY.md", ROOT / "CONTRIBUTING.md",
                 ROOT / "docs/moe-support.md"],
        help="current (non-historical) document that may not contain stale versions",
    )
    args = parser.parse_args(argv)
    return 0 if check(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
