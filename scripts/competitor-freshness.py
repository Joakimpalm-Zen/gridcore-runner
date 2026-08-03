#!/usr/bin/env python3
"""Check published torture-result runtime versions against upstream releases.

This performs metadata-only registry requests. It never starts a runtime,
loads a model, or runs inference. Registry failures are reported as skips so a
transient network outage cannot make the scheduled workflow red.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests" / "torture" / "results"
REGISTRIES = {
    "llama.cpp": "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
    "ollama": "https://api.github.com/repos/ollama/ollama/releases/latest",
    "vllm": "https://pypi.org/pypi/vllm/json",
}


def normalized_version(runtime, value):
    text = str(value).strip().split()[0].removeprefix("v")
    if runtime == "llama.cpp":
        match = re.fullmatch(r"b(\d+)", text)
        if not match:
            raise ValueError(f"invalid llama.cpp release {value!r}")
        return (int(match.group(1)),), f"b{int(match.group(1))}"
    match = re.fullmatch(r"(\d+(?:\.\d+)+)(?:[-+].*)?", text)
    if not match:
        raise ValueError(f"invalid {runtime} release {value!r}")
    canonical = match.group(1)
    return tuple(int(part) for part in canonical.split(".")), canonical


def read_published(results_dir=RESULTS):
    """Return the newest published row for each tracked competitor."""
    newest = {}
    for path in sorted(Path(results_dir).glob("**/report.json")):
        try:
            runtime = json.loads(path.read_text())["runtime"]
            name, raw = runtime["name"], runtime["version"]
            if name not in REGISTRIES:
                continue
            key, version = normalized_version(name, raw)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid published report {path}: {exc}") from exc
        previous = newest.get(name)
        if previous is None or key > previous["key"]:
            newest[name] = {"key": key, "version": version,
                            "reports": [str(path)]}
        elif key == previous["key"]:
            previous["reports"].append(str(path))
    return {name: {"version": row["version"], "reports": row["reports"]}
            for name, row in sorted(newest.items())}


def fetch_json(url):
    request = Request(url, headers={"Accept": "application/json",
                                    "User-Agent": "gridcore-runner-freshness"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def upstream_version(runtime, payload):
    raw = payload["info"]["version"] if runtime == "vllm" else payload["tag_name"]
    return normalized_version(runtime, raw)


def check_freshness(published, fetch=fetch_json):
    result = {"stale": [], "current": [], "skipped": []}
    for runtime, row in sorted(published.items()):
        try:
            payload = fetch(REGISTRIES[runtime])
            upstream_key, upstream = upstream_version(runtime, payload)
            published_key, _ = normalized_version(runtime, row["version"])
        except Exception as exc:  # registry/network/schema failures are skips
            result["skipped"].append({"runtime": runtime, "reason": str(exc)})
            continue
        finding = {"runtime": runtime, "published": row["version"],
                   "upstream": upstream, "reports": row["reports"]}
        result["stale" if published_key < upstream_key else "current"].append(finding)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_freshness(read_published(args.results))
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for status in ("stale", "current", "skipped"):
            for row in result[status]:
                if status == "skipped":
                    print(f"SKIP {row['runtime']}: {row['reason']}")
                else:
                    print(f"{status.upper()} {row['runtime']}: published "
                          f"{row['published']}, upstream {row['upstream']}")
    return 1 if result["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
