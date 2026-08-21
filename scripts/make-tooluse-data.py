#!/usr/bin/env python3
"""Deterministic tool-calling adaptation dataset (train + held-out eval).

Four tools with distinct JSON-schema arg shapes, natural-language requests
generated combinatorially from templates, plus refusal cases (requests whose
tool is NOT in the available set must answer {"tool": "none", "args": {}}).
The behaviors targeted are the ones the engine's own tool discipline cares
about: pick the right tool, produce schema-shaped args, refuse nonexistent
tools, and emit no prose around the JSON.

Everything is seeded and order-stable: the emitted files are byte-identical
across runs, so the dataset sha256 in a training provenance record is
meaningful.

Usage: make-tooluse-data.py <out-dir>   (writes train.jsonl + eval.jsonl)
"""
import json
import pathlib
import random
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tooluse-data")
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(0xC0DE)

TOOLS = "search_files(pattern, path), read_file(path), " \
        "write_file(path, content), list_dir(path)"

SYSTEM = ("You are a tool-calling assistant. Available tools: %s. "
          "Reply with ONLY one JSON object of the form "
          "{\"tool\": \"<name>\", \"args\": {...}} and nothing else. "
          "If no available tool fits the request, reply exactly "
          "{\"tool\": \"none\", \"args\": {}}.\nRequest: %%s\nJSON:" % TOOLS)

EXTS = ["py", "c", "h", "log", "md", "json", "yaml", "txt", "csv", "toml"]
DIRS = ["src", "tests", "docs", "scripts", "build", "data", "config",
        "logs", "assets", "lib"]
FILES = ["README.md", "config.yaml", "main.py", "notes.txt", "Makefile",
         "index.html", "report.csv", "settings.json", "server.log",
         "todo.md"]
CONTENTS = ["hello world", "done", "TODO: fix later", "42", "release notes"]

SEARCH_T = ["find all %s files in the %s directory",
            "search %s for anything ending in .%s",
            "locate every .%s file under %s",
            "which %s files live in %s?"]
READ_T = ["open %s and show it", "display the contents of %s",
          "what does %s say?", "read %s for me"]
WRITE_T = ["write '%s' into %s", "save the text '%s' to %s",
           "create %s containing '%s'"]
LIST_T = ["list the files in %s", "what is inside the %s directory?",
          "show the contents of the %s folder"]
NONE_T = ["deploy the app to production", "send an email to the team",
          "resize the image to 200x200", "play some music",
          "restart the database server", "translate this page to French",
          "post this on the company blog", "schedule a meeting for Friday",
          "order more coffee", "compress the video file"]


def ex_search():
    ext, d = rng.choice(EXTS), rng.choice(DIRS)
    t = rng.choice(SEARCH_T)
    req = t % ((ext, d) if "%s files in" in t or "live in" in t
               else (d, ext) if "search" in t else (ext, d))
    args = {"pattern": "*.%s" % ext, "path": d}
    return req, {"tool": "search_files", "args": args}


def ex_read():
    f = rng.choice(FILES)
    return rng.choice(READ_T) % f, {"tool": "read_file", "args": {"path": f}}


def ex_write():
    f, c = rng.choice(FILES), rng.choice(CONTENTS)
    t = rng.choice(WRITE_T)
    req = t % ((c, f) if t.index("%s") < t.rindex("%s") and "containing"
               not in t else (f, c))
    return req, {"tool": "write_file", "args": {"path": f, "content": c}}


def ex_list():
    d = rng.choice(DIRS)
    return rng.choice(LIST_T) % d, {"tool": "list_dir", "args": {"path": d}}


def ex_none():
    return rng.choice(NONE_T), {"tool": "none", "args": {}}


examples = []
for _ in range(70):
    for make in (ex_search, ex_read, ex_write, ex_list):
        examples.append(make())
for _ in range(56):
    examples.append(ex_none())
rng.shuffle(examples)

seen = set()
uniq = []
for req, gold in examples:
    if req in seen:
        continue
    seen.add(req)
    uniq.append((req, gold))

split = int(len(uniq) * 0.85)
train, evalset = uniq[:split], uniq[split:]

with open(OUT / "train.jsonl", "w") as f:
    for req, gold in train:
        f.write(json.dumps({"prompt": SYSTEM % req,
                            "completion": " " + json.dumps(gold),
                            "weight": 1.0}) + "\n")
with open(OUT / "eval.jsonl", "w") as f:
    for req, gold in evalset:
        f.write(json.dumps({"prompt": SYSTEM % req, "gold": gold}) + "\n")

print("wrote %s (%d train) and %s (%d eval)" %
      (OUT / "train.jsonl", len(train), OUT / "eval.jsonl", len(evalset)))
