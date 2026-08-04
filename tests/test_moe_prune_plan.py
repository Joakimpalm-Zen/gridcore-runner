"""scripts/moe-prune-plan.py: saliency-based --prune-experts plan generator.

The gate, per the goal: "unit test on a synthetic trace with a constructed
right answer." The synthetic trace below has hand-computed masses, so every
assertion here is "the tool must produce THIS exact plan," not "the tool's
output looks plausible."

Synthetic masses (one record per (layer, expert), gate value == its whole
mass, so no arithmetic is needed to know the right answer by construction):

  layer 0: expert 0=10, 1=5, 2=3, 3=2   (total 20)
  layer 1: expert 0=7,  1=1, 2=1, 3=1   (total 10)

norms (only used with --use-norms) make expert 3 in layer 0 dominant by
gate*norm despite ranking last by gate mass alone — proves --use-norms
actually changes the ranking, not just scales every score by a constant:

  layer 0 norms: 0=1, 1=1, 2=1, 3=20   -> gate*norm: 10, 5, 3, 40
  layer 1 norms: all 1                 -> gate*norm: same order as mass
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "moe-prune-plan.py"

RECORDS = [
    {"pos": 0, "layer": 0, "experts": [0], "gates": [10], "norms": [1]},
    {"pos": 1, "layer": 0, "experts": [1], "gates": [5], "norms": [1]},
    {"pos": 2, "layer": 0, "experts": [2], "gates": [3], "norms": [1]},
    {"pos": 3, "layer": 0, "experts": [3], "gates": [2], "norms": [20]},
    {"pos": 0, "layer": 1, "experts": [0], "gates": [7], "norms": [1]},
    {"pos": 1, "layer": 1, "experts": [1], "gates": [1], "norms": [1]},
    {"pos": 2, "layer": 1, "experts": [2], "gates": [1], "norms": [1]},
    {"pos": 3, "layer": 1, "experts": [3], "gates": [1], "norms": [1]},
]


@pytest.fixture()
def trace(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return p


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *args],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc


def test_keep_n_uniform_by_gate_mass(trace, tmp_path):
    out = tmp_path / "plan.json"
    proc = _run(str(trace), "--keep-n", "2", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(out.read_text())
    assert plan == {"layer_0": [0, 1], "layer_1": [0, 1]}


def test_coverage_is_nonuniform_per_layer(trace, tmp_path):
    out = tmp_path / "plan.json"
    proc = _run(str(trace), "--coverage", "0.7", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(out.read_text())
    # layer 0 needs its top-2 (10+5=15/20=0.75 >= 0.7; top-1 alone is 0.5);
    # layer 1's mass is concentrated enough that its top-1 ALONE already
    # clears 0.7 (7/10) — a genuinely different COUNT per layer, not just
    # different ids, which is the whole point of coverage mode.
    assert plan == {"layer_0": [0, 1], "layer_1": [0]}
    assert "layer_0: keep 2 of 4, coverage=0.7500" in proc.stderr
    assert "layer_1: keep 1 of 4, coverage=0.7000" in proc.stderr


def test_use_norms_changes_the_ranking(trace, tmp_path):
    out = tmp_path / "plan.json"
    proc = _run(str(trace), "--keep-n", "2", "--use-norms", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(out.read_text())
    # by gate*norm, layer 0 becomes 3=40, 0=10, 1=5, 2=3 -> top-2 = {0, 3},
    # NOT {0, 1} (the gate-mass-only answer from the first test) — expert 3
    # ranks last by mass alone and first by gate*norm.
    assert plan == {"layer_0": [0, 3], "layer_1": [0, 1]}


def test_precision_out_hot_cold_split(trace, tmp_path):
    plan_out = tmp_path / "plan.json"
    prec_out = tmp_path / "prec.json"
    proc = _run(str(trace), "--keep-n", "2", "--out", str(plan_out),
                "--precision-out", str(prec_out), "--hot-fraction", "0.5")
    assert proc.returncode == 0, proc.stderr
    hc = json.loads(prec_out.read_text())
    # 8 (layer,expert) pairs total, hot_fraction=0.5 -> 4 hot. Pooled scores
    # descending: (0,0)=10, (1,0)=7, (0,1)=5, (0,2)=3 | (0,3)=2, (1,1)=1,
    # (1,2)=1, (1,3)=1 -> hot is exactly the first four.
    assert len(hc["hot"]) == 4
    assert len(hc["cold"]) == 4
    assert [0, 0] in hc["hot"] and [1, 0] in hc["hot"]
    assert [0, 3] in hc["cold"]


def test_requires_exactly_one_mode(trace, tmp_path):
    out = tmp_path / "plan.json"
    proc = _run(str(trace), "--out", str(out))
    assert proc.returncode != 0
    proc = _run(str(trace), "--keep-n", "2", "--coverage", "0.7", "--out", str(out))
    assert proc.returncode != 0


def test_prune_plan_output_is_valid_prune_experts_input(trace, tmp_path):
    # The whole point: this tool's --out is directly consumable by
    # runner --prune-experts, no reformatting.
    out = tmp_path / "plan.json"
    proc = _run(str(trace), "--keep-n", "2", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(out.read_text())
    assert all(k.startswith("layer_") for k in plan)
    assert all(isinstance(v, list) and all(isinstance(x, int) for x in v)
               for v in plan.values())
