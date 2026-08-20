"""scripts/type-plan-size.py: what will this --type-plan actually produce?

A --type-plan is written blind: the rules are substrings, first match wins,
and the quantizer silently declines a rule whose target block width does not
divide the tensor's row (`type_fits_row`) or that would GROW the tensor. Both
declines produce a successful build and a file that is not what the plan said.

The gate here is exactness, not an estimate: the predicted byte size must
equal the size `--quantize` actually writes, and the predicted per-type
histogram must equal the one the quantizer prints.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/type-plan-size.py"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def moe_model(tmp_path_factory):
    base = tmp_path_factory.mktemp("tps") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return pathlib.Path(f"{base}.moe4.gguf")


def _predict(model, plan_path):
    out = subprocess.run([sys.executable, SCRIPT, str(model), str(plan_path), "--json"],
                         check=True, cwd=ROOT, stdout=subprocess.PIPE)
    return json.loads(out.stdout)


def _build(runner_bin, model, plan_path, out_path):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--quantize", str(out_path),
         "--type-plan", str(plan_path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return (proc.stdout + proc.stderr).decode(errors="replace")


def _histogram_from_log(log):
    m = re.search(r"output histogram: (.+)", log)
    assert m, f"quantizer printed no histogram:\n{log}"
    return {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", m.group(1))}


def test_predicted_size_and_histogram_are_exact(runner_bin, moe_model, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"default": "q8_0",
                                "rules": [{"match": "_exps.weight", "type": "q4_0"}]}))
    pred = _predict(moe_model, plan)
    out = tmp_path / "out.gguf"
    log = _build(runner_bin, moe_model, plan, out)

    assert pred["predicted_bytes"] == out.stat().st_size
    assert pred["histogram"] == _histogram_from_log(log)


def test_keep_default_predicts_a_byte_copy(runner_bin, moe_model, tmp_path):
    plan = tmp_path / "keep.json"
    plan.write_text(json.dumps({"default": "keep", "rules": []}))
    pred = _predict(moe_model, plan)
    out = tmp_path / "keep.gguf"
    _build(runner_bin, moe_model, plan, out)
    assert pred["predicted_bytes"] == out.stat().st_size
    assert pred["declined_row_width"] == []


def test_row_width_decline_is_reported_not_silently_applied(runner_bin, moe_model, tmp_path):
    # The fixture's rows are far narrower than Q3_K's 256-wide super-block, so
    # every q3_k rule must be reported as declined and cost nothing: the built
    # file has to come out the same size as the keep-everything rewrite.
    keep_plan = tmp_path / "keep.json"
    keep_plan.write_text(json.dumps({"default": "keep", "rules": []}))
    keep_out = tmp_path / "keep.gguf"
    _build(runner_bin, moe_model, keep_plan, keep_out)

    plan = tmp_path / "q3k.json"
    plan.write_text(json.dumps({"default": "keep",
                                "rules": [{"match": "_exps.weight", "type": "q3_k"}]}))
    pred = _predict(moe_model, plan)
    out = tmp_path / "q3k.gguf"
    _build(runner_bin, moe_model, plan, out)

    assert pred["declined_row_width"], "a q3_k rule on sub-256 rows reported no decline"
    assert pred["predicted_bytes"] == out.stat().st_size == keep_out.stat().st_size
