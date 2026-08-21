"""--prune-experts: stacked-layout MoE expert pruning in the quantize path.

Gate (b), oracle equivalence, built the way the goal asked — "a fixture
whose router provably never selects the pruned experts" — literally, not
just empirically: make-test-moe.py's `pruneprobe` fixture (see its comment)
uses a ZERO router (every expert's raw logit is 0 for any input, since
dot(0, x) == 0) plus an exp_probs_b bias of [1000, 500, 0, -1000] on 4
experts, top-2. Selection is decided entirely by that fixed bias, for every
token at every position, regardless of the model's actual activations:
experts 0 and 1 always win, 2 and 3 never do. That is a proof, not a
measurement from one probe run — verified once below as a sanity check on
the fixture itself, then relied on directly.

Gate (a), fixture round-trip: prune -> load -> generate must simply work
end to end (both halves of gate (b) already exercise this).

Gate (c) (make test green + a dedicated target) is the Makefile's
test-prune-experts target, not this file.
"""
import json
import os
import pathlib
import struct
import subprocess
import sys
from collections import defaultdict

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT, N_GEN = "hello world", 8


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def pruneprobe_model(tmp_path_factory):
    base = tmp_path_factory.mktemp("prune") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT)
    return pathlib.Path(f"{base}.pruneprobe.gguf")


@pytest.fixture(scope="module")
def pruneprobe_bias_model(tmp_path_factory):
    """The same pruneprobe fixture with the OTHER on-disk spelling of the
    selection bias: `blk.N.exp_probs_b.bias`, which is what the shipped
    `nemotron_h_moe` GGUFs use. model.c accepts either name, so the pruner
    has to slice either name."""
    base = tmp_path_factory.mktemp("prune-bias") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT)
    return pathlib.Path(f"{base}.pruneprobe-bias.gguf")


@pytest.fixture(scope="module")
def nemotron_prune_model(tmp_path_factory):
    """Generated Nemotron-H-MoE with two independently prunable MoE layers."""
    base = tmp_path_factory.mktemp("prune-nemotron") / "m"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(base),
         "--arch", "nemotron_h_moe", "--prune-fixture"],
        check=True, cwd=ROOT)
    return pathlib.Path(f"{base}.gguf")


def _generate(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", PROMPT, "-n", str(N_GEN),
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout, proc.stderr


def _write_plan(path, layer_kept):
    path.write_text(json.dumps({f"layer_{l}": ids for l, ids in layer_kept.items()}))


def _prune(runner_bin, src, out, plan_path, extra=()):
    return subprocess.run(
        [runner_bin, "-m", str(src), "--quantize", str(out),
         "--prune-experts", str(plan_path), *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)


def _router_ne1(path, target_layer):
    """ne[1] of blk.{target_layer}.ffn_gate_inp.weight, read directly from
    the GGUF header — independent proof that a layer's declared expert
    count actually changed on disk, not just that the runner behaved as
    if it had."""
    with open(path, "rb") as f:
        f.read(4)
        struct.unpack("<I", f.read(4))
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode()

        type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip(et)
            elif t == 8:
                rd_str()
            else:
                f.read(type_sizes[t])

        for _ in range(n_kv):
            rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8:
                rd_str()
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip(et)
            else:
                skip(t)
        for _ in range(n_tensors):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            if name == f"blk.{target_layer}.ffn_gate_inp.weight":
                return ne[1]
    return None


def _tensor_ne0(path, tensor_name):
    """ne[0] of a named tensor, read straight from the GGUF header."""
    with open(path, "rb") as f:
        f.read(4)
        struct.unpack("<I", f.read(4))
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode()

        type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip(et)
            elif t == 8:
                rd_str()
            else:
                f.read(type_sizes[t])

        for _ in range(n_kv):
            rd_str()
            skip(struct.unpack("<I", f.read(4))[0])
        for _ in range(n_tensors):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            if name == tensor_name:
                return ne[0]
    return None


def _patch_tensor_dim(src, dst, tensor_name, dim, value):
    """Patch one descriptor dimension without touching its backing bytes."""
    data = bytearray(pathlib.Path(src).read_bytes())
    pos = 8
    n_tensors, n_kv = struct.unpack_from("<QQ", data, pos)
    pos += 16
    type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                  7: 1, 10: 8, 11: 8, 12: 8}

    def rd_str():
        nonlocal pos
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        out = bytes(data[pos:pos + n]).decode()
        pos += n
        return out

    def skip(t):
        nonlocal pos
        if t == 9:
            et = struct.unpack_from("<I", data, pos)[0]
            n = struct.unpack_from("<Q", data, pos + 4)[0]
            pos += 12
            for _ in range(n):
                skip(et)
        elif t == 8:
            rd_str()
        else:
            pos += type_sizes[t]

    for _ in range(n_kv):
        rd_str()
        t = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        skip(t)
    found = False
    for _ in range(n_tensors):
        name = rd_str()
        nd = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims_pos = pos
        pos += 8 * nd + 12  # dimensions, type, data offset
        if name == tensor_name:
            assert 0 <= dim < nd
            struct.pack_into("<Q", data, dims_pos + 8 * dim, value)
            found = True
    assert found, tensor_name
    dst.write_bytes(data)


def test_prune_slices_exp_probs_b_bias_spelling(runner_bin, pruneprobe_bias_model, tmp_path):
    """A pruned file whose selection bias still lists every original expert is
    not loadable: model.c reads `exp_probs_b` with the layer's post-prune
    expert count. Nemotron-3.5-Lightning ships the `.bias` spelling, so the
    pruner must slice that name as well as `.weight`."""
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1], 1: [0, 1]})  # drop the two never-selected experts
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_bias_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _router_ne1(pruned, 0) == 2

    assert _tensor_ne0(pruned, "blk.0.exp_probs_b.bias") == 2, (
        "the selection bias kept all 4 entries beside a 2-expert router")

    # and the file the pruner produced has to load and run
    base_out, _ = _generate(runner_bin, pruneprobe_bias_model)
    pruned_out, pruned_err = _generate(runner_bin, pruned)
    assert pruned_out == base_out, (
        "pruning the two never-selected experts changed output\nstderr: "
        + pruned_err.decode(errors="replace"))


def test_nemotron_prune_loads_nonuniform_layers_deterministically(
        runner_bin, nemotron_prune_model, tmp_path):
    """The Nemotron loader must take each layer's count from its own router.

    The two layers also cover both exp_probs_b spellings. Their zero routers
    make those biases solely responsible for selecting the retained experts,
    so base/pruned byte identity proves the biases were sliced and consumed.
    """
    plan = tmp_path / "plan.json"
    _write_plan(plan, {2: [2, 3], 3: [1, 2, 3]})
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, nemotron_prune_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _router_ne1(pruned, 2) == 2
    assert _router_ne1(pruned, 3) == 3
    assert _tensor_ne0(pruned, "blk.2.exp_probs_b.weight") == 2
    assert _tensor_ne0(pruned, "blk.3.exp_probs_b.bias") == 3

    base_out, _ = _generate(runner_bin, nemotron_prune_model)
    first, first_err = _generate(runner_bin, pruned)
    second, second_err = _generate(runner_bin, pruned)
    assert first == base_out, (
        "coverage-pruning provably selected Nemotron experts changed output\n" +
        first_err.decode(errors="replace"))
    assert second == first, (
        "the non-uniform pruned Nemotron decode was not deterministic\n" +
        second_err.decode(errors="replace"))


def test_pruneprobe_fixture_never_selects_2_or_3(runner_bin, pruneprobe_model, tmp_path):
    # Sanity check on the fixture's own claim, not the pruning tool: confirms
    # the zero-router + exp_probs_b construction actually behaves as the
    # module docstring says before any test relies on it as ground truth.
    trace = tmp_path / "sanity.jsonl"
    env = {**os.environ, "RUNNER_MOE_TRACE": str(trace)}
    proc = subprocess.run(
        [runner_bin, "-m", str(pruneprobe_model), "-p", PROMPT, "-n", str(N_GEN),
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    used = defaultdict(set)
    for line in trace.read_text().splitlines():
        rec = json.loads(line)
        used[rec["layer"]].update(rec["experts"])
    for layer, ids in used.items():
        assert ids == {0, 1}, f"blk.{layer} used {ids}, expected exactly {{0, 1}}"


def test_prune_unused_experts_are_byte_identical(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1], 1: [0, 1]})  # drop 2 and 3 from both layers
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _router_ne1(pruned, 0) == 2 and _router_ne1(pruned, 1) == 2

    base_out, _ = _generate(runner_bin, pruneprobe_model)
    pruned_out, pruned_err = _generate(runner_bin, pruned)
    assert pruned_out == base_out, (
        "pruning experts 2 and 3 (never selected — the fixture's whole "
        "point) changed output\nstderr: " + pruned_err.decode(errors="replace"))


def test_prune_used_expert_changes_output(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [1, 2, 3], 1: [0, 1, 2, 3]})  # drop USED expert 0, only layer 0
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _router_ne1(pruned, 0) == 3
    assert _router_ne1(pruned, 1) == 4  # untouched layer keeps all 4

    base_out, _ = _generate(runner_bin, pruneprobe_model)
    pruned_out, _ = _generate(runner_bin, pruned)
    assert pruned_out != base_out, (
        "pruning expert 0 (always selected, always half the weighted sum) "
        "from blk.0 left output unchanged")


def test_prune_missing_layer_keeps_all_experts(runner_bin, pruneprobe_model, tmp_path):
    # A plan naming only layer 0 must leave layer 1 with every expert.
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1]})
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert _router_ne1(pruned, 0) == 2, "blk.0 was in the plan, should be pruned to 2"
    assert _router_ne1(pruned, 1) == 4, "blk.1 was absent from the plan, should keep all 4"
    # And the model must still load and run (exercises the mixed-geometry
    # load path: one MoE layer with 2 experts, the sibling with 4).
    _generate(runner_bin, pruned)


def test_prune_uniform_updates_expert_count_metadata(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1], 1: [0, 1]})  # uniform: every MoE layer -> 2
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"llama.expert_count 4 -> 2" in proc.stderr, (
        "uniform prune across every MoE layer should rewrite expert_count "
        "metadata (see quantize.c's uniform_prune)\nstderr: " +
        proc.stderr.decode(errors="replace"))


def test_prune_nonuniform_leaves_expert_count_metadata_alone(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1]})  # only layer 0 -> non-uniform vs layer 1's untouched 4
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"expert_count" not in proc.stderr, (
        "non-uniform prune must NOT rewrite the single global expert_count "
        "metadata (it can't honestly describe two different per-layer "
        "counts)\nstderr: " + proc.stderr.decode(errors="replace"))


def test_prune_out_of_range_expert_id_is_rejected(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1, 4]})  # only 4 experts exist: ids 0..3
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode != 0
    assert not pruned.exists()


@pytest.mark.parametrize("bad_id", [1.5, 2147483648])
def test_prune_expert_id_must_fit_a_nonnegative_int(runner_bin,
                                                     pruneprobe_model,
                                                     tmp_path, bad_id):
    """Fractional and wider-than-int JSON numbers must not reach tensor math."""
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, bad_id]})
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode != 0
    assert b"is not a non-negative integer" in proc.stderr
    assert not pruned.exists()


@pytest.mark.parametrize("bad_key", ["layer_0junk", "layer_2147483648"])
def test_prune_layer_key_is_parsed_strictly(runner_bin, pruneprobe_model,
                                             tmp_path, bad_key):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({bad_key: [0, 1]}))
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode != 0
    assert b'is not "layer_N"' in proc.stderr
    assert not pruned.exists()


def test_prune_rejects_expert_tensor_that_disagrees_with_router(
        runner_bin, pruneprobe_model, tmp_path):
    malformed = tmp_path / "mismatched.gguf"
    _patch_tensor_dim(pruneprobe_model, malformed,
                      "blk.0.exp_probs_b.weight", 0, 2)
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 3]})  # 3 exists in the router, not in the bias
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, malformed, pruned, plan)
    assert proc.returncode != 0
    assert b"expert axis" in proc.stderr
    assert not pruned.exists()


def test_prune_plan_for_missing_model_layer_is_rejected(runner_bin, pruneprobe_model, tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, {99: [0, 1]})  # this fixture only has layers 0 and 1
    pruned = tmp_path / "pruned.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan)
    assert proc.returncode != 0
    assert not pruned.exists()


def test_prune_with_quant_also_requantizes_survivors(runner_bin, pruneprobe_model, tmp_path):
    # --prune-experts composes with --quant: survivors get requantized too,
    # not just kept at their original (here F32) type.
    plan = tmp_path / "plan.json"
    _write_plan(plan, {0: [0, 1], 1: [0, 1]})
    pruned = tmp_path / "pruned_q8.gguf"
    proc = _prune(runner_bin, pruneprobe_model, pruned, plan, extra=("--quant", "q8_0"))
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert pruned.stat().st_size < pathlib.Path(pruneprobe_model).stat().st_size
    _generate(runner_bin, pruned)  # must still load and run
