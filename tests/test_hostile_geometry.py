"""Geometry metadata is untrusted input, and the forward pass indexes with it.

A GGUF's header decides loop bounds and buffer offsets long before any tensor
is read, so a value the loader accepts but the buffers cannot hold is a memory
error, not a wrong answer. These cases are all *valid* fixtures with ONE header
field rewritten to a value a hostile (or merely mis-converted) file could
carry; every one of them was a confirmed out-of-bounds access under
`make debug` before the load-time check that now refuses it.
"""
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _patch_u32(src, dst, key, value):
    """Rewrite one U32-typed metadata value in place, leaving everything else.

    Rewriting rather than generating keeps the fixture generators honest: they
    emit models that are supposed to work, and the hostility belongs here.
    """
    b = bytearray(src.read_bytes())
    k = struct.pack("<Q", len(key)) + key.encode()
    i = b.find(k)
    assert i >= 0, f"{key} not found in {src}"
    j = i + len(k)
    kind = struct.unpack_from("<I", b, j)[0]
    assert kind == 4, f"{key} is type {kind}, not U32"
    struct.pack_into("<I", b, j + 4, value)
    dst.write_bytes(bytes(b))
    return dst


def _patch_tensor_type(src, dst, name, value):
    """Rewrite one tensor descriptor's ggml type, leaving its shape and data."""
    b = bytearray(src.read_bytes())
    k = struct.pack("<Q", len(name)) + name.encode()
    i = b.find(k)
    assert i >= 0, f"{name} not found in {src}"
    j = i + len(k)
    n_dims = struct.unpack_from("<I", b, j)[0]
    j += 4 + 8 * n_dims
    struct.pack_into("<I", b, j, value)
    dst.write_bytes(bytes(b))
    return dst


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _run(runner_bin, model):
    return subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "-b", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )


def test_gemma4_rope_dims_wider_than_the_head_are_refused(runner_bin, tmp_path):
    """rope rotates pairs INSIDE a head; a wider rope dim writes past q/k.

    `rope.dimension_count_swa` is per-layer geometry the general gate never
    saw: it bounds `rope.dimension_count` against `attention.key_length`, but
    a sliding layer rotates `dimension_count_swa` dims inside a
    `key_length_swa`-wide head. With 128 rotated dims in a 16-wide head the
    last head's rope wrote 111 floats past the end of m->q (ASan:
    heap-buffer-overflow in rope_apply).
    """
    good = tmp_path / "g4h.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--gemma4-hetero", str(good)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    bad = _patch_u32(good, tmp_path / "g4h-rope.gguf",
                     "gemma4.rope.dimension_count_swa", 128)

    assert _run(runner_bin, good).returncode == 0, "the unmodified fixture must run"
    proc = _run(runner_bin, bad)
    assert proc.returncode != 0, "a rope dim wider than the head must be refused"
    err = proc.stderr.decode(errors="replace")
    assert "invalid gemma4 per-layer geometry" in err


def test_shared_expert_width_beyond_its_tensor_is_refused(runner_bin, tmp_path):
    """The shared-expert FFN width drove matvec but was never shape-checked.

    `expert_shared_feed_forward_length` decides how many rows of
    ffn_gate_shexp/ffn_up_shexp the FFN reads, and how long a row of
    ffn_down_shexp is. Every other projection in the layer is validated
    against the geometry it will be driven with; these three were not, so a
    width past the tensor read straight off the end of the mapping (ASan:
    SEGV in vec_dot, reached from the shared-expert matvec).
    """
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp_path / "moe")],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    good = tmp_path / "moe.shexp.gguf"
    bad = _patch_u32(good, tmp_path / "moe-shexp-wide.gguf",
                     "llama.expert_shared_feed_forward_length", 4096)

    assert _run(runner_bin, good).returncode == 0, "the unmodified fixture must run"
    proc = _run(runner_bin, bad)
    assert proc.returncode > 0, "a shared-expert width past its tensor must be " \
                                "refused at load, not crash mid-forward"
    err = proc.stderr.decode(errors="replace")
    assert "ffn_gate_shexp" in err


def test_weight_of_an_unsupported_type_is_refused(runner_bin, tmp_path):
    """A weight the engine cannot decode must refuse, not read stale stack.

    gguf_open leaves an unsupported tensor type for its user to check ("checked
    at use time"), and need_tensor does check it — but the OPTIONAL weights
    (gemma4's V, apertus's ffn_gate, the fused expert banks) come through
    opt_tensor, which does not. dequant_block ignores a type it does not know,
    so such a weight dequantized to whatever the scratch buffer already held
    and the model produced fluent output from uninitialized memory: the exact
    "plausible but silently wrong" outcome the architecture allowlist exists
    to prevent.
    """
    good = tmp_path / "g4h.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--gemma4-hetero", str(good)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    # blk.0 is a sliding layer, so it carries the optional attn_v tensor.
    bad = _patch_tensor_type(good, tmp_path / "g4h-type.gguf",
                             "blk.0.attn_v.weight", 44)

    assert _run(runner_bin, good).returncode == 0, "the unmodified fixture must run"
    proc = _run(runner_bin, bad)
    assert proc.returncode != 0, "a weight of an undecodable type must be refused"
    err = proc.stderr.decode(errors="replace")
    assert "unsupported type" in err


def test_qwen35_recurrent_geometry_beyond_its_tensors_is_refused(runner_bin, tmp_path):
    """The Gated DeltaNet geometry keys index tensors nothing checked.

    ssm.inner_size / state_size / group_count / time_step_rank decide the
    convolution width, the per-head fan-out and how many entries of ssm_dt and
    ssm_a the recurrence reads — and the qwen35 layer path deliberately skips
    the llama-family shape checks (its Q is fused with a gate), which left its
    tensors validated by nothing at all. Doubling the inner size and the head
    count (a pair that still satisfies the internal ratio check) read past
    ssm_dt's converted buffer: ASan heap-buffer-overflow in qwen35_linear,
    0 bytes after a 16-byte region.
    """
    good = tmp_path / "orn.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-ornith.py", str(good)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    wide = _patch_u32(good, tmp_path / "orn-inner.gguf",
                      "qwen35.ssm.inner_size", 64)
    bad = _patch_u32(wide, tmp_path / "orn-heads.gguf",
                     "qwen35.ssm.time_step_rank", 8)

    assert _run(runner_bin, good).returncode == 0, "the unmodified fixture must run"
    proc = _run(runner_bin, bad)
    assert proc.returncode > 0, "recurrent geometry past its tensors must be " \
                                "refused at load, not read past them mid-forward"
    err = proc.stderr.decode(errors="replace")
    assert "attn_qkv" in err
