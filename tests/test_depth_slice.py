"""gguf-depth-slice.py: drop whole layers from a GGUF, keep semantics.

The muse fixture is the hard case on purpose: its per-layer bool
sliding_window_pattern array must be filtered to the survivors (a slice
that kept the old array would misassign SWA/NoPE roles to every layer
after the first cut), and its tensors must renumber densely or the
loader's need_tensor sweep fails.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _slice(src, dst, drop):
    subprocess.run([sys.executable, ROOT / "scripts/gguf-depth-slice.py",
                    str(src), str(dst), drop],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def _gen(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def _tensor_bytes(path):
    """name -> raw data bytes, resolved through the GGUF header."""
    import struct
    b = pathlib.Path(path).read_bytes()
    p = 4
    _, nt, nk = struct.unpack_from("<IQQ", b, p)
    p += 20
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def st():
        nonlocal p
        n, = struct.unpack_from("<Q", b, p)
        p += 8 + n
        return b[p - n:p]

    align = 32
    for _ in range(nk):
        k = st().decode()
        t, = struct.unpack_from("<I", b, p)
        p += 4
        if t == 8:
            st()
        elif t == 9:
            at, n = struct.unpack_from("<IQ", b, p)
            p += 12
            if at == 8:
                for _ in range(n):
                    st()
            else:
                p += n * sizes[at]
        else:
            if k == "general.alignment":
                align, = struct.unpack_from("<I", b, p)
            p += sizes[t]
    table = []
    for _ in range(nt):
        name = st().decode()
        nd, = struct.unpack_from("<I", b, p)
        p += 4 + 8 * nd
        tt, off = struct.unpack_from("<IQ", b, p)
        p += 12
        table.append((name, off))
    ds = (p + align - 1) & ~(align - 1)
    by_off = sorted(table, key=lambda x: x[1])
    out = {}
    for i, (name, off) in enumerate(by_off):
        end = by_off[i + 1][1] if i + 1 < len(by_off) else len(b) - ds
        out[name] = b[ds + off:ds + end]
    return out


def test_sliced_model_loads_and_decodes(runner_bin, tmp_path):
    base = tmp_path / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--muse-glimmer", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    sliced = tmp_path / "muse-keep6.gguf"
    # drop one sliding layer and one full/NoPE layer: the surviving pattern
    # array is irregular, which is exactly what the filter must preserve
    _slice(base, sliced, "5,7")
    out_base = _gen(runner_bin, base)
    out_sliced = _gen(runner_bin, sliced)
    assert out_sliced  # loads, decodes
    assert out_base != out_sliced  # the cut changed the function
    assert out_sliced == _gen(runner_bin, sliced)  # deterministically
    # a loadable-but-corrupt slice decodes deterministically too (this file
    # shipped one: data_start computed before the tensor table), so the gate
    # that matters is byte identity of every surviving tensor
    src = _tensor_bytes(base)
    dst = _tensor_bytes(sliced)
    assert dst["blk.0.attn_q.weight"] == src["blk.0.attn_q.weight"]
    assert dst["blk.5.ffn_down.weight"] == src["blk.6.ffn_down.weight"]
    assert dst["token_embd.weight"] == src["token_embd.weight"]
    assert dst["output.weight"] == src["output.weight"]


def test_slice_refuses_mtp_models(tmp_path):
    base = tmp_path / "mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    proc = subprocess.run([sys.executable, ROOT / "scripts/gguf-depth-slice.py",
                           str(base), str(tmp_path / "out.gguf"), "0"],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode != 0
    assert b"NextN/MTP" in proc.stderr + proc.stdout
