"""`--fit`: what a GGUF would cost on this machine, from its header alone.

The feature exists to answer "should I download this" before the bandwidth is
spent, so the load-bearing property is that it works on a file whose data
section is ABSENT — and, just as importantly, that making it work did not
weaken the normal loader, which refuses a file whose data section does not
cover the tensors it declares. That refusal is a safety property: it is what
stops a truncated or crafted file from pointing a tensor outside the mapping.
Both halves are asserted here.
"""
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test.gguf"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def run(runner_bin, *args):
    return subprocess.run([runner_bin, *args], cwd=ROOT, timeout=180,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture(scope="module")
def header_only(tmp_path_factory):
    """The first 1 MiB of the toy fixture: header present, data section not."""
    if not FIXTURE.exists():
        pytest.skip("test.gguf not generated")
    whole = FIXTURE.read_bytes()
    cut = tmp_path_factory.mktemp("fit") / "head.gguf"
    # Enough for the toy fixture's metadata and tensor descriptors, far less
    # than its data. If this ever stops being a truncation, the test is void.
    n = min(len(whole) - 1, 1 << 20)
    cut.write_bytes(whole[:n])
    assert cut.stat().st_size < FIXTURE.stat().st_size
    return cut


def test_fit_reports_on_a_whole_file(runner_bin):
    p = run(runner_bin, "--fit", str(FIXTURE))
    out = p.stdout.decode()
    assert p.returncode == 0, p.stderr.decode()
    assert "weights" in out and "kv cache" in out and "verdict" in out


def test_fit_works_without_a_data_section(runner_bin, header_only):
    """The whole point: an answer from a header-only download."""
    p = run(runner_bin, "--fit", str(header_only))
    assert p.returncode == 0, p.stderr.decode()
    assert "verdict" in p.stdout.decode()


def test_header_only_sizes_match_the_whole_file(runner_bin, header_only):
    """A truncated file must not report a truncated model.

    `weights` comes from the tensor descriptors, not from how many bytes are
    actually present — so a 1 MiB slice of a larger file must still report the
    larger file's size. Getting this wrong would make every truncated header
    look like it fits.
    """
    def weights(path):
        out = run(runner_bin, "--fit", str(path)).stdout.decode()
        m = re.search(r"weights\s+([\d.]+) GiB", out)
        assert m, out
        return m.group(1)

    assert weights(header_only) == weights(FIXTURE)


def test_normal_load_still_refuses_the_truncated_file(runner_bin, header_only):
    """The safety property the header-tolerant path must not have weakened.

    gguf_open() rejects a file whose data section does not cover its declared
    tensors. `--fit` reads through a SEPARATE path; if it had instead relaxed
    the shared one, this would start loading and the refusal would be gone.
    """
    p = run(runner_bin, "-m", str(header_only), "-p", "hi", "-n", "1",
            "--temp", "0")
    assert p.returncode != 0, "the loader accepted a file with no data section"
    assert b"error" in p.stderr.lower()


def test_context_changes_the_kv_line(runner_bin):
    """-c feeds the estimate, so the answer tracks how it would be run."""
    small = run(runner_bin, "--fit", str(FIXTURE), "-c", "512").stdout.decode()
    large = run(runner_bin, "--fit", str(FIXTURE), "-c", "8192").stdout.decode()
    assert "ctx 512" in small and "ctx 8192" in large
    assert small != large


def test_fit_does_not_fetch(runner_bin):
    """A URL is not a path, and the runner does not speak HTTP.

    Recorded as a test because "just add a download" is the obvious next
    request and the answer is deliberately no — the README documents a ranged
    read instead.
    """
    p = run(runner_bin, "--fit", "https://example.invalid/model.gguf")
    assert p.returncode != 0
    assert b"http" not in p.stdout.lower()
