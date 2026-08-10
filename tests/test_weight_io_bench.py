"""Tests for the weight I/O path benchmark.

The failure mode that matters here is a confidently wrong number rather than a
crash, so these cover offset generation and the reported arithmetic. Absolute
bandwidths are machine-dependent and are not asserted.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "weight-io-bench.py"

spec = importlib.util.spec_from_file_location("weight_io_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(os.urandom(4 * 1024 * 1024))
    return path


def test_offsets_are_page_aligned_and_in_bounds():
    size = 64 * 1024 * 1024
    slice_bytes = 1024 * 1024
    got = bench.offsets(size, slice_bytes, 200, 104729)
    assert len(got) == 200
    for offset in got:
        assert offset % bench.page_size() == 0
        assert 0 <= offset
        assert offset + slice_bytes <= size


def test_offsets_do_not_repeat_while_positions_remain():
    # A stride sharing a factor with the position count revisits a handful of
    # offsets and would measure a warm cache instead of the device.
    page = bench.page_size()
    got = bench.offsets(80 * page, 4 * page, 30, 8)
    assert len(set(got)) == len(got)


def test_offsets_refuse_a_file_smaller_than_one_slice():
    with pytest.raises(SystemExit):
        bench.offsets(1024, 8 * 1024 * 1024, 4, 3)


def test_read_path_reports_consistent_totals(sample):
    result = bench.measure_read(str(sample), 64 * 1024, 5, 1, 104729, 1)
    assert result["samples"] == 5
    assert result["total_bytes"] == 5 * 64 * 1024
    assert result["min_bytes_per_second"] <= result["p50_bytes_per_second"]
    assert result["p50_bytes_per_second"] <= result["max_bytes_per_second"]
    assert result["total_seconds"] > 0


def test_read_path_splits_a_slice_into_parts(sample):
    result = bench.measure_read(str(sample), 96 * 1024, 3, 0, 104729, 3)
    assert result["parts_per_slice"] == 3
    assert result["total_bytes"] == 3 * 96 * 1024


def test_read_path_rejects_indivisible_part_counts(sample):
    with pytest.raises(SystemExit):
        bench.measure_read(str(sample), 100 * 1024, 2, 0, 104729, 3)


def test_fault_path_reports_page_count(sample):
    slice_bytes = 64 * bench.page_size()
    result = bench.measure_faults(str(sample), slice_bytes, 3, 0, 104729)
    assert result["pages_per_slice"] == 64
    assert result["samples"] == 3


def test_cli_emits_both_paths_and_their_ratio(sample, tmp_path):
    out = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(sample),
         "--slice-bytes", str(64 * 1024), "--samples", "4", "--warmup", "1",
         "--out", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["file_bytes"] == 4 * 1024 * 1024
    assert report["read_path"]["samples"] == 4
    assert report["fault_path"]["samples"] == 4
    expected = (report["read_path"]["p50_bytes_per_second"] /
                report["fault_path"]["p50_bytes_per_second"])
    assert report["read_over_fault_p50_ratio"] == pytest.approx(expected)
    assert "cached_second_pass_suspected" in report["cache_probe"]


def test_cli_can_skip_the_fault_path(sample):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(sample),
         "--slice-bytes", str(64 * 1024), "--samples", "2", "--warmup", "0",
         "--skip-faults"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "fault_path" not in report
    assert "read_over_fault_p50_ratio" not in report


def test_cli_refuses_a_slice_smaller_than_a_page(sample):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(sample),
         "--slice-bytes", "64", "--samples", "2"],
        capture_output=True, text=True)
    assert result.returncode != 0
