import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

from scripts import compare_llamacpp


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_llamacpp_comparison_fixture_report(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/compare_llamacpp.py",
            "--fixture",
            "--out-dir",
            tmp_path,
            "--prompt",
            "fixture prompt",
            "--ctx",
            "128",
            "--tokens",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["status"] == "fixture"

    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["schema_version"] == "gridcore.runner.llamacpp-comparison.v1"
    assert report["status"] == "fixture"
    assert report["real_results"] == "pending"
    assert report["settings"]["prompt"] == "fixture prompt"

    # The readable report must carry the same provenance/settings categories
    # as the machine-readable result, even when fixture values are pending.

    md = (tmp_path / "comparison.md").read_text()
    assert "Runner vs llama.cpp comparison" in md
    assert "Runner commit:" in md
    assert "llama.cpp commit:" in md
    assert "Quantization:" in md
    assert "## Hardware and driver" in md
    assert "## VRAM" in md
    assert "## Generated output" in md
    assert "comparison.json" in md
    assert "Real Qwen3/MoE GPU results are pending" in md


def test_vram_delta_is_reported_per_device():
    before = [{"name": "GPU 0", "memory_used_mib": 100}]
    after = [{"name": "GPU 0", "memory_used_mib": 356}]
    assert compare_llamacpp.vram_delta(before, after) == [
        {"device": 0, "name": "GPU 0", "used_delta_mib": 256}
    ]
    assert compare_llamacpp.vram_delta(None, after) is None


def test_nvidia_snapshot_tolerates_unavailable_mig_memory(monkeypatch):
    output = (
        "NVIDIA RTX PRO 6000, 580.65.06, [Insufficient Permissions], "
        "[Insufficient Permissions], 24192\n"
    )
    monkeypatch.setattr(
        compare_llamacpp,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    snapshot = compare_llamacpp.nvidia_snapshot()
    assert snapshot == [{
        "name": "NVIDIA RTX PRO 6000",
        "driver_version": "580.65.06",
        "memory_used_mib": None,
        "memory_free_mib": None,
        "memory_total_mib": 24192,
    }]
    assert compare_llamacpp.vram_delta(snapshot, snapshot) is None


def test_runtime_commands_allow_the_requested_generation_length(tmp_path):
    args = SimpleNamespace(
        runner=tmp_path / "runner",
        llamacpp=tmp_path / "llama-server",
        model=tmp_path / "model.gguf",
        ctx=4096,
        tokens=128,
        runner_gpu="auto",
        llamacpp_gpu_layers=-1,
        llamacpp_arg=None,
    )

    runner, _ = compare_llamacpp.runtime_commands(args, 8000, 8001)
    assert runner[-4:] == ["--gpu", "auto", "-n", "128"]


def test_top_logprob_comparison_quantifies_common_token_deltas():
    runner = {
        "status": "captured",
        "positions": [{
            "token": "A",
            "logprob": -0.2,
            "top_logprobs": [
                {"token": "A", "logprob": -0.2},
                {"token": "B", "logprob": -1.0},
            ],
        }],
    }
    llama = {
        "status": "captured",
        "positions": [{
            "token": "A",
            "logprob": -0.3,
            "top_logprobs": [
                {"token": "A", "logprob": -0.3},
                {"token": "B", "logprob": -1.4},
            ],
        }],
    }
    result = compare_llamacpp.compare_top_logprobs(runner, llama)
    assert result["status"] == "captured"
    assert result["positions_compared"] == 1
    assert result["max_abs_common_logprob_delta"] == 0.4
    row = result["positions"][0]
    assert row["chosen_logprob_delta"] == 0.1
    assert row["common_top_logprobs"] == [
        {
            "token": "A",
            "runner_logprob": -0.2,
            "llamacpp_logprob": -0.3,
            "abs_delta": 0.1,
        },
        {
            "token": "B",
            "runner_logprob": -1.0,
            "llamacpp_logprob": -1.4,
            "abs_delta": 0.4,
        },
    ]
