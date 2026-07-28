import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace
import stat

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


def test_completion_request_selects_the_loaded_model_implicitly():
    assert compare_llamacpp.completion_request("hello", 128, False) == {
        "prompt": "hello",
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }


def test_version_probe_resolves_a_relative_executable(tmp_path, monkeypatch):
    if sys.platform == "win32":
        runner = tmp_path / "runner.cmd"
        runner.write_text("@echo off\necho runner test-version\n")
    else:
        runner = tmp_path / "runner"
        runner.write_text("#!/bin/sh\necho 'runner test-version'\n")
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    monkeypatch.chdir(tmp_path)

    assert compare_llamacpp.first_line_version(pathlib.Path(runner.name)) == (
        "runner test-version"
    )


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


def test_first_token_divergence_reports_competing_logprobs():
    runner = {"positions": [
        {"token": " Paris", "logprob": -0.01, "top_logprobs": []},
        {"token": ".", "logprob": -0.02, "top_logprobs": []},
        {"token": " Japan", "logprob": -0.40,
         "top_logprobs": [{"token": " Japan", "logprob": -0.40},
                          {"token": " Russia", "logprob": -0.42}]},
    ]}
    llama = {"positions": [
        {"token": " Paris", "logprob": -0.01, "top_logprobs": []},
        {"token": ".", "logprob": -0.02, "top_logprobs": []},
        {"token": " Russia", "logprob": -0.39,
         "top_logprobs": [{"token": " Russia", "logprob": -0.39},
                          {"token": " Japan", "logprob": -0.41}]},
    ]}

    assert compare_llamacpp.first_token_divergence(runner, llama) == {
        "status": "diverged",
        "position": 2,
        "shared_tokens": 2,
        "runner": runner["positions"][2],
        "llamacpp": llama["positions"][2],
    }


def test_correctness_gate_uses_shared_prefix_and_numeric_delta():
    divergence = {"status": "diverged", "shared_tokens": 55}
    logprobs = {"status": "captured", "max_abs_common_logprob_delta": 0.4}
    assert compare_llamacpp.correctness_gate(divergence, logprobs, 32, 1.0)["status"] == "pass"
    failed = compare_llamacpp.correctness_gate(divergence, logprobs, 64, 0.3)
    assert failed["status"] == "fail"
    assert len(failed["reasons"]) == 2


def test_legacy_completion_logprobs_are_normalized():
    response = {"choices": [{"logprobs": {
        "tokens": ["A"],
        "token_logprobs": [-0.25],
        "top_logprobs": [{"A": -0.25, "B": -1.5}],
    }}]}
    assert compare_llamacpp.legacy_completion_logprobs(response) == {
        "status": "captured",
        "positions": [{
            "token": "A",
            "logprob": -0.25,
            "top_logprobs": [
                {"token": "A", "logprob": -0.25},
                {"token": "B", "logprob": -1.5},
            ],
        }],
    }


def test_completion_logprobs_accept_content_rows():
    response = {"choices": [{"logprobs": {"content": [{
        "token": "A",
        "logprob": -0.25,
        "top_logprobs": [
            {"token": "A", "logprob": -0.25},
            {"token": "B", "logprob": -1.5},
        ],
    }]}}]}
    assert compare_llamacpp.legacy_completion_logprobs(response) == {
        "status": "captured",
        "positions": [{
            "token": "A",
            "logprob": -0.25,
            "top_logprobs": [
                {"token": "A", "logprob": -0.25},
                {"token": "B", "logprob": -1.5},
            ],
        }],
    }
