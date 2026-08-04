"""Guards for the two harness bugs that produced false engine bugs.

Both were found by the 2026-08-02 shelf pass (recorded in the CHANGELOG).
Neither was an engine defect; both would have been published as one.
"""
import importlib.util
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "stress_models", ROOT / "scripts" / "stress-models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = load()


def test_identity_is_a_prefix_test_not_string_equality():
    """The legs may run different token budgets, so equality is the wrong test.

    The host leg gets a smaller budget for a model larger than RAM. Comparing
    truncated character slices reported a mismatch on the trailing newline
    alone and flagged three Gemma models as CPU!=GPU when each GPU text was a
    clean continuation of its CPU text.
    """
    cpu = "The capital of France is Paris.\nThe capital of Germany is\n"
    gpu = ("The capital of France is Paris.\nThe capital of Germany is Berlin."
           "\nThe capital of Italy is\n")

    assert sm.same_greedy_output(gpu, cpu, gpu_gen=16, cpu_gen=8)
    # the naive comparison this replaced
    assert gpu[:len(cpu)] != cpu


def test_identity_still_catches_a_real_divergence():
    cpu = "The capital of France is Paris."
    gpu = "The capital of France is Berlin."
    assert not sm.same_greedy_output(gpu, cpu, gpu_gen=8, cpu_gen=8)


def test_identity_handles_the_device_leg_being_the_shorter_one():
    gpu = "The capital of France is"
    cpu = "The capital of France is Paris."
    assert sm.same_greedy_output(gpu, cpu, gpu_gen=4, cpu_gen=8)


def test_empty_output_is_never_reported_as_agreement():
    assert not sm.same_greedy_output("anything", "", gpu_gen=8, cpu_gen=8)


def test_cpu_only_auto_run_is_not_reported_as_cpu_cuda_identity():
    gpu = {"text": "same", "split": None, "faults": []}
    cpu = {"text": "same", "split": None, "faults": []}
    assert sm.cpu_cuda_identity(gpu, cpu, gpu_gen=8, cpu_gen=8) is None


def test_real_gpu_run_can_prove_cpu_cuda_identity():
    gpu = {"text": "same", "split": "gpu-split: G=2/2", "faults": []}
    cpu = {"text": "same", "split": None, "faults": []}
    assert sm.cpu_cuda_identity(gpu, cpu, gpu_gen=8, cpu_gen=8) is True


def _gguf(path, kv):
    """Minimal GGUF v3 header with string/uint32 KV entries and no tensors."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
    out += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        out += s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + s(value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    path.write_bytes(out)
    return path


def test_moe_detection_reads_metadata_not_the_split_banner(tmp_path):
    """Detection must not depend on the flag it is deciding.

    The first version looked for "expert" in the split banner, which only
    appears once --cpu-moe is already in use. That circularity silently skipped
    the placement sweep for every MoE model, including gpt-oss.
    """
    moe = _gguf(tmp_path / "moe.gguf",
                {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 128})
    dense = _gguf(tmp_path / "dense.gguf",
                  {"general.architecture": "llama", "llama.block_count": 32})

    assert sm.is_moe(moe) is True
    assert sm.is_moe(dense) is False


def test_moe_detection_treats_zero_experts_as_dense(tmp_path):
    zero = _gguf(tmp_path / "zero.gguf",
                 {"general.architecture": "llama", "llama.expert_count": 0})
    assert sm.is_moe(zero) is False


def test_moe_detection_survives_a_file_that_is_not_a_gguf(tmp_path):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert sm.is_moe(junk) is False


def test_degenerate_output_is_an_observation_not_a_fault():
    """Raw completions are not a quality signal for thinking-tuned families.

    This repo already paid for that mistake once with a false "gemma4 Q4_0 bug"
    and a wrong catalog refusal, so degenerate() must stay conservative.
    """
    assert sm.degenerate("") == "empty output"
    assert sm.degenerate("a a a a a a a a a") is not None
    assert sm.degenerate("Paris is the capital and largest city of France") is None


def test_machine_resources_come_from_runner_caps():
    ram, vram = sm.parse_resource_caps(
        '{"ram_bytes":268000000000,"gpu":{"vram_bytes":25000000000}}')
    assert ram == 268.0
    assert vram == 25.0


def test_machine_resources_tolerate_cpu_only_or_bad_caps():
    assert sm.parse_resource_caps('{"ram_bytes":16000000000,"gpu":null}') == (16.0, 0.0)
    assert sm.parse_resource_caps("not json") == (None, None)
