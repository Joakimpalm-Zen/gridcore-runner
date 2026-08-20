"""The MoE margin-qualified tolerance for the cpu_cuda identity gate.

Dense models keep byte-identity; a top-k MoE routing near-tie that flips under
CPU-vs-CUDA reduction order is tolerated ONLY when it is a genuine coin-flip
inside bar-v2's tie band, all other probes are exact, and the flips are few
enough not to read as a systematic skew. This pins that logic without a GPU.
"""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "cpu_cuda_check",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cpu_cuda_check.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def blk(tokens, tops):
    """A runner logprobs block: chosen tokens + per-position top_logprobs dicts."""
    return {"tokens": list(tokens),
            "logprobs": {"tokens": list(tokens),
                         "token_logprobs": [tops[i][tokens[i]] for i in range(len(tokens))],
                         "top_logprobs": tops}}


def test_identical_streams_are_exact():
    a = blk(["A", "B"], [{"A": -0.1}, {"B": -0.2}])
    assert cc.classify_divergence(a, a)[0] == "exact"


def test_near_tie_flip_is_tolerated():
    # both sides rate the two competitors within the band (0.2 nats <= 0.5)
    cpu = blk(["A", "Z"], [{"A": -0.1, "B": -0.3}, {"Z": -0.1}])
    gpu = blk(["B", "Y"], [{"B": -0.1, "A": -0.3}, {"Y": -0.1}])
    kind, d = cc.classify_divergence(cpu, gpu)
    assert kind == "near_tie" and d["pos"] == 0


def test_confident_divergence_is_real():
    # CPU is sure (0.9 nats gap to GPU's pick) -> a real difference, not a coin-flip
    cpu = blk(["A", "Z"], [{"A": -0.1, "B": -1.0}, {"Z": -0.1}])
    gpu = blk(["B", "Y"], [{"B": -0.1, "A": -1.0}, {"Y": -0.1}])
    assert cc.classify_divergence(cpu, gpu)[0] == "real"


def test_pick_off_top_n_is_real():
    # GPU's pick isn't even on CPU's top-N list -> unmeasured != tied -> real
    cpu = blk(["A"], [{"A": -0.1, "C": -0.3}])
    gpu = blk(["B"], [{"B": -0.1, "A": -0.3}])
    assert cc.classify_divergence(cpu, gpu)[0] == "real"


def test_run_result_thresholds():
    assert cc.run_result(near_tie=0, real=0, total=9) == "pass"
    assert cc.run_result(near_tie=1, real=0, total=9) == "pass_margin_qualified"
    # a real divergence anywhere fails, regardless of near-ties (cond. 4)
    assert cc.run_result(near_tie=1, real=1, total=9) == "fail"
    # near-ties across too many probes read as a skew, not chaos (cond. 3):
    # cap = max(1, int(0.34*9)) = 3, so 3 qualifies but 4 fails
    assert cc.run_result(near_tie=3, real=0, total=9) == "pass_margin_qualified"
    assert cc.run_result(near_tie=4, real=0, total=9) == "fail"
