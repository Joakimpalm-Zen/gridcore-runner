import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ornith_reference", ROOT / "scripts" / "ornith-reference.py"
)
ORNITH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORNITH)


def test_official_9b_layout_contract_accepts_hybrid_pattern():
    metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 8,
        "qwen35.full_attention_interval": 4,
        "tokenizer.ggml.pre": "qwen35",
    }
    tensors = {
        *(f"blk.{block}.attn_q.weight" for block in (3, 7)),
        *(f"blk.{block}.attn_qkv.weight" for block in (0, 1, 2, 4, 5, 6)),
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    }

    report = ORNITH.validate_layout(metadata, tensors)

    assert report["ok"] is True
    assert report["full_attention_blocks"] == [3, 7]
    assert report["linear_attention_blocks"] == [0, 1, 2, 4, 5, 6]


def test_layout_contract_reports_wrong_attention_pattern():
    metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 4,
        "qwen35.full_attention_interval": 4,
        "tokenizer.ggml.pre": "qwen35",
    }
    tensors = {
        "blk.0.attn_q.weight",
        "blk.1.attn_qkv.weight",
        "blk.2.attn_qkv.weight",
        "blk.3.attn_qkv.weight",
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    }

    report = ORNITH.validate_layout(metadata, tensors)

    assert report["ok"] is False
    assert any("full-attention blocks" in error for error in report["errors"])
