"""scripts/verify-gguf.py: structural completeness check for a GGUF file.

Its whole job is to say TRUNCATED before the loader does, so its block-size
table has to agree with the loader's. It is an independent transcription of
src/quants.c, which means it can drift from it silently — and a table that
UNDER-counts bytes reports a short file COMPLETE, which is the one answer this
tool must never give.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify-gguf.py"
SPEC = importlib.util.spec_from_file_location("verify_gguf", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

# ggml type id -> the block struct src/quants.c implements it with. Q8_1 and
# Q8_K have no runner struct (they exist only as intermediate forms), so the
# table's entries for them are unverifiable here and left out.
BLOCK_STRUCT = {
    2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
    10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl",
    21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m", 39: "mxfp4",
}


def declared_block_bytes():
    """Block sizes as src/quants.c declares them next to each typedef."""
    source = (ROOT / "src" / "quants.c").read_text()
    return {name: int(size) for name, size in re.findall(
        r"\}\s*block_([A-Za-z0-9_]+);\s*//\s*(\d+)", source)}


@pytest.mark.parametrize("type_id,struct", sorted(BLOCK_STRUCT.items()))
def test_block_sizes_match_the_engines_own_layouts(type_id, struct):
    declared = declared_block_bytes()
    assert struct in declared, f"src/quants.c no longer declares block_{struct}"
    assert type_id in MOD.GGML_TYPES, (
        f"{MOD.TYPE_NAMES.get(type_id, type_id)} is absent from the table, so "
        f"verify-gguf refuses every file containing one")
    assert MOD.GGML_TYPES[type_id][1] == declared[struct]
def test_a_tensorless_gguf_is_verified_rather_than_crashing():
    """make-vocab-fixture.py writes n_tensors = 0 on purpose and src/gguf.c
    accepts it. `max(tensors, ...)` raised ValueError on those, and the
    traceback's exit 1 reads to a caller as TRUNCATED."""
    fixture = ROOT / "tests" / "fixtures" / "vocab-spm.gguf"
    if not fixture.is_file():
        pytest.skip(f"{fixture} not present")

    proc = subprocess.run([sys.executable, str(SCRIPT), str(fixture)],
                          capture_output=True, text=True, timeout=60)

    assert proc.returncode == 0, proc.stderr
    assert "COMPLETE" in proc.stdout
