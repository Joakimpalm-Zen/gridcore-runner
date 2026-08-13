import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capture_cli_accepts_an_immutable_reference_revision():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "difftok.py"), "--help"],
        text=True, capture_output=True, check=True)
    assert "--ref-revision" in proc.stdout
