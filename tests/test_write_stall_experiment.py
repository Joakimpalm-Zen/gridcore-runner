import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "write_stall", ROOT / "scripts" / "write-stall.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_reads_server_transmit_queue_for_the_exact_connection(tmp_path):
    tcp = tmp_path / "tcp"
    tcp.write_text(
        "  sl  local_address rem_address st tx_queue:rx_queue\n"
        "  1: 0100007F:46CB 0100007F:8898 01 0002B34C:00000000 00\n"
        "  2: 0100007F:46CB 00000000:0000 0A 00000000:00000000 00\n")
    assert MOD.server_tx_queue(tcp, 18123, 34968) == 0x2B34C


def test_write_pressure_requires_far_more_than_the_peer_can_receive():
    assert MOD.queue_under_pressure(512 * 1024)
    assert not MOD.queue_under_pressure(255 * 1024)
