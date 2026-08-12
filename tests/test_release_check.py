import argparse
import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release", ROOT / "scripts/check-release.py"
)
check_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_release)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def good_args(tmp_path):
    return argparse.Namespace(
        tag="v0.1.3-alpha",
        binary=tmp_path / "runner",
        readme=write(
            tmp_path / "README.md",
            "Public alpha (`0.1.3-alpha`)\n"
            "./runner --version   # -> runner 0.1.3-alpha\n",
        ),
        changelog=write(tmp_path / "CHANGELOG.md", "## v0.1.3-alpha\n"),
        build_info=write(
            tmp_path / "BUILD-INFO.txt",
            "runner 0.1.3-alpha\n"
            "tag:        v0.1.3-alpha\n"
            "commit:     abc123\n",
        ),
        release_workflow=write(
            tmp_path / "release.yml", "# release v0.1.3-alpha\n"
        ),
        python_pyproject=write(
            tmp_path / "pyproject.toml",
            '[project]\nversion = "0.1.3a0"\n',
        ),
        commit="abc123",
        current_docs=[],
        compat_reports=report_dir(tmp_path, "0.1.3-alpha-2026-01-01.json"),
    )


def report_dir(tmp_path, *names):
    """A compat-report directory holding EXACTLY the named dated reports.

    Cleared on entry: the helper is called twice per test (once by good_args,
    once by the test itself), and a leftover report from the first call would
    satisfy the gate the second call is trying to trip.
    """
    d = tmp_path / "compat-reports"
    d.mkdir(exist_ok=True)
    for stale in d.iterdir():
        stale.unlink()
    for name in names:
        write(d / name, "{}\n")
    return d


def test_release_check_accepts_consistent_artifacts(monkeypatch, tmp_path):
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert check_release.check(args)


def test_release_check_rejects_stale_readme(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.readme.write_text(
        args.readme.read_text() + "old example 0.1.2-alpha\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "stale release string '0.1.2-alpha'" in capsys.readouterr().err


def test_release_check_rejects_build_commit_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.commit = "different"
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "BUILD-INFO commit line is inconsistent" in capsys.readouterr().err


def test_release_check_rejects_python_version_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.python_pyproject.write_text(
        '[project]\nversion = "0.1.2a0"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "Python package version" in capsys.readouterr().err


def test_release_check_rejects_stale_current_document(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.current_docs = [write(tmp_path / "SECURITY.md", "public 0.1.2-alpha\n")]
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "current document" in capsys.readouterr().err


# The ledger is only credible if every release ships one. The reports existed
# from 2026-08-13 but nothing made producing them part of cutting a release,
# so the habit depended on somebody remembering.

def test_release_check_requires_a_compat_report_for_this_release(
        monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(tmp_path, "0.1.2-alpha-2026-01-01.json")
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "compat report" in capsys.readouterr().err


def test_release_check_accepts_any_dated_report_for_this_release(
        monkeypatch, tmp_path):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(
        tmp_path, "0.1.2-alpha-2026-01-01.json", "0.1.3-alpha-2026-02-09.json"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert check_release.check(args)


def test_release_check_skips_the_report_gate_when_no_directory_is_given(
        monkeypatch, tmp_path):
    """An older checkout, or a caller that does not pass the flag, still
    releases: the gate binds the process, it does not break the tool."""
    args = good_args(tmp_path)
    args.compat_reports = None
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert check_release.check(args)
