"""Safety and threshold tests for the canonical path-B release gates."""
import numpy as np

from conftest import REPO_ROOT  # noqa: F401
from demo.server import board_checks, board_runner


def _pass(name):
    return {"name": name, "pass": True, "value": "ok", "detail": "ok"}


def test_two_channel_ebr_is_mandatory():
    assert board_checks.check_ebr_values((8.0, 8.1))["pass"] is True
    assert board_checks.check_ebr_values((8.0, 7.99))["pass"] is False
    assert board_checks.check_ebr_values(None)["pass"] is False


def test_error_std_uses_per_channel_baseline_and_lower_is_improvement():
    low = tuple(board_checks.ERROR_STD_BASELINE * 0.90)
    assert board_checks.check_error_std_values(low)["pass"] is True
    degraded = tuple(board_checks.ERROR_STD_BASELINE * 1.021)
    assert board_checks.check_error_std_values(degraded)["pass"] is False


def test_no_occupancy_confirmation_never_touches_board(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("board must not be touched")

    monkeypatch.setattr(board_runner, "probe_board", forbidden)
    result = board_checks.all_checks(confirmed_idle=False)
    assert result["all_pass"] is False
    assert len(result["checks"]) == 4
    assert all(item.get("blocked") for item in result["checks"])


def test_all_four_checks_are_required(monkeypatch):
    monkeypatch.setattr(
        board_checks, "preflight",
        lambda *a, **k: ([_pass("idle"), _pass("calib")], "fresh.json"))
    monkeypatch.setattr(
        board_checks, "check_evb",
        lambda: (_pass("ebr"), _pass("error_std")))
    monkeypatch.setattr(board_checks, "check_canary", lambda: _pass("canary"))
    monkeypatch.setattr(board_checks, "check_minirun", lambda *a, **k: _pass("mini"))
    result = board_checks.all_checks(
        model_name="model10", images=np.zeros((1, 3, 64, 64)),
        labels=np.zeros(1), confirmed_idle=True)
    assert result["all_pass"] is True
    assert [item["name"] for item in result["checks"]] == [
        "ebr", "error_std", "canary", "mini"]


def test_failed_evb_blocks_heavier_workloads(monkeypatch):
    monkeypatch.setattr(
        board_checks, "preflight",
        lambda *a, **k: ([_pass("idle"), _pass("calib")], "fresh.json"))
    failed = {"name": "ebr", "pass": False, "value": "bad", "detail": "bad"}
    monkeypatch.setattr(
        board_checks, "check_evb", lambda: (failed, _pass("error_std")))
    monkeypatch.setattr(
        board_checks, "check_canary",
        lambda: (_ for _ in ()).throw(AssertionError("must be blocked")))
    result = board_checks.all_checks(confirmed_idle=True)
    assert result["all_pass"] is False
    assert result["checks"][2]["blocked"] is True


def test_evb_parser_reads_both_channels(monkeypatch):
    sample = "error_std: [ 4.694 4.473 ]\nebr: [ 9.71 9.82 ]\n"
    monkeypatch.setattr(board_runner, "ssh_sudo_run", lambda *a, **k: sample)
    result = board_runner.run_evb()
    assert result["ebr"] == (9.71, 9.82)
    assert result["error_std"] == (4.694, 4.473)
