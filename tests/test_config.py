"""The bot may only dial a valid number that is explicitly on the allowlist."""

import pytest

from caller import config


@pytest.fixture
def target(monkeypatch):
    def set_target(number, allowed):
        monkeypatch.setattr(config, "TARGET_NUMBER", number)
        monkeypatch.setattr(config, "ALLOWED_TARGETS", set(allowed))

    return set_target


def test_allowlisted_number_is_returned(target):
    target("+15555550100", {"+15555550100"})
    assert config.checked_target() == "+15555550100"


def test_number_missing_from_allowlist_is_refused(target):
    target("+15555550100", {"+15555550199"})
    with pytest.raises(RuntimeError, match="not listed in ALLOWED_TARGETS"):
        config.checked_target()


def test_empty_allowlist_refuses_everything(target):
    target("+15555550100", set())
    with pytest.raises(RuntimeError, match="not listed"):
        config.checked_target()


@pytest.mark.parametrize("number", ["", "5555550100", "+1 555 555 0100", "+0123456789", "+1555"])
def test_malformed_number_is_refused_even_if_allowlisted(target, number):
    target(number, {number})
    with pytest.raises(RuntimeError, match="E.164"):
        config.checked_target()


def test_require_reports_missing_variable(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    with pytest.raises(SystemExit, match="SOME_UNSET_KEY"):
        config.require("SOME_UNSET_KEY")
