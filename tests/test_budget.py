import json
import pytest
from minicpm_t2l.budget import Budget, BudgetError


def make(tmp_path, confirmed=True):
    (tmp_path / "config").mkdir(); (tmp_path / "budget").mkdir()
    (tmp_path / "config/budget.json").write_text(json.dumps({
        "currency": "USD", "provider": "modal", "project_cap_usd": 20, "confirmed_by_user": confirmed,
        "stage_caps_usd": {"gate0": 10, "other": 15}, "max_single_reservation_usd": 6}))
    return Budget(tmp_path)


def test_refuses_when_unconfirmed(tmp_path):
    with pytest.raises(BudgetError, match="not confirmed"):
        make(tmp_path, confirmed=False).reserve("gate0", 1)


def test_stage_and_single_caps(tmp_path):
    b = make(tmp_path)
    with pytest.raises(BudgetError, match="single-run"):
        b.reserve("gate0", 7)
    r1 = b.reserve("gate0", 6)
    with pytest.raises(BudgetError, match="stage"):
        b.reserve("gate0", 5)
    b.finish(r1, "completed", actual_usd=2.0)
    b.reserve("gate0", 5)  # 2 actual + 5 held = 7 <= 10
    assert b.spent("gate0") == 7.0


def test_project_cap_and_release(tmp_path):
    b = make(tmp_path)
    b.reserve("gate0", 6); b.reserve("other", 6); r = b.reserve("other", 6)
    with pytest.raises(BudgetError, match="project"):
        b.reserve("gate0", 3)
    b.finish(r, "released")
    b.reserve("gate0", 3)
    with pytest.raises(BudgetError, match="already"):
        b.finish(r, "completed")


def test_real_policy_caps():
    p = Budget().policy()
    assert p["project_cap_usd"] == 600 and p["stage_caps_usd"]["gate0"] == 40
