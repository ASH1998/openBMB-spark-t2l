from minicpm_t2l import pilot_report as pr
from minicpm_t2l.tasks import TASKS


def _run(tps, n_gold=1):
    return {"scores": {"parse_fail_rate": 0.0},
            "docs": [{"id": f"d{i}", "tp": tp, "n_pred": 1, "n_gold": n_gold, "parsed": True} for i, tp in enumerate(tps)]}


GOOD, BAD = [1] * 40, [0] * 40
HALF = [1, 0] * 20


def _fake(p03_4x=HALF, mt=GOOD):
    pd = {}
    for t in TASKS:
        for m in ("ocr", "v47"):
            for a in pr.BASE_ARMS:
                pd[f"gate0/runs_a10g/{m}/{a}/{t}"] = _run(BAD)
            pd[f"gate0/runs_a10g/{m}/loraB/{t}"] = _run(GOOD)
        for a in pr.BASE_ARMS:
            pd[f"pilots/p03_ocr4x/{a}/{t}"] = _run(p03_4x)
        pd[f"pilots/p01_multitask/eval/{t}"] = _run(mt)
    return pd


def test_p03_wins_and_shift():
    rep = pr.p03(_fake())
    assert rep["n_tasks"] == 20 and rep["loraB16x_wins_vs_best4x"] == 20 and rep["headroom_holds"]
    assert rep["tasks_where_4x_baseline_significantly_higher"] == 20
    rep = pr.p03(_fake(p03_4x=GOOD))
    assert rep["loraB16x_wins_vs_best4x"] == 0 and not rep["headroom_holds"]
    assert "| cord_menu |" in pr.markdown("p03", rep)


def test_p01_retention_and_reading():
    rep = pr.p01(_fake(mt=GOOD))
    assert rep["retention"] == 1.0 and rep["mt_wins_vs_best_baseline"] == 20 and rep["reading"] == "strong comparison 3"
    rep = pr.p01(_fake(mt=BAD))
    assert rep["retention"] == 0.0 and rep["mt_vs_loraB"]["worse"] == 20 and rep["reading"] == "tasks interfere"
    assert "| sroie_all |" in pr.markdown("p01", rep)


def test_misaligned_docs_raise():
    a, b = _run(GOOD), _run(GOOD)
    b["docs"][0]["id"] = "other"
    try:
        pr.compare(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
