from minicpm_t2l import gate1_report as gr


def _run(tps):
    return {"scores": {}, "docs": [{"id": f"d{i}", "tp": tp, "n_pred": 1, "n_gold": 1, "parsed": True} for i, tp in enumerate(tps)]}


def _pd(tasks, emit_good=True):
    pd = {}
    for t in tasks:
        for a in gr.ARMS:
            good = a == "oracle" or (a.startswith("emit") and emit_good)
            pd[f"gate1/eval/{a}/{t}"] = _run([1] * 30 if good else [1, 0] * 15)
    return pd


TASKS = [f"syn_x_{i}" for i in range(6)] + [f"pub_{i}" for i in range(4)]


def test_primary_and_groups():
    rep = gr.analyse(_pd(TASKS), TASKS)
    assert rep["n_complete"] == 10 and rep["shipped"]
    g = rep["groups"]
    assert g["synthetic"]["n"] == 6 and g["public"]["n"] == 4
    assert abs(g["all"]["primary_emit4_minus_base"]["mean"] - 0.5) < 1e-9
    assert g["all"]["oracle_gain_recovered"] == 1.0 and g["all"]["full_bar_pass"]
    assert "| all |" in gr.markdown(rep)


def test_fail_and_missing():
    pd = _pd(TASKS, emit_good=False)
    del pd["gate1/eval/emit8/pub_0"]
    rep = gr.analyse(pd, TASKS)
    assert not rep["shipped"] and rep["n_complete"] == 9 and rep["missing"] == ["emit8/pub_0"]


def test_task_bootstrap():
    b = gr.task_bootstrap([0.1] * 10)
    assert abs(b["mean"] - 0.1) < 1e-12 and abs(b["ci95"][0] - 0.1) < 1e-12
