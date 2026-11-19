from minicpm_t2l import pool_report as pr


def _m(n_train, n_held):
    t = {}
    for i in range(n_train):
        t[f"t{i}"] = {"source": "synth", "family": "synthetic_form", "lang": "en", "heldout": False, "licence": "ours",
                      "commercial": "yes", "n_train": 200, "n_eval": 50, "dropped_train": 1, "dropped_eval": 0}
    for i in range(n_held):
        t[f"h{i}"] = {"source": "fintabnet", "family": "table", "lang": "en", "heldout": True, "licence": "x",
                      "commercial": "yes", "n_train": 150, "n_eval": 50, "dropped_train": 0, "dropped_eval": 0}
    return {"tasks": t, "dropped_tasks": {}}


def test_summary_pass_rule():
    s = pr.summary(_m(300, 30))
    assert s["pass"] and s["training_tasks"] == 300 and s["heldout_tasks"] == 30
    assert s["per_source"]["synth"]["dropped_train"] == 300 and s["train_docs_min"] == 150
    assert not pr.summary(_m(299, 30))["pass"] and not pr.summary(_m(300, 29))["pass"]


def test_heldout_list():
    h = pr.heldout_list(_m(2, 3), "2026-11-20")
    assert h["heldout_tasks"] == ["h0", "h1", "h2"] and h["heldout_sources"] == ["fintabnet"] and h["n_training"] == 2
