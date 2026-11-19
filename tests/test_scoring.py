from minicpm_t2l.scoring import flatten, normalise, paired_bootstrap, parse_json_output, score_doc, TaskScore


def test_normalise():
    assert normalise("  Total:  9.00 ") == "total:  9.00".replace("  ", " ")
    assert normalise("ＡＢＣ") == "abc"  # NFKC + casefold


def test_parse_fenced_and_prefixed():
    assert parse_json_output('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_output('Answer: {"a": "x"} trailing') == {"a": "x"}
    assert parse_json_output("no json here") is None


def test_flatten_lists_are_order_free_multisets():
    a = flatten({"menu": [{"nm": "Tea"}, {"nm": "Rice"}, {"nm": "Tea"}]})
    b = flatten({"menu": [{"nm": "rice"}, {"nm": "tea"}, {"nm": "TEA"}]})
    assert a == b and a[("menu[].nm", "tea")] == 2


def test_score_doc_partial_and_parse_fail():
    gold = {"company": "ACME", "total": "9.00"}
    d = score_doc('{"company": "acme", "total": "8.00"}', gold)
    assert (d.tp, d.n_pred, d.n_gold, d.parsed, d.exact) == (1, 2, 2, True, False)
    f = score_doc("garbage", gold)
    assert not f.parsed and f.tp == 0 and f.n_gold == 2
    ts = TaskScore(); ts.add(d); ts.add(f)
    s = ts.summary()
    assert abs(s["f1"] - 2 * 1 / (2 + 4)) < 1e-9 and s["parse_fail_rate"] == 0.5


def test_paired_bootstrap_detects_improvement():
    gold = {"x": "1"}
    bad = [score_doc('{"x": "0"}', gold) for _ in range(30)]
    good = [score_doc('{"x": "1"}', gold) for _ in range(30)]
    r = paired_bootstrap(bad, good, n_boot=200)
    assert r["diff"] == 1.0 and r["ci95"][0] > 0
