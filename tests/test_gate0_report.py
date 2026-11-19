import json
from minicpm_t2l.gate0_report import report
from minicpm_t2l.tasks import TASKS


def write(d, rows):
    d.mkdir(parents=True)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "scores.json").write_text("{}")


def test_unfinished_run_is_missing(tmp_path):
    rows = [{"id": "d0", "tp": 1, "n_pred": 1, "n_gold": 1, "parsed": True}]
    for arm in ("base", "icl1", "icl4", "loraA"):
        write(tmp_path / "v47" / arm / "cord_menu", rows)
    (tmp_path / "v47" / "loraA" / "cord_menu" / "scores.json").unlink()
    assert "cord_menu" in report(tmp_path)["v47"]["loraA"]["missing"]


def test_report_counts_wins(tmp_path):
    good = [{"id": f"d{i}", "tp": 2, "n_pred": 2, "n_gold": 2, "parsed": True} for i in range(40)]
    bad = [{"id": f"d{i}", "tp": 1, "n_pred": 2, "n_gold": 2, "parsed": True} for i in range(40)]
    for i, t in enumerate(TASKS):
        for arm in ("base", "icl1", "icl4"):
            write(tmp_path / "v47" / arm / t, bad)
        write(tmp_path / "v47" / "loraA" / t, good if i < 12 else bad)
    r = report(tmp_path)["v47"]["loraA"]
    assert r["wins"] == 12 and r["pass"] and not r["missing"]
