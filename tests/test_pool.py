import json

from PIL import Image

from minicpm_t2l import pool
from minicpm_t2l import tasks as g0


def test_registry_consistent():
    for t in pool.PUBLIC_TASKS.values():
        assert t.source in pool.SOURCES and t.extract in pool.EXTRACTORS
        assert "Return only JSON" in t.prompt()
    # Gate 0 tasks are carried over unchanged.
    for t in g0.TASKS.values():
        p = pool.PUBLIC_TASKS[t.task_id]
        assert (p.source, p.instruction, p.schema, p.extract) == (t.source, t.instruction, t.schema, t.extract)


def test_gate0_sources_training_only():
    for k in g0.SOURCES:
        assert not pool.SOURCES[k].heldout


def test_heldout_sources():
    held = {k for k, s in pool.SOURCES.items() if s.heldout}
    assert held == {"fintabnet", "chartqa", "invoices"}


def test_order_ids_reproduces_gate0_split():
    ids = [f"d{i}" for i in range(300)]
    tr, ev = pool.order_ids(ids, None, g0.SEED, g0.EVAL_N)
    g_tr, g_ev = g0.split_ids(ids, None)
    assert ev[: g0.EVAL_N] == g_ev and tr[: g0.TRAIN_CAP] == g_tr
    evs = [f"v{i}" for i in range(80)]
    tr, ev = pool.order_ids(ids, evs, g0.SEED, g0.EVAL_N)
    g_tr, g_ev = g0.split_ids(ids, evs)
    assert ev[: g0.EVAL_N] == g_ev and tr[: g0.TRAIN_CAP] == g_tr


def test_select_for_task_skips_ineligible():
    labels = {"a": {"x": 1}, "b": None, "c": {"x": 2}}
    assert pool.select_for_task(["a", "b", "c"], labels, 5) == ["a", "c"]


HTML = """<table><thead><tr><td></td><td colspan="2"><b>Age</b></td></tr>
<tr><td>Group</td><td>Mean</td><td>Grade<sup>a</sup></td></tr></thead>
<tbody><tr><td>OA</td><td>17 (9/8)</td><td>2</td></tr><tr><td>Control</td><td>12</td><td>1</td></tr></tbody></table>"""


def test_table_extractors():
    rows, width = pool.parse_table(HTML)
    assert rows[1] == ["Group", "Mean", "Gradea"] and width == 3
    t = {"html": HTML}
    assert pool.tab_header(t) == {"header": ["Age"]}
    assert pool.tab_first_col(t) == {"first_column": ["Group", "OA", "Control"]}
    assert pool.tab_shape(t) == {"rows": "4", "columns": "3"}
    assert pool.tab_last_row(t) == {"last_row": ["Control", "12", "1"]}
    assert pool.tab_small_full(t)["rows"][2] == ["OA", "17 (9/8)", "2"]


def test_chart_extractors():
    c = {"table": [["Country", "Share"], ["Nigeria", "43.54"]], "chart_source": "owid"}
    assert pool.EXTRACTORS["chart_table_owid"](c) == {"columns": ["Country", "Share"], "rows": [["Nigeria", "43.54"]]}
    assert pool.EXTRACTORS["chart_table_pew"](c) is None
    assert pool._chart_source("00006834003065.png") == "owid"
    assert pool._chart_source("two_col_4809.png") == "statista"
    assert pool._chart_source("10095.png") == "pew"
    assert pool._csv_rows('Country,"A, B"\r\nX,1\r\n') == [["Country", "A, B"], ["X", "1"]]


def test_dvqa_and_latex():
    rec = {"qa": [{"question": "Which bar has the largest value?", "answer": "Soil."}]}
    assert pool.EXTRACTORS["dvqa_largest_bar"](rec) == {"answer": "Soil"}
    assert pool.EXTRACTORS["dvqa_smallest_bar"](rec) is None
    assert pool.EXTRACTORS["latex_short"]({"latex": "q _ { i }"}) == {"latex": "q _ { i }"}
    assert pool.EXTRACTORS["latex_long"]({"latex": "q _ { i }"}) is None


def test_invoice_extractors():
    gt = {"header": {"invoice_no": "1", "seller": "S"}, "items": [{"item_desc": "x", "item_qty": "1,00"}],
          "summary": {"total_vat": "$1"}}
    assert pool.inv_header(gt)["invoice_no"] == "1" and pool.inv_header(gt)["iban"] == ""
    assert pool.inv_items(gt)["items"][0]["item_qty"] == "1,00"
    assert pool.inv_items({"items": []}) is None
    assert pool.inv_summary(gt) == {"total_net_worth": "", "total_vat": "$1", "total_gross_worth": ""}


def test_hashes_detect_identical_pixels():
    a = Image.new("RGB", (40, 30), (200, 10, 10))
    b = a.copy()
    c = Image.new("RGB", (40, 30), (10, 200, 10))
    ha, hb, hc = (pool.pixel_hash_and_dhash(x) for x in (a, b, c))
    assert ha[0] == hb[0] != hc[0]
    assert len(ha[1]) == 16 and ha[2] == (40, 30)


def test_doc_text_for_tables_and_forms():
    assert "Control" in pool.doc_text_for("pubtabnet", {"html": HTML})
    assert pool.doc_text_for("funsd", [{"text": "Name"}, {"text": "Bob"}]) == "Name Bob"


def test_build_synth_writes_task(tmp_path, monkeypatch):
    from minicpm_t2l import synth

    monkeypatch.setattr(pool, "TRAIN_CAP", 2)
    monkeypatch.setattr(pool, "TRAIN_RESERVE", 1)
    monkeypatch.setattr(pool, "EVAL_N", 2)
    tid = synth.make_templates()[3].template_id
    out = pool.build_synth_templates([tid], tmp_path, log=lambda *_: None)
    assert out[tid] == {"eval": 2, "train_candidates": 3}
    card = json.loads((tmp_path / "tasks" / tid / "task.json").read_text(encoding="utf-8"))
    assert card["source"] == "synth" and "Return only JSON" in card["prompt"]
    rows = pool.read_jsonl(tmp_path / "tasks" / tid / "eval.jsonl")
    assert (tmp_path / rows[0]["image"]).exists()
    docs = pool.read_jsonl(tmp_path / "docs" / "synth" / f"{tid}.jsonl")
    assert len(docs) == 5 and all(len(d["pixel_sha256"]) == 64 for d in docs)
