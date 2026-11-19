from minicpm_t2l.tasks import TASKS, EXTRACTORS, SOURCES, cord_full, cord_menu, form_qa_pairs, split_ids, sroie_all, wr_money


def test_registry():
    assert len(TASKS) == 20
    for t in TASKS.values():
        assert t.extract in EXTRACTORS and t.source in SOURCES
        assert "Return only JSON" in t.prompt()


def test_cord_single_item_dict_and_full():
    gt = {"menu": {"nm": "Tea", "cnt": "1", "price": "5", "unitprice": "5"},
          "sub_total": {"subtotal_price": "5"}, "total": {"total_price": "5", "menuqty_cnt": "1"}}
    assert cord_menu(gt) == {"menu": [{"nm": "Tea", "cnt": "1", "price": "5"}]}
    assert cord_full(gt)["total"] == {"total_price": "5"}


def test_sroie_all_fills_missing():
    assert sroie_all({"company": "A", "total": "1"}) == {"company": "A", "date": "", "address": "", "total": "1"}


def test_wr_money():
    gt = {"Total_value": "$ 9", "Tax_list": [{"Tax_value": "1"}, {"Tax_value": ""}]}
    assert wr_money(gt) == {"Total_value": "$ 9", "Tax_list": [{"Tax_value": "1"}]}


def test_form_qa_pairs_links_and_orders_answers():
    ents = [
        {"id": 1, "label": "question", "text": "Name:", "box": [0, 0, 10, 10], "linking": [[1, 2], [1, 3]]},
        {"id": 3, "label": "answer", "text": "Doe", "box": [20, 0, 30, 10], "linking": [[1, 3]]},
        {"id": 2, "label": "answer", "text": "John", "box": [12, 0, 18, 10], "linking": [[1, 2]]},
        {"id": 4, "label": "question", "text": "Unanswered", "box": [0, 20, 10, 30], "linking": []},
        {"id": 5, "label": "header", "text": "FORM", "box": [0, 40, 10, 50], "linking": []},
    ]
    assert form_qa_pairs(ents) == {"pairs": [{"question": "Name:", "answer": "John Doe"}]}


def test_split_ids_deterministic_and_disjoint():
    ids = [f"d{i}" for i in range(300)]
    tr, ev = split_ids(ids, None)
    assert len(ev) == 50 and len(tr) == 200 and not set(tr) & set(ev)
    assert (tr, ev) == split_ids(list(reversed(ids)), None)
    tr2, ev2 = split_ids(ids[:120], [f"v{i}" for i in range(80)])
    assert len(ev2) == 50 and all(e.startswith("v") for e in ev2) and len(tr2) == 120
