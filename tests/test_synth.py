import json

import pytest

from minicpm_t2l import synth
from minicpm_t2l.synth_fields import DOMAINS


@pytest.fixture(scope="module")
def templates():
    return synth.make_templates()


def test_template_count_and_distinct_schemas(templates):
    assert len(templates) == len(DOMAINS) * len(synth.LANGS) * synth.TEMPLATES_PER_DOMAIN_LANG
    sigs = {(t.domain, t.lang, frozenset(t.schema_fields), t.items, t.item_cols, t.key_style) for t in templates}
    assert len(sigs) == len(templates)
    assert len({t.template_id for t in templates}) == len(templates)


def test_templates_are_deterministic(templates):
    again = synth.make_templates()
    assert [t.to_json() for t in again] == [t.to_json() for t in templates]


def test_heldout_domains_fixed():
    held = synth.heldout_domains()
    assert held == synth.heldout_domains() and len(held) == 4 and set(held) <= set(DOMAINS)


def test_schema_is_json_shaped(templates):
    t = next(t for t in templates if t.items)
    s = t.schema()
    assert s.startswith("{") and t.items_json_key() in s


@pytest.mark.parametrize("idx", [0, 7, 100, 250, 383])
def test_every_gold_value_is_printed(templates, idx):
    t = templates[idx]
    for i in (0, 1, 60):
        img, label, drawn = synth.render_doc(t, i)
        assert img.size[0] > 300 and img.size[1] > 200
        printed = set(drawn)
        for v in json.loads(json.dumps(label)).values():
            vals = [x for row in v for x in row.values()] if isinstance(v, list) else [v]
            for x in vals:
                assert x == "" or x in printed, (t.template_id, x)


def test_render_is_deterministic(templates):
    t = templates[11]
    a = synth.render_doc(t, 5)
    b = synth.render_doc(t, 5)
    assert a[1] == b[1] and a[0].tobytes() == b[0].tobytes()


def test_label_keys_follow_key_style(templates):
    for t in templates[:40]:
        label = synth.label_for(t, synth.sample_values(t, 0))
        keys = [t.json_key(t.field(k)) for k in t.schema_fields]
        assert list(label)[: len(keys)] == keys
        if t.key_style == "snake":
            assert keys == list(t.schema_fields)


def test_date_format():
    assert synth.fmt_date(2024, 3, 5, "%Y年%m月%d日") == "2024年03月05日"
    assert synth.fmt_date(2024, 3, 5, "%d %b %Y") == "05 Mar 2024"


def test_doc_text_flattens():
    assert synth.doc_text({"a": "x", "items": [{"b": "y", "c": ""}]}) == "x y"
