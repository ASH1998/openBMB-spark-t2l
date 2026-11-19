import json

from minicpm_t2l import dedup, pool
from minicpm_t2l.dedup import Doc, Usage


def test_jaccard_and_shingles():
    a = dedup.shingles("Hello  World")
    assert a == dedup.shingles("hello world")
    assert dedup.jaccard(a, a) == 1.0
    assert dedup.jaccard(a, dedup.shingles("something else entirely")) < 0.1


def test_minhash_estimates_similarity():
    perms = dedup._perms()
    s1 = dedup.shingles("The quick brown fox jumps over the lazy dog near the river bank today")
    s2 = dedup.shingles("The quick brown fox jumps over the lazy dog near the river bank toda")
    s3 = dedup.shingles("Completely unrelated receipt text with totals and taxes 12.50 USD")
    m1, m2, m3 = (dedup.minhash(s, perms) for s in (s1, s2, s3))
    agree = lambda x, y: sum(p == q for p, q in zip(x, y)) / len(x)
    assert agree(m1, m2) > 0.7 and agree(m1, m3) < 0.2


def test_exact_and_near_pairs():
    long = "Invoice 12345 issued to Example Corp, 1 Main Street, Springfield, total 99.00"
    docs = [Doc("s/a", "h1", long), Doc("s/b", "h1", "other"), Doc("t/c", "h2", long + "!"),
            Doc("t/d", "h3", "short"), Doc("t/e", "h4", "A totally different document about formulas and tables")]
    assert dedup.exact_pairs(docs) == {("s/a", "s/b")}
    near, stats = dedup.near_text_pairs(docs)
    assert {(a, b) for a, b, _ in near} == {("s/a", "t/c")}
    assert stats["skipped_short"] == 2


def test_resolve_rules():
    usage = {
        "x/1": [Usage("t1", "train", False)],
        "x/2": [Usage("t1", "eval", False)],
        "y/3": [Usage("h1", "eval", True)],
        "x/4": [Usage("t2", "train", False), Usage("t3", "eval", False)],
        "x/5": [Usage("t9", "train", False)],
    }
    ex, actions = dedup.resolve([("x/1", "x/2"), ("x/4", "y/3"), ("x/1", "x/5")], usage)
    assert ("t1", "x/1") in ex                      # same task, train vs eval
    assert ("t2", "x/4") in ex and ("t3", "x/4") in ex  # training doc duplicates a held-out doc
    assert ("h1", "y/3") not in ex                  # held-out side never changes
    assert ("t9", "x/5") not in ex                  # different training tasks: no boundary crossed
    assert {a["rule"] for a in actions} == {"train_vs_eval", "train_vs_heldout"}


def _task(root, tid, src, heldout, ev, tr):
    d = root / "tasks" / tid
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": tid, "source": src, "heldout": heldout, "family": "x", "lang": "en"}))
    pool.write_jsonl(d / "eval.jsonl", [{"id": i, "image": "", "label": {}} for i in ev])
    pool.write_jsonl(d / "train_candidates.jsonl", [{"id": i, "image": "", "label": {}} for i in tr])


def test_finalize_end_to_end(tmp_path):
    words = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima", "mike november oscar",
             "papa quebec romeo", "sierra tango uniform", "victor whiskey xray"]
    docs = [{"id": f"d{i}", "source": "s", "pixel_sha256": f"h{i}", "text": f"{w} {w.upper()} {i * 7919} {w[::-1]}"}
            for i, w in enumerate(words)]
    docs[1]["pixel_sha256"] = "h0"                                   # d1 (train) duplicates d0 (eval)
    docs.append({"id": "q0", "source": "h", "pixel_sha256": "h5", "text": "held"})  # h/q0 duplicates s/d5
    pool.write_jsonl(tmp_path / "docs" / "s.jsonl", docs[:8])
    pool.write_jsonl(tmp_path / "docs" / "h.jsonl", docs[8:])
    _task(tmp_path, "train_task", "s", False, ["d0"], ["d1", "d2", "d3", "d4", "d5", "d6"])
    _task(tmp_path, "held_task", "h", True, ["q0"], ["q0"][:0])
    out = dedup.finalize(tmp_path, train_cap=3, min_train=0, min_eval_training=1, log=lambda *_: None)
    final = [r["id"] for r in pool.read_jsonl(tmp_path / "tasks" / "train_task" / "train.jsonl")]
    assert final == ["d2", "d3", "d4"]          # d1 dropped, refilled from the reserve, capped at 3
    m = out["manifest"]["tasks"]["train_task"]
    assert m["dropped_train"] == 2               # d1 (train vs eval) and d5 (vs held-out)
    assert (tmp_path / "FINALIZED").exists()
