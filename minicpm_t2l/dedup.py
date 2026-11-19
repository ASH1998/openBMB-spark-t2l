"""Duplicate checks across splits and the final train/eval files.

Two checks, both frozen in studies/task_pool/protocol.md:

1. Exact image duplicates: equal sha256 of the decoded RGB pixels (+ size).
2. Near-duplicate text: Jaccard similarity of character 5-gram shingles of the
   normalised document text >= TEXT_JACCARD, for texts of at least
   MIN_TEXT_CHARS characters. Candidates come from MinHash LSH and are then
   confirmed with the exact Jaccard.

A duplicate pair only matters when it crosses a boundary:
- same task, one doc in train and one in eval -> drop the train doc from that task;
- one doc used by a training task, the other by a held-out task -> drop the doc
  from every training task (train and eval).
Held-out tasks and eval splits are never refilled or reshuffled; only training
splits lose docs, and train is refilled from the ordered reserve candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from minicpm_t2l.scoring import normalise

TEXT_JACCARD = 0.8
MIN_TEXT_CHARS = 30
SHINGLE = 5
NUM_PERM = 64
BANDS = 16
MAX_BUCKET = 2000
_P = (1 << 31) - 1  # Mersenne prime; a*x + b stays below 2**63


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    t = normalise(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _perms(num_perm: int = NUM_PERM, seed: int = 1):
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _P, size=num_perm, dtype=np.uint64)
    return a, b


def minhash(sh: set[str], perms) -> tuple[int, ...]:
    """MinHash signature: min over shingles of (a*x + b) mod p, one value per permutation."""
    import numpy as np

    a, b = perms
    xs = np.fromiter((_h64(s) % _P for s in sh), dtype=np.uint64, count=len(sh))
    vals = (a[:, None] * xs[None, :] + b[:, None]) % np.uint64(_P)
    return tuple(int(v) for v in vals.min(axis=1))


@dataclass(frozen=True)
class Doc:
    uid: str            # "<source>/<id>"
    pixel_sha256: str
    text: str


def exact_pairs(docs: Iterable[Doc]) -> set[tuple[str, str]]:
    by = defaultdict(list)
    for d in docs:
        by[d.pixel_sha256].append(d.uid)
    pairs = set()
    for uids in by.values():
        uids = sorted(set(uids))
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                pairs.add((uids[i], uids[j]))
    return pairs


def near_text_pairs(docs: Iterable[Doc], threshold: float = TEXT_JACCARD, min_chars: int = MIN_TEXT_CHARS,
                    relevant=None) -> tuple[set[tuple[str, str, float]], dict]:
    """Near-duplicate text pairs. `relevant(uid_a, uid_b)` can skip pairs that cross no boundary."""
    perms = _perms()
    sh, bands = {}, defaultdict(list)
    rows = NUM_PERM // BANDS
    skipped_short = 0
    for d in docs:
        if len(normalise(d.text)) < min_chars:
            skipped_short += 1
            continue
        s = shingles(d.text)
        sh[d.uid] = s
        sig = minhash(s, perms)
        for bnd in range(BANDS):
            bands[(bnd, sig[bnd * rows:(bnd + 1) * rows])].append(d.uid)
    cand, overflow = set(), 0
    for uids in bands.values():
        if len(uids) < 2:
            continue
        if len(uids) > MAX_BUCKET:
            overflow += 1
        uids = sorted(set(uids))[:MAX_BUCKET]
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                cand.add((uids[i], uids[j]))
    out = set()
    for a, b in cand:
        if relevant is not None and not relevant(a, b):
            continue
        jac = jaccard(sh[a], sh[b])
        if jac >= threshold:
            out.add((a, b, round(jac, 4)))
    return out, {"texts_checked": len(sh), "skipped_short": skipped_short, "candidates": len(cand),
                 "overflow_buckets": overflow}


# --------------------------------------------------------------------------- resolution

@dataclass(frozen=True)
class Usage:
    task: str
    split: str          # "train" (candidate) | "eval"
    heldout: bool


def resolve(pairs: Iterable[tuple[str, str]], usage: dict[str, list[Usage]]) -> tuple[set[tuple[str, str]], list[dict]]:
    """Apply the protocol rules. Returns (excluded (task, uid) set, list of actions taken)."""
    excluded: set[tuple[str, str]] = set()
    actions = []
    for a, b in pairs:
        ua, ub = usage.get(a, []), usage.get(b, [])
        # Rule 1: same task, train vs eval -> drop the train doc.
        for x, y, ux, uy in ((a, b, ua, ub), (b, a, ub, ua)):
            for u in ux:
                if u.split == "train" and any(v.task == u.task and v.split == "eval" for v in uy):
                    if (u.task, x) not in excluded:
                        excluded.add((u.task, x))
                        actions.append({"rule": "train_vs_eval", "task": u.task, "dropped": x, "dup_of": y})
        # Rule 2: training task vs held-out task -> drop from every training task.
        for x, y, ux, uy in ((a, b, ua, ub), (b, a, ub, ua)):
            if any(v.heldout for v in uy):
                for u in ux:
                    if not u.heldout and (u.task, x) not in excluded:
                        excluded.add((u.task, x))
                        actions.append({"rule": "train_vs_heldout", "task": u.task, "split": u.split, "dropped": x, "dup_of": y})
    return excluded, actions


def finalize(root: Path, train_cap: int, min_train: int, min_eval_training: int, log=print) -> dict:
    """Run both checks over the whole pool, write final train.jsonl files and a manifest."""
    from minicpm_t2l.pool import read_jsonl, write_jsonl

    root = Path(root)
    docs: dict[str, Doc] = {}
    for f in sorted((root / "docs").rglob("*.jsonl")):
        for r in read_jsonl(f):
            uid = f"{r['source']}/{r['id']}"
            docs[uid] = Doc(uid, r["pixel_sha256"], r.get("text", ""))
    log(f"docs indexed: {len(docs)}")
    usage: dict[str, list[Usage]] = defaultdict(list)
    cards, files = {}, {}
    for tdir in sorted((root / "tasks").iterdir()):
        if not (tdir / "task.json").exists():
            continue
        card = json.loads((tdir / "task.json").read_text(encoding="utf-8"))
        tid, src = card["task_id"], card["source"]
        cards[tid] = card
        ev, tr = read_jsonl(tdir / "eval.jsonl"), read_jsonl(tdir / "train_candidates.jsonl")
        files[tid] = (ev, tr)
        for split, rows in (("eval", ev), ("train", tr)):
            for r in rows:
                usage[f"{src}/{r['id']}"].append(Usage(tid, split, bool(card["heldout"])))

    def relevant(a: str, b: str) -> bool:
        ua, ub = usage.get(a, []), usage.get(b, [])
        if any(u.heldout for u in ua) != any(u.heldout for u in ub):
            return True
        ta = {u.task for u in ua}
        return any(u.task in ta for u in ub)

    ex = exact_pairs(docs.values())
    log(f"exact pixel-duplicate pairs: {len(ex)}")
    near, near_stats = near_text_pairs(docs.values(), relevant=relevant)
    log(f"near-duplicate text pairs (crossing a boundary): {len(near)} {near_stats}")
    pairs = ex | {(a, b) for a, b, _ in near}
    excluded, actions = resolve(pairs, usage)
    manifest = {"tasks": {}, "dropped_tasks": {}}
    for tid, (ev, tr) in files.items():
        card = cards[tid]
        src = card["source"]
        keep_tr = [r for r in tr if (tid, f"{src}/{r['id']}") not in excluded][:train_cap]
        keep_ev = ev if card["heldout"] else [r for r in ev if (tid, f"{src}/{r['id']}") not in excluded]
        tdir = root / "tasks" / tid
        reason = None
        if len(keep_tr) < min_train:
            reason = f"train {len(keep_tr)} < {min_train}"
        elif not card["heldout"] and len(keep_ev) < min_eval_training:
            reason = f"eval {len(keep_ev)} < {min_eval_training}"
        elif card["heldout"] and len(keep_ev) < len(ev):
            reason = "held-out eval changed"  # cannot happen under the rules; guard
        write_jsonl(tdir / "train.jsonl", keep_tr)
        if len(keep_ev) != len(ev):
            write_jsonl(tdir / "eval.jsonl", keep_ev)
        entry = {"source": src, "family": card.get("family"), "lang": card.get("lang"), "heldout": card["heldout"],
                 "licence": card.get("licence"), "commercial": card.get("commercial"),
                 "n_train": len(keep_tr), "n_eval": len(keep_ev),
                 "dropped_train": sum(1 for r in tr if (tid, f"{src}/{r['id']}") in excluded),
                 "dropped_eval": len(ev) - len(keep_ev)}
        if reason:
            manifest["dropped_tasks"][tid] = {**entry, "reason": reason}
        else:
            manifest["tasks"][tid] = entry
    report = {"n_docs": len(docs), "exact_pairs": len(ex), "near_text_pairs": len(near), "near_text_stats": near_stats,
              "exact_pair_examples": sorted(ex)[:50], "near_pair_examples": sorted(near)[:50],
              "actions": actions, "n_actions": len(actions),
              "rules": {"text_jaccard": TEXT_JACCARD, "min_text_chars": MIN_TEXT_CHARS, "shingle": SHINGLE,
                        "num_perm": NUM_PERM, "bands": BANDS}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "dedup_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "FINALIZED").write_text("ok\n", encoding="utf-8")
    return {"manifest": manifest, "report": {k: v for k, v in report.items() if k != "actions"}}
