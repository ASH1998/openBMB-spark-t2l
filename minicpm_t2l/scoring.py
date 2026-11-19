"""Scoring for document -> JSON extraction tasks.

The primary metric is field-level F1. Both prediction and label are flattened
into a multiset of (path, value) pairs:

- dict keys extend the path ("total.total_price")
- list elements share their parent path with a "[]" suffix, so list order does
  not matter but counts do ("menu[].nm")
- values are normalised: Unicode NFKC, case-folded, whitespace collapsed, and
  surrounding punctuation stripped

A document with an unparseable prediction scores as an empty prediction, and the
parse failure is counted separately. Task-level F1 is micro-averaged over pairs.
Exact match (all pairs equal) is reported as a secondary metric.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_WS = re.compile(r"\s+")
_EDGE_PUNCT = " \t\n.,;:'\"`"


def normalise(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        value = "true" if value else "false"
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    s = _WS.sub(" ", s).strip(_EDGE_PUNCT)
    return s


def flatten(obj, path: str = "") -> Counter:
    """Multiset of (path, normalised value); empty values are dropped."""
    out: Counter = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for v in obj:
            out.update(flatten(v, f"{path}[]"))
    else:
        v = normalise(obj)
        if v:
            out[(path, v)] += 1
    return out


def parse_json_output(text: str):
    """Extract the first JSON object/array from model output. Returns None on failure."""
    if text is None:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        for i, ch in enumerate(cand):
            if ch in "{[":
                try:
                    obj, _ = dec.raw_decode(cand[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
    return None


@dataclass
class DocScore:
    tp: int
    n_pred: int
    n_gold: int
    parsed: bool

    @property
    def exact(self) -> bool:
        return self.parsed and self.tp == self.n_pred == self.n_gold


def score_doc(pred_text: str, gold) -> DocScore:
    pred = parse_json_output(pred_text)
    parsed = pred is not None
    p = flatten(pred) if parsed else Counter()
    g = flatten(gold)
    tp = sum((p & g).values())
    return DocScore(tp=tp, n_pred=sum(p.values()), n_gold=sum(g.values()), parsed=parsed)


def _f1(tp: int, n_pred: int, n_gold: int) -> float:
    if n_pred == 0 and n_gold == 0:
        return 1.0
    return 2 * tp / (n_pred + n_gold) if (n_pred + n_gold) else 0.0


@dataclass
class TaskScore:
    docs: list[DocScore] = field(default_factory=list)

    def add(self, d: DocScore) -> None:
        self.docs.append(d)

    def summary(self) -> dict:
        tp = sum(d.tp for d in self.docs)
        npred = sum(d.n_pred for d in self.docs)
        ngold = sum(d.n_gold for d in self.docs)
        n = len(self.docs) or 1
        return {
            "n_docs": len(self.docs),
            "f1": _f1(tp, npred, ngold),
            "precision": tp / npred if npred else 0.0,
            "recall": tp / ngold if ngold else 0.0,
            "exact_match": sum(d.exact for d in self.docs) / n,
            "parse_fail_rate": sum(not d.parsed for d in self.docs) / n,
        }


def paired_bootstrap(a: list[DocScore], b: list[DocScore], n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the micro-F1 difference b - a over documents (same docs, same order)."""
    import random

    if len(a) != len(b):
        raise ValueError("paired bootstrap needs the same documents in both runs")
    rng = random.Random(seed)
    n = len(a)

    def f1(ds, idx):
        return _f1(sum(ds[i].tp for i in idx), sum(ds[i].n_pred for i in idx), sum(ds[i].n_gold for i in idx))

    full = list(range(n))
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(f1(b, idx) - f1(a, idx))
    diffs.sort()
    return {
        "diff": f1(b, full) - f1(a, full),
        "ci95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]),
    }
