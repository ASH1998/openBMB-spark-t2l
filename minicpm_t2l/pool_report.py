"""Summaries of the finalized task pool, and the frozen held-out list.

    python -m minicpm_t2l.pool_report summary                 # counts from studies/task_pool/manifest.json
    python -m minicpm_t2l.pool_report freeze --date 2026-11-20 # writes studies/task_pool/heldout_<date>.json (refuses to overwrite)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STUDY = Path("studies/task_pool")
MIN_TRAINING_TASKS = 300
MIN_HELDOUT_TASKS = 30


def summary(manifest: dict) -> dict:
    tasks = manifest["tasks"]
    train = {k: v for k, v in tasks.items() if not v["heldout"]}
    held = {k: v for k, v in tasks.items() if v["heldout"]}
    by = lambda d, key: dict(sorted(Counter(v[key] or "?" for v in d.values()).items()))
    per_source = defaultdict(lambda: {"tasks": 0, "train_docs": 0, "eval_docs": 0, "dropped_train": 0, "dropped_eval": 0})
    for v in tasks.values():
        s = per_source[v["source"]]
        s["tasks"] += 1
        s["train_docs"] += v["n_train"]
        s["eval_docs"] += v["n_eval"]
        s["dropped_train"] += v["dropped_train"]
        s["dropped_eval"] += v["dropped_eval"]
        s["heldout"] = v["heldout"]
        s["licence"] = v.get("licence")
        s["commercial"] = v.get("commercial")
    return {
        "training_tasks": len(train), "heldout_tasks": len(held), "dropped_tasks": len(manifest["dropped_tasks"]),
        "pass": len(train) >= MIN_TRAINING_TASKS and len(held) >= MIN_HELDOUT_TASKS,
        "training_by_family": by(train, "family"), "heldout_by_family": by(held, "family"),
        "training_by_lang": by(train, "lang"), "heldout_by_lang": by(held, "lang"),
        "training_by_commercial": by(train, "commercial"),
        "per_source": dict(sorted(per_source.items())),
        "train_docs_min": min((v["n_train"] for v in tasks.values()), default=0),
        "eval_docs_min": min((v["n_eval"] for v in tasks.values()), default=0),
    }


def heldout_list(manifest: dict, date: str, protocol: str = "studies/task_pool/protocol.md") -> dict:
    held = sorted(k for k, v in manifest["tasks"].items() if v["heldout"])
    train = sorted(k for k, v in manifest["tasks"].items() if not v["heldout"])
    return {"frozen": date, "protocol": protocol, "n_heldout": len(held), "n_training": len(train),
            "heldout_tasks": held, "training_tasks": train,
            "heldout_sources": sorted({manifest["tasks"][k]["source"] for k in held if manifest["tasks"][k]["source"] != "synth"}),
            "note": "Never train or tune on any held-out task. Changes need a new dated file with a reason."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "freeze"])
    ap.add_argument("--date")
    a = ap.parse_args()
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    if a.cmd == "summary":
        print(json.dumps(summary(manifest), indent=2))
        return
    if not a.date:
        raise SystemExit("--date is required")
    out = STUDY / f"heldout_{a.date}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; held-out lists are never overwritten")
    out.write_text(json.dumps(heldout_list(manifest, a.date), indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
