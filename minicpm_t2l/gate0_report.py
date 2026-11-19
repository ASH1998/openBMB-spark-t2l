"""Gate 0 decision report from run outputs.

Reads <runs>/<model>/<arm>/<task>/predictions.jsonl (pulled from the volume with
`modal volume get minicpm-t2l-artifacts gate0/runs data/gate0_runs`) and applies
the pass rule frozen in studies/gate0/protocol.md.

    python -m minicpm_t2l.gate0_report --runs data/gate0_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicpm_t2l.scoring import DocScore, TaskScore, paired_bootstrap
from minicpm_t2l.tasks import TASKS

MARGIN = 0.05          # absolute F1 points a LoRA must gain over the best baseline
MIN_WINS = 11          # of 20 tasks ("most")
BASELINE_ARMS = ("base", "icl1", "icl4")


def load(path: Path) -> list[DocScore] | None:
    p = path / "predictions.jsonl"
    if not p.exists() or not (path / "scores.json").exists():  # scores.json marks a finished run
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    return [DocScore(r["tp"], r["n_pred"], r["n_gold"], r["parsed"]) for r in rows]


def f1(ds: list[DocScore]) -> float:
    ts = TaskScore(docs=list(ds))
    return ts.summary()["f1"]


def report(runs: Path) -> dict:
    out = {}
    for model_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        model = model_dir.name
        lora_arms = sorted(p.name for p in model_dir.iterdir() if p.name.startswith("lora"))
        out[model] = {}
        for arm in lora_arms:
            rows, wins, missing = [], 0, []
            for task in TASKS:
                lora = load(model_dir / arm / task)
                bases = {a: load(model_dir / a / task) for a in BASELINE_ARMS}
                if lora is None or any(v is None for v in bases.values()):
                    missing.append(task)
                    continue
                best_arm = max(bases, key=lambda a: f1(bases[a]))
                bs = paired_bootstrap(bases[best_arm], lora)
                win = bs["diff"] >= MARGIN and bs["ci95"][0] > 0
                wins += win
                rows.append({"task": task, **{a: round(f1(v), 3) for a, v in bases.items()},
                             arm: round(f1(lora), 3), "best_baseline": best_arm,
                             "gain": round(bs["diff"], 3), "ci95": [round(x, 3) for x in bs["ci95"]], "win": win})
            out[model][arm] = {"wins": wins, "n_tasks": len(rows), "missing": missing,
                               "pass": wins >= MIN_WINS and not missing, "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/gate0_runs")
    ap.add_argument("--json", default=None, help="also write the report here")
    a = ap.parse_args()
    rep = report(Path(a.runs))
    for model, arms in rep.items():
        for arm, r in arms.items():
            print(f"\n== {model} / {arm}: {r['wins']}/{r['n_tasks']} wins -> {'PASS' if r['pass'] else 'FAIL'}"
                  + (f" (missing {len(r['missing'])})" if r["missing"] else ""))
            for row in r["rows"]:
                print(f"  {row['task']:16s} base={row['base']:.3f} icl1={row['icl1']:.3f} icl4={row['icl4']:.3f} "
                      f"{arm}={row[arm]:.3f} gain={row['gain']:+.3f} ci={row['ci95']} {'WIN' if row['win'] else ''}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
