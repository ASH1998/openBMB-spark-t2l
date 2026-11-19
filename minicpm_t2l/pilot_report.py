"""Analysis for pilots P-03 (OCR at 4x) and P-01 (multi-task LoRA), as fixed in their protocols.

Input: `studies/pilots/<pilot>/per_doc.json` from `modal run modal_app/pilots.py::pull`,
mapping a run path to {"scores": {...}, "docs": [{"id", "tp", "n_pred", "n_gold", "parsed"}]}.

    python -m minicpm_t2l.pilot_report p03
    python -m minicpm_t2l.pilot_report p01
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from minicpm_t2l.scoring import DocScore, TaskScore, paired_bootstrap
from minicpm_t2l.tasks import TASKS

WIN_MARGIN = 0.05
BASE_ARMS = ("base", "icl1", "icl4")


def docs_of(run: dict) -> list[DocScore]:
    return [DocScore(d["tp"], d["n_pred"], d["n_gold"], d["parsed"]) for d in run["docs"]]


def f1_of(run: dict) -> float:
    ts = TaskScore()
    for d in docs_of(run):
        ts.add(d)
    return ts.summary()["f1"]


def _aligned(a: dict, b: dict) -> None:
    if [d["id"] for d in a["docs"]] != [d["id"] for d in b["docs"]]:
        raise ValueError("runs are not on the same docs in the same order")


def compare(a: dict, b: dict, seed: int = 0) -> dict:
    """b - a with a paired bootstrap CI; `win` uses Gate 0's rule."""
    _aligned(a, b)
    r = paired_bootstrap(docs_of(a), docs_of(b), n_boot=2000, seed=seed)
    lo, hi = r["ci95"]
    return {"diff": r["diff"], "ci95": [lo, hi], "win": r["diff"] >= WIN_MARGIN and lo > 0,
            "better": lo > 0, "worse": hi < 0}


def best_of(runs: dict[str, dict]) -> tuple[str, dict]:
    arm = max(runs, key=lambda a: (f1_of(runs[a]), -BASE_ARMS.index(a) if a in BASE_ARMS else 0))
    return arm, runs[arm]


def _get(pd: dict, path: str) -> dict:
    r = pd.get(path)
    if r is None:
        raise KeyError(f"missing run {path}")
    return r


def p03(pd: dict) -> dict:
    rows, wins, shift_pos = [], 0, 0
    v47_abs = {"best_baseline": 0, "loraB": 0}
    for t in TASKS:
        g16 = {a: _get(pd, f"gate0/runs_a10g/ocr/{a}/{t}") for a in BASE_ARMS}
        g4 = {a: _get(pd, f"pilots/p03_ocr4x/{a}/{t}") for a in BASE_ARMS}
        a16, b16 = best_of(g16)
        a4, b4 = best_of(g4)
        lora = _get(pd, f"gate0/runs_a10g/ocr/loraB/{t}")
        _, v47b = best_of({a: _get(pd, f"gate0/runs_a10g/v47/{a}/{t}") for a in BASE_ARMS})
        v47l = _get(pd, f"gate0/runs_a10g/v47/loraB/{t}")
        shift = compare(b16, b4)
        head = compare(b4, lora)
        wins += head["win"]
        shift_pos += shift["better"]
        v47_abs["best_baseline"] += f1_of(b4) > f1_of(v47b)
        v47_abs["loraB"] += f1_of(b4) > f1_of(v47l)
        rows.append({"task": t, "best16x_arm": a16, "best16x": f1_of(b16), "best4x_arm": a4, "best4x": f1_of(b4),
                     "shift": shift, "loraB16x": f1_of(lora), "headroom_vs_4x": head,
                     "v47_best_baseline": f1_of(v47b), "v47_loraB": f1_of(v47l),
                     "parse_fail_4x": {a: g4[a]["scores"]["parse_fail_rate"] for a in BASE_ARMS}})
    n = len(rows)
    return {"rows": rows, "loraB16x_wins_vs_best4x": wins, "n_tasks": n,
            "tasks_where_4x_baseline_significantly_higher": shift_pos,
            "mean_best16x": sum(r["best16x"] for r in rows) / n, "mean_best4x": sum(r["best4x"] for r in rows) / n,
            "ocr4x_best_baseline_above_v47": v47_abs, "headroom_holds": wins >= 11}


def p01(pd: dict) -> dict:
    rows, wins, better, worse = [], 0, 0, 0
    num, den = 0.0, 0.0
    for t in TASKS:
        base = {a: _get(pd, f"gate0/runs_a10g/ocr/{a}/{t}") for a in BASE_ARMS}
        arm, best = best_of(base)
        lora = _get(pd, f"gate0/runs_a10g/ocr/loraB/{t}")
        mt = _get(pd, f"pilots/p01_multitask/eval/{t}")
        vs_base = compare(best, mt)
        vs_lora = compare(lora, mt)
        lora_vs_base = compare(best, lora)
        wins += vs_base["win"]
        better += vs_lora["better"]
        worse += vs_lora["worse"]
        if lora_vs_base["win"]:
            num += f1_of(mt) - f1_of(best)
            den += f1_of(lora) - f1_of(best)
        rows.append({"task": t, "best_baseline_arm": arm, "best_baseline": f1_of(best), "loraB": f1_of(lora),
                     "mt": f1_of(mt), "mt_vs_best": vs_base, "mt_vs_loraB": vs_lora, "loraB_win": lora_vs_base["win"],
                     "mt_parse_fail": mt["scores"]["parse_fail_rate"]})
    n = len(rows)
    retention = num / den if den else float("nan")
    return {"rows": rows, "mt_wins_vs_best_baseline": wins, "n_tasks": n,
            "mean_gain_mt": sum(r["mt"] - r["best_baseline"] for r in rows) / n,
            "mean_gain_loraB": sum(r["loraB"] - r["best_baseline"] for r in rows) / n,
            "mt_vs_loraB": {"better": better, "worse": worse, "no_clear_difference": n - better - worse},
            "retention": retention,
            "reading": ("strong comparison 3" if retention >= 0.8 and worse <= 3
                        else "tasks interfere" if retention < 0.5 else "in between")}


def _ci(c: dict) -> str:
    return f"{c['diff']:+.3f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"


def markdown(name: str, rep: dict) -> str:
    if name == "p03":
        lines = ["| Task | Best 16x (arm) | Best 4x (arm) | Shift 4x − 16x [95% CI] | LoRA B 16x | LoRA B − best 4x [CI] | Win | V-4.7 best base | V-4.7 LoRA B |",
                 "| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: |"]
        for r in rep["rows"]:
            h = r["headroom_vs_4x"]
            lines.append(f"| {r['task']} | {r['best16x']:.3f} ({r['best16x_arm']}) | {r['best4x']:.3f} ({r['best4x_arm']}) | {_ci(r['shift'])} | "
                         f"{r['loraB16x']:.3f} | {_ci(h)} | {'yes' if h['win'] else 'no'} | {r['v47_best_baseline']:.3f} | {r['v47_loraB']:.3f} |")
        return "\n".join(lines)
    lines = ["| Task | Best baseline (arm) | Per-task LoRA B | Multi-task LoRA | MT − best [CI] | Win | MT − LoRA B [CI] |",
             "| --- | ---: | ---: | ---: | --- | --- | --- |"]
    for r in rep["rows"]:
        lines.append(f"| {r['task']} | {r['best_baseline']:.3f} ({r['best_baseline_arm']}) | {r['loraB']:.3f} | {r['mt']:.3f} | "
                     f"{_ci(r['mt_vs_best'])} | {'yes' if r['mt_vs_best']['win'] else 'no'} | {_ci(r['mt_vs_loraB'])} |")
    return "\n".join(lines)


if __name__ == "__main__":
    name = sys.argv[1]
    d = Path("studies/pilots") / ("p03_ocr4x" if name == "p03" else "p01_multitask")
    pd = json.loads((d / "per_doc.json").read_text(encoding="utf-8"))
    rep = p03(pd) if name == "p03" else p01(pd)
    (d / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))
    print(markdown(name, rep))
