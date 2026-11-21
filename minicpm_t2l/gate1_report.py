"""Gate 1 analysis, as fixed in studies/gate1/protocol.md.

Input: studies/gate1/per_doc.json from `modal run modal_app/gate1.py::pull`
({"gate1/eval/<arm>/<task>": {"scores", "docs"}}).

    python -m minicpm_t2l.gate1_report
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from minicpm_t2l.pilot_report import compare, f1_of

ARMS = ("base", "icl1", "icl4", "mt", "oracle", "emit1", "emit4", "emit8")
PRIMARY = "emit4"
N_BOOT = 2000


def task_bootstrap(diffs: list[float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean of per-task differences with a 95% CI from resampling tasks."""
    if not diffs:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(diffs) / n, "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]], "n": n}


def group_of(task: str) -> str:
    return "synthetic" if task.startswith("syn_") else "public"


def analyse(pd: dict, tasks: list[str], arms: tuple[str, ...] = ARMS) -> dict:
    """`arms` limits the analysis to the arms that were run (amendment 4: base and emit4 only).
    Comparisons that need a missing arm are reported as not run."""
    f1 = {a: {} for a in arms}
    missing = []
    for a in arms:
        for t in tasks:
            r = pd.get(f"gate1/eval/{a}/{t}")
            if r is None:
                missing.append(f"{a}/{t}")
            else:
                f1[a][t] = f1_of(r)
    complete = [t for t in tasks if all(t in f1[a] for a in arms)]
    out = {"n_tasks": len(tasks), "n_complete": len(complete), "missing": missing, "groups": {}}
    for gname, members in (("all", complete), ("synthetic", [t for t in complete if group_of(t) == "synthetic"]),
                           ("public", [t for t in complete if group_of(t) == "public"])):
        if not members:
            continue
        mean = {a: sum(f1[a][t] for t in members) / len(members) for a in arms}
        prim = task_bootstrap([f1[PRIMARY][t] - f1["base"][t] for t in members])
        g = {"n": len(members), "mean_f1": mean, "primary_emit4_minus_base": prim, "primary_pass": prim["ci95"][0] > 0,
             "better_tasks": sum(f1[PRIMARY][t] > f1["base"][t] + 0.005 for t in members),
             "worse_tasks": sum(f1[PRIMARY][t] < f1["base"][t] - 0.005 for t in members)}
        if "mt" in arms and "oracle" in arms:
            vs_mt = task_bootstrap([f1[PRIMARY][t] - f1["mt"][t] for t in members])
            gain_or = sum(f1["oracle"][t] - f1["base"][t] for t in members)
            gain_em = sum(f1[PRIMARY][t] - f1["base"][t] for t in members)
            g.update(emit4_minus_mt=vs_mt, oracle_gain_recovered=gain_em / gain_or if gain_or > 0 else float("nan"),
                     full_bar_pass=prim["ci95"][0] > 0 and vs_mt["ci95"][0] > 0 and gain_or > 0 and gain_em / gain_or >= 0.5)
        else:
            g.update(emit4_minus_mt=None, oracle_gain_recovered=None, full_bar_pass="not run (amendment 4)")
        out["groups"][gname] = g
    rows = []
    for t in complete:
        c = compare(pd[f"gate1/eval/base/{t}"], pd[f"gate1/eval/{PRIMARY}/{t}"])
        rows.append({"task": t, "group": group_of(t), **{a: f1[a][t] for a in arms}, "emit4_vs_base": c})
    out["rows"] = rows
    out["arms"] = list(arms)
    out["per_task_wins_emit4_vs_base"] = sum(r["emit4_vs_base"]["win"] for r in rows)
    out["shipped"] = out["groups"].get("all", {}).get("primary_pass", False)
    return out


def markdown(rep: dict) -> str:
    arms = rep.get("arms", list(ARMS))
    lines = ["| Group | n | " + " | ".join(arms) + " | emit4 - base [95% CI] | better / worse tasks |",
             "| --- | ---: |" + " ---: |" * len(arms) + " --- | --- |"]
    for g, v in rep["groups"].items():
        m, p = v["mean_f1"], v["primary_emit4_minus_base"]
        lines.append(f"| {g} | {v['n']} | " + " | ".join(f"{m[a]:.3f}" for a in arms)
                     + f" | {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {v['better_tasks']} / {v['worse_tasks']} |")
    lines += ["", "| Task | " + " | ".join(arms) + " |", "| --- |" + " ---: |" * len(arms)]
    for r in sorted(rep["rows"], key=lambda r: (r["group"], r["task"])):
        lines.append(f"| {r['task']} | " + " | ".join(f"{r[a]:.3f}" for a in arms) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    pd = json.loads(Path("studies/gate1/per_doc.json").read_text(encoding="utf-8"))
    held = json.loads(Path("studies/task_pool/heldout_2026-11-19.json").read_text(encoding="utf-8"))["heldout_tasks"]
    import sys

    arms = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ARMS
    rep = analyse(pd, held, arms)
    Path("studies/gate1/report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows",)}, indent=2)[:4000])
    print(markdown(rep))
