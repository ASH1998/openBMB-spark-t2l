"""Task pool build on Modal (CPU only). Protocol: studies/task_pool/protocol.md.

    modal run modal_app/task_pool.py::smoke             # 2 public sources (capped) + 2 templates into task_pool_smoke/
    modal run modal_app/task_pool.py::build             # every public source + every synthetic template
    modal run modal_app/task_pool.py::finalize          # duplicate checks, final train files, manifest
    modal run modal_app/task_pool.py::fetch             # copy manifest + reports into studies/task_pool/
    modal run modal_app/task_pool.py::inspect --path task_pool/manifest.json

Writes only under /vol/task_pool (or /vol/task_pool_smoke). Reads /vol/gate0/tasks to
check that the Gate 0 tasks are rebuilt with identical splits; never writes there.
Every entrypoint reserves money through minicpm_t2l.budget (stage "task_pool").
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

APP = os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-task-pool"
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = PurePosixPath("/vol")
FONT_APT = ("fonts-dejavu-core", "fonts-liberation", "fonts-noto-cjk", "fonts-arphic-ukai", "fonts-arphic-uming", "fonts-wqy-zenhei")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*FONT_APT)
    .pip_install("datasets>=4.3", "pillow==11.2.1", "numpy", "faker==40.54.0", "huggingface_hub>=1.10")
    .env({"HF_HOME": "/tmp/hf", "T2L_FONT_DIRS": "/usr/share/fonts"})
    .add_local_python_source("minicpm_t2l", "t2l")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


def _check_fonts() -> dict:
    """On Modal every font role must resolve to its first (Linux) candidate, so renders are reproducible."""
    from minicpm_t2l import synth

    got = {r: Path(synth.font_path(r)).name for r in synth.FONT_ROLES}
    bad = {r: f for r, f in got.items() if f != synth.FONT_ROLES[r][0]}
    if bad:
        raise RuntimeError(f"font roles not on their first candidate: {bad}")
    return got


@app.function(volumes={str(VOL): vol}, secrets=secrets, timeout=4 * 3600, cpu=2, memory=16384)
def build_public(source: str, out: str = "task_pool", scan_cap: int = 0) -> dict:
    from minicpm_t2l import pool
    from minicpm_t2l import tasks as g0

    t0 = time.time()
    root = Path(str(VOL / out))
    if scan_cap:  # smoke runs only
        s = pool.SOURCES[source]
        pool.SOURCES[source] = pool.PoolSource(**{**s.__dict__, "scan_train": scan_cap, "scan_eval": scan_cap})
    res = pool.build_public_source(source, root, Path("/tmp/raw"), log=lambda m: print(m, flush=True))
    # Gate 0 tasks must come out with exactly Gate 0's doc ids.
    checks = {}
    for t in pool.tasks_for(source):
        g = Path(str(VOL / "gate0/tasks")) / t.task_id
        if t.task_id in g0.TASKS and g.exists() and not scan_cap:
            ids = lambda p: [r["id"] for r in pool.read_jsonl(p)]
            checks[t.task_id] = {
                "eval_same": ids(g / "eval.jsonl") == ids(root / "tasks" / t.task_id / "eval.jsonl"),
                "train_same": ids(g / "train.jsonl") == ids(root / "tasks" / t.task_id / "train_candidates.jsonl")[: g0.TRAIN_CAP],
            }
    res["gate0_split_checks"] = checks
    res["seconds"] = time.time() - t0
    (root / "build_logs").mkdir(parents=True, exist_ok=True)
    (root / "build_logs" / f"{source}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    vol.commit()
    return res


@app.function(volumes={str(VOL): vol}, timeout=3 * 3600, cpu=1, memory=3072)
def build_synth(template_ids: list[str], out: str = "task_pool", n_per_split: int = 0) -> dict:
    from minicpm_t2l import pool

    fonts = _check_fonts()
    t0 = time.time()
    if n_per_split:  # smoke runs only
        pool.EVAL_N, pool.TRAIN_CAP, pool.TRAIN_RESERVE = n_per_split, n_per_split, 0
    res = pool.build_synth_templates(template_ids, Path(str(VOL / out)), log=lambda m: print(m, flush=True))
    vol.commit()
    return {"templates": res, "fonts": fonts, "seconds": time.time() - t0}


@app.function(volumes={str(VOL): vol}, timeout=3 * 3600, cpu=4, memory=32768)
def finalize_pool(out: str = "task_pool", min_train: int = 0) -> dict:
    from minicpm_t2l import dedup, pool

    vol.reload()
    root = Path(str(VOL / out))
    res = dedup.finalize(root, pool.TRAIN_CAP, min_train or pool.MIN_TRAIN, pool.MIN_EVAL_TRAINING_TASK,
                         log=lambda m: print(m, flush=True))
    vol.commit()
    return res


@app.function(volumes={str(VOL): vol}, timeout=600, cpu=1, memory=2048)
def read_files(paths: list[str]) -> dict:
    vol.reload()
    out = {}
    for p in paths:
        f = Path(str(VOL / p))
        if f.is_dir():
            out[p] = sorted(x.name for x in f.iterdir())[:500]
        elif f.exists():
            out[p] = f.read_text(encoding="utf-8") if f.stat().st_size < 50_000_000 else f"<{f.stat().st_size} bytes>"
        else:
            out[p] = None
    return out


@app.function(volumes={str(VOL): vol}, timeout=1800, cpu=2, memory=8192)
def task_cards(out: str = "task_pool") -> dict:
    vol.reload()
    root = Path(str(VOL / out)) / "tasks"
    return {d.name: json.loads((d / "task.json").read_text(encoding="utf-8")) for d in sorted(root.iterdir())
            if (d / "task.json").exists()}


# --------------------------------------------------------------------------- local entrypoints

def _guarded(run: str, usd: float, fn):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("task_pool", usd, run=run)
    t0 = time.time()
    try:
        result = fn()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    return result


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


@app.local_entrypoint()
def smoke():
    from minicpm_t2l import synth

    ts = [t.template_id for t in synth.make_templates()][:2]

    def go():
        pub = list(build_public.starmap([("sroie", "task_pool_smoke", 0), ("pubtabnet", "task_pool_smoke", 300)]))
        syn = build_synth.remote(ts, "task_pool_smoke", 3)
        fin = finalize_pool.remote("task_pool_smoke", 1)
        return {"public": pub, "synth": syn, "finalize": fin["report"], "tasks": fin["manifest"]["tasks"]}

    print(json.dumps(_guarded("task-pool-smoke", 1, go), indent=2, default=str)[:20000])


@app.local_entrypoint()
def build(sources: str = "all", templates: str = "all", chunk: int = 6, usd: float = 6):
    """`sources` / `templates`: "all", "none", or a comma-separated list (to re-run failures)."""
    from minicpm_t2l import pool, synth

    pick = lambda arg, every: every if arg == "all" else [] if arg == "none" else arg.split(",")
    srcs = pick(sources, list(pool.SOURCES))
    tids = pick(templates, [t.template_id for t in synth.make_templates()])

    def go():
        pub_calls = [build_public.spawn(s) for s in srcs]
        syn_calls = [build_synth.spawn(c) for c in _chunks(tids, chunk)]
        out = {"public": {}, "synth": {}, "errors": []}
        for s, c in zip(srcs, pub_calls):
            try:
                r = c.get()
                out["public"][s] = r
                print(f"[public] {s}: read {r['docs_read']}, saved {r['docs_saved']}, gate0 checks {r['gate0_split_checks']}", flush=True)
            except Exception as e:  # keep going; failures are logged and retried by re-running with --sources
                out["errors"].append({"source": s, "error": repr(e)})
                print(f"[public] {s}: FAILED {e!r}", flush=True)
        for i, c in enumerate(syn_calls):
            try:
                r = c.get()
                out["synth"].update(r["templates"])
                print(f"[synth] chunk {i}: {len(r['templates'])} templates in {r['seconds']:.0f}s", flush=True)
            except Exception as e:
                out["errors"].append({"chunk": i, "error": repr(e)})
                print(f"[synth] chunk {i}: FAILED {e!r}", flush=True)
        return out

    res = _guarded("task-pool-build", usd, go)
    Path("studies/task_pool/logs").mkdir(parents=True, exist_ok=True)
    Path(f"studies/task_pool/logs/build_result_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"errors: {res['errors']}")


@app.local_entrypoint()
def finalize(usd: float = 1):
    res = _guarded("task-pool-finalize", usd, lambda: finalize_pool.remote())
    print(json.dumps(res["report"], indent=2, default=str)[:5000])
    print(f"tasks kept: {len(res['manifest']['tasks'])}, dropped: {len(res['manifest']['dropped_tasks'])}")


@app.local_entrypoint()
def fetch():
    out = Path("studies/task_pool")
    files = read_files.remote(["task_pool/manifest.json", "task_pool/dedup_report.json"])
    for p, text in files.items():
        if text is not None:
            (out / Path(p).name).write_text(text, encoding="utf-8")
    cards = task_cards.remote()
    (out / "task_cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    logs = read_files.remote(["task_pool/build_logs"])["task_pool/build_logs"] or []
    got = read_files.remote([f"task_pool/build_logs/{n}" for n in logs])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for p, text in got.items():
        if text is not None:
            (out / "logs" / f"public_{Path(p).name}").write_text(text, encoding="utf-8")
    print(f"fetched manifest, dedup report, {len(cards)} task cards, {len(got)} build logs")


@app.local_entrypoint()
def inspect(path: str = "task_pool"):
    for p, v in read_files.remote([path]).items():
        print(p, json.dumps(v, indent=1)[:8000] if isinstance(v, list) else (v or "")[:8000])
