"""Gate 0 on Modal: build tasks, evaluate base / in-prompt / LoRA, train LoRAs.

NOT RUN YET. Every entrypoint reserves money via minicpm_t2l.budget first, which refuses
until the user confirms the cap in config/budget.json.

    modal run modal_app/gate0.py::probe                 # module names, both models (cheap)
    modal run modal_app/gate0.py::build                 # task dirs onto the volume (CPU)
    modal run modal_app/gate0.py::baselines --model v47 # base + k=1,4 in-prompt, 20 tasks
    modal run modal_app/gate0.py::loras --model v47 --targets A
    modal run modal_app/gate0.py::loras --model v47 --targets B

Layout on the volume (minicpm-t2l-artifacts):
    /vol/gate0/tasks/<task>/...              built by minicpm_t2l.tasks
    /vol/gate0/runs/<model>/<arm>/<task>/    predictions.jsonl, scores.json, (adapter/)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP = os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-gate0"
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")
GPU = "A10G"  # amendment 1 (2026-11-19): L4 supply too low; all arms on A10G
RUNS = "runs_a10g"

MODELS = {
    "v47": "openbmb/MiniCPM-V-4.7-1B",
    "ocr": "openbmb/MiniCPM-OCR-1.0",
}

# LoRA target sets. The language model is a Qwen3.5 hybrid: 6 of 24 layers are
# full attention (q/k/v/o_proj), 18 are linear attention (in_proj_qkv, in_proj_z,
# out_proj). Set A is the plan as written (q,v only -> 6 layers). Set B covers the
# attention projections of all 24 layers.
TARGETS = {
    "A": ("self_attn.q_proj", "self_attn.v_proj"),
    "B": ("self_attn.q_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj"),
}
EXPECTED_COUNTS = {"A": 12, "B": 12 + 36}
VISION_MARKERS = ("vision", "vpm", "visual", "resampler", "merger", "siglip", "mtp")

# Frozen in studies/gate0/protocol.md -- change there first (dated amendment).
GEN = dict(max_new_tokens=768, do_sample=False, repetition_penalty=1.0)
PROC = dict(downsample_mode="16x", max_slice_nums=9, enable_thinking=False)
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05, lr=2e-4, epochs=3, grad_accum=8, warmup_ratio=0.05)
ICL_KS = (1, 4)
ICL_SEED = 7

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "flash-linear-attention", "transformers>=5.21", "peft>=0.19", "accelerate", "datasets>=4.3",
                 "pillow", "huggingface_hub>=1.10", "python-dotenv")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("minicpm_t2l", "t2l")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
# Only the HF token goes to Modal; other .env credentials stay local.
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


# --------------------------------------------------------------------------- helpers (remote)

def _load(model_key: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mid = MODELS[model_key]
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda")
    return model, proc


def _lm_targets(model, target_set: str) -> list[str]:
    suffixes = TARGETS[target_set]
    names = [n for n, _ in model.named_modules()
             if n.endswith(suffixes) and not any(m in n.lower() for m in VISION_MARKERS)]
    if len(names) != EXPECTED_COUNTS[target_set]:
        raise RuntimeError(f"target set {target_set}: expected {EXPECTED_COUNTS[target_set]} modules, got {len(names)}: {names[:8]}")
    return names


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _user_turn(img_path: Path, prompt: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": str(img_path)}, {"type": "text", "text": prompt}]}


def _messages(tdir: Path, prompt: str, doc: dict, shots: list[dict]) -> list[dict]:
    msgs = []
    for s in shots:
        msgs.append(_user_turn(tdir / s["image"], prompt))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(s["label"], ensure_ascii=False)}]})
    msgs.append(_user_turn(tdir / doc["image"], prompt))
    return msgs


def _generate(model, proc, msgs) -> str:
    import torch

    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", **PROC).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, downsample_mode=PROC["downsample_mode"], **GEN)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _eval_task(model, proc, task: str, arm: str, out_dir: Path, k: int = 0, limit: int = 0) -> dict:
    import random

    from minicpm_t2l.scoring import TaskScore, score_doc

    if not limit and (out_dir / "scores.json").exists():
        return json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    tdir = VOL / "gate0/tasks" / task
    meta = json.loads((tdir / "task.json").read_text(encoding="utf-8"))
    train, evals = _read_jsonl(tdir / "train.jsonl"), _read_jsonl(tdir / "eval.jsonl")
    shots = random.Random(ICL_SEED).sample(train, k) if k else []
    if limit:
        evals = evals[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, t0 = TaskScore(), time.time()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for doc in evals:
            text = _generate(model, proc, _messages(tdir, meta["prompt"], doc, shots))
            d = score_doc(text, doc["label"])
            ts.add(d)
            fh.write(json.dumps({"id": doc["id"], "output": text, "tp": d.tp, "n_pred": d.n_pred,
                                 "n_gold": d.n_gold, "parsed": d.parsed}, ensure_ascii=False) + "\n")
    summ = {**ts.summary(), "task": task, "arm": arm, "k": k, "shots": [s["id"] for s in shots],
            "gen": GEN, "proc": PROC, "seconds": time.time() - t0}
    (out_dir / "scores.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    vol.commit()
    return summ


# --------------------------------------------------------------------------- remote functions

@app.function(volumes={VOL: vol}, secrets=secrets, timeout=1800, cpu=2, memory=8192)
def probe_modules(model_key: str) -> dict:
    """Instantiate on the meta device (no weights) and report LM linear module names."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(MODELS[model_key])
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    res = {"n_linear": len(linear), "sample": linear[:40] + linear[-20:]}
    for ts in TARGETS:
        try:
            res[f"targets_{ts}"] = len(_lm_targets(model, ts))
        except RuntimeError as e:
            res[f"targets_{ts}"] = str(e)
    return res


@app.function(volumes={VOL: vol}, secrets=secrets, timeout=6 * 3600, cpu=4, memory=16384)
def build_tasks(sources: list[str] | None = None) -> list[str]:
    from minicpm_t2l.tasks import SOURCES, build_source

    for s in sources or list(SOURCES):
        build_source(s, VOL / "gate0/tasks", VOL / "gate0/raw")
        vol.commit()
    return sorted(p.name for p in (VOL / "gate0/tasks").iterdir())


@app.function(gpu=GPU, volumes={VOL: vol}, secrets=secrets, timeout=6 * 3600)
def run_baselines(model_key: str, tasks: list[str], runs: str = RUNS, limit: int = 0) -> list[dict]:
    model, proc = _load(model_key)
    model.eval()
    res = []
    for task in tasks:
        base = VOL / "gate0" / runs / model_key
        res.append(_eval_task(model, proc, task, "base", base / "base" / task, limit=limit))
        for k in ICL_KS:
            res.append(_eval_task(model, proc, task, f"icl{k}", base / f"icl{k}" / task, k=k, limit=limit))
    return res


@app.function(gpu=GPU, volumes={VOL: vol}, secrets=secrets, timeout=3 * 3600)
def eval_arm(model_key: str, task: str, k: int) -> dict:
    """One baseline arm (k=0 base, else in-prompt k) for one task, in its own container."""
    arm = "base" if k == 0 else f"icl{k}"
    out = VOL / "gate0" / RUNS / model_key / arm / task
    if (out / "scores.json").exists():
        return json.loads((out / "scores.json").read_text(encoding="utf-8"))
    model, proc = _load(model_key)
    model.eval()
    return _eval_task(model, proc, task, arm, out, k=k)


@app.function(gpu=GPU, volumes={VOL: vol}, secrets=secrets, timeout=5 * 3600)
def job(kind: str, model_key: str, task: str, arg) -> dict:
    """Generic worker, one job per container (workspace limit: 10 concurrent GPUs).

    Not @modal.concurrent: Triton's autotuner (used by flash-linear-attention) is not
    thread-safe and crashed with several jobs in one process (2026-11-19).
    """
    if kind == "eval":
        return eval_arm.local(model_key, task, arg)
    if kind == "lora":
        return train_and_eval_lora.local(model_key, task, arg)
    raise ValueError(kind)


@app.function(volumes={VOL: vol}, secrets=secrets, timeout=3600, cpu=2, memory=8192)
def fetch_weights() -> list[str]:
    """Cache both models' files on the volume once, before many containers read them."""
    from huggingface_hub import snapshot_download

    paths = [snapshot_download(m) for m in MODELS.values()]
    vol.commit()
    return paths


@app.function(gpu=GPU, volumes={VOL: vol}, secrets=secrets, timeout=4 * 3600)
def train_and_eval_lora(model_key: str, task: str, target_set: str, runs: str = RUNS, limit: int = 0) -> dict:
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    out = VOL / "gate0" / runs / model_key / f"lora{target_set}" / task
    if not limit and (out / "scores.json").exists():
        return json.loads((out / "scores.json").read_text(encoding="utf-8"))
    model, proc = _load(model_key)
    targets = _lm_targets(model, target_set)
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"],
                                             lora_dropout=LORA["lora_dropout"], target_modules=targets))
    tdir = VOL / "gate0/tasks" / task
    meta = json.loads((tdir / "task.json").read_text(encoding="utf-8"))
    train = _read_jsonl(tdir / "train.jsonl")
    if limit:
        train = train[:limit]

    def encode(doc):
        prompt_msgs = [_user_turn(tdir / doc["image"], meta["prompt"])]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(doc["label"], ensure_ascii=False)}]}]
        p = proc.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                     return_tensors="pt", **PROC)
        f = proc.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=True,
                                     return_tensors="pt", **PROC)
        labels = f["input_ids"].clone()
        labels[:, : p["input_ids"].shape[1]] = -100
        f["labels"] = labels
        return f

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LORA["lr"], weight_decay=0.0)
    steps = LORA["epochs"] * len(train) // LORA["grad_accum"]
    sched = get_cosine_schedule_with_warmup(opt, int(LORA["warmup_ratio"] * steps), max(steps, 1))
    model.train()
    rng, losses, t0, i = random.Random(0), [], time.time(), 0
    for _ in range(LORA["epochs"]):
        order = train[:]
        rng.shuffle(order)
        for doc in order:
            batch = encode(doc).to(model.device)
            loss = model(**batch, downsample_mode=PROC["downsample_mode"]).loss / LORA["grad_accum"]
            loss.backward()
            losses.append(loss.item() * LORA["grad_accum"])
            i += 1
            if i % LORA["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
    model.save_pretrained(out / "adapter")
    (out / "train_log.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps({"losses": losses, "lora": LORA, "targets": targets,
                                                    "seconds": time.time() - t0}), encoding="utf-8")
    model.eval()
    return _eval_task(model, proc, task, f"lora{target_set}", out, limit=limit)


# --------------------------------------------------------------------------- local entrypoints (budget-guarded)

def _guarded(run: str, usd: float, fn):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("gate0", usd, run=run)
    t0 = time.time()
    try:
        result = fn()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    # actual cost is reconciled later from the Modal billing page; keep the hold until then
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    return result


@app.local_entrypoint()
def probe():
    print(json.dumps(_guarded("gate0-probe", 1, lambda: {m: probe_modules.remote(m) for m in MODELS}), indent=2))


@app.local_entrypoint()
def build(sources: str = ""):
    print(_guarded("gate0-build", 2, lambda: build_tasks.remote(sources.split(",") if sources else None)))


@app.local_entrypoint()
def baselines(model: str = "v47", tasks: str = ""):
    from minicpm_t2l.tasks import TASKS

    names = tasks.split(",") if tasks else list(TASKS)
    res = _guarded(f"gate0-baselines-{model}", 6, lambda: run_baselines.remote(model, names))
    for r in res:
        print(f"{r['task']:18s} {r['arm']:6s} f1={r['f1']:.3f} parse_fail={r['parse_fail_rate']:.2f}")


@app.local_entrypoint()
def loras(model: str = "v47", targets: str = "A", tasks: str = ""):
    from minicpm_t2l.tasks import TASKS

    names = tasks.split(",") if tasks else list(TASKS)
    res = _guarded(f"gate0-lora{targets}-{model}", 8,
                   lambda: list(train_and_eval_lora.starmap([(model, t, targets) for t in names])))
    for r in res:
        print(f"{r['task']:18s} {r['arm']:6s} f1={r['f1']:.3f} parse_fail={r['parse_fail_rate']:.2f}")


@app.local_entrypoint()
def smoke(model: str = "v47", task: str = "sroie_total"):
    """Tiny end-to-end check (3 eval docs, 16 train docs) into gate0/smoke/, not part of results."""
    def go():
        b = run_baselines.remote(model, [task], runs="smoke", limit=3)
        l = train_and_eval_lora.remote(model, task, "B", runs="smoke", limit=16)
        return b + [l]
    for r in _guarded(f"gate0-smoke-{model}", 1, go):
        print(f"{r['task']:18s} {r['arm']:6s} f1={r['f1']:.3f} parse_fail={r['parse_fail_rate']:.2f} s={r['seconds']:.0f}")


@app.local_entrypoint()
def everything(models: str = "v47,ocr", targets: str = "A,B", usd: float = 40):
    """Fan out all Gate 0 work at once: every (model, task, arm) in its own container.

    Finished results on the volume are reused, so re-running only does what is missing.
    """
    from minicpm_t2l.tasks import TASKS

    ms, ts = models.split(","), targets.split(",")

    def go():
        print("weights:", fetch_weights.remote(), flush=True)
        # Order = rough execution order under a small GPU limit: finish v47 first.
        jobs = []
        for m in ms:
            jobs += [("eval", m, t, k) for t in TASKS for k in (0, *ICL_KS)]
            jobs += [("lora", m, t, s) for s in reversed(ts) for t in TASKS]
        calls = [job.spawn(*j) for j in jobs]
        print(f"spawned {len(calls)} jobs", flush=True)
        results, failures = [], []
        for (kind, m, t, arg), call in zip(jobs, calls):
            label = f"k{arg}" if kind == "eval" else f"lora{arg}"
            try:
                r = call.get()
                results.append(r)
                print(f"{m:4s} {r['task']:18s} {r['arm']:6s} f1={r['f1']:.3f} "
                      f"parse_fail={r['parse_fail_rate']:.2f}", flush=True)
            except Exception as e:  # keep collecting; report failures at the end
                failures.append((m, t, label, f"{type(e).__name__}: {e}"[:300]))
                print(f"FAILED {m} {t} {label}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return results, failures

    results, failures = _guarded("gate0-everything", usd, go)
    print()
    print(f"done: {len(results)} ok, {len(failures)} failed")
    for f in failures:
        print("FAILED", *f)
