"""Pilots P-01 (multi-task LoRA) and P-03 (OCR-1.0 at 4x) on Modal.

Protocols: studies/pilots/p01_multitask/protocol.md, studies/pilots/p03_ocr4x/protocol.md.

    modal run modal_app/pilots.py::p03          # 20 tasks x {base, icl1, icl4}, OCR-1.0 at 4x
    modal run modal_app/pilots.py::p01          # one LoRA on all 20 Gate 0 train splits, then eval per task
    modal run modal_app/pilots.py::pull --pilot p03   # per-doc scores (this pilot + Gate 0) -> studies/pilots/<pilot>/

Reads gate0/tasks and gate0/runs_a10g (never writes there). Writes under pilots/<pilot>/.
A run directory counts as finished only when its scores.json exists.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path, PurePosixPath

import modal

APP = os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-pilots"
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")
GPU = "A10G"
OCR = "openbmb/MiniCPM-OCR-1.0"

# Same as Gate 0 (studies/gate0/protocol.md) except where a pilot protocol says otherwise.
GEN = dict(max_new_tokens=768, do_sample=False, repetition_penalty=1.0)
PROC_16X = dict(downsample_mode="16x", max_slice_nums=9, enable_thinking=False)
PROC_4X = dict(downsample_mode="4x", max_slice_nums=9, enable_thinking=False)
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05, lr=2e-4, epochs=3, grad_accum=8, warmup_ratio=0.05)
TARGETS_B = ("self_attn.q_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj")
EXPECTED_B = 48
VISION_MARKERS = ("vision", "vpm", "visual", "resampler", "merger", "siglip", "mtp")
ICL_SEED = 7
P01_SEED = 0
MAX_TRAIN_SECONDS = 3 * 3600

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "flash-linear-attention", "transformers>=5.21", "peft>=0.19",
                 "accelerate", "datasets>=4.3", "pillow", "huggingface_hub>=1.10")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("minicpm_t2l", "t2l")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]
G0_TASKS = VOL / "gate0/tasks"


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(OCR)
    model = AutoModelForImageTextToText.from_pretrained(OCR, torch_dtype=torch.bfloat16, device_map="cuda")
    return model, proc


def _user_turn(img_path: Path, prompt: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": str(img_path)}, {"type": "text", "text": prompt}]}


def _messages(tdir: Path, prompt: str, doc: dict, shots: list[dict]) -> list[dict]:
    msgs = []
    for s in shots:
        msgs.append(_user_turn(tdir / s["image"], prompt))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(s["label"], ensure_ascii=False)}]})
    msgs.append(_user_turn(tdir / doc["image"], prompt))
    return msgs


def _generate(model, proc, msgs, p: dict) -> str:
    import torch

    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", **p).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, downsample_mode=p["downsample_mode"], **GEN)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _eval_task(model, proc, task: str, arm: str, out_dir: Path, p: dict, k: int = 0) -> dict:
    from minicpm_t2l.scoring import TaskScore, score_doc

    if (out_dir / "scores.json").exists():
        return json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    tdir = G0_TASKS / task
    meta = json.loads((tdir / "task.json").read_text(encoding="utf-8"))
    train, evals = _read_jsonl(tdir / "train.jsonl"), _read_jsonl(tdir / "eval.jsonl")
    shots = random.Random(ICL_SEED).sample(train, k) if k else []
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, t0 = TaskScore(), time.time()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for doc in evals:
            text = _generate(model, proc, _messages(tdir, meta["prompt"], doc, shots), p)
            d = score_doc(text, doc["label"])
            ts.add(d)
            fh.write(json.dumps({"id": doc["id"], "output": text, "tp": d.tp, "n_pred": d.n_pred,
                                 "n_gold": d.n_gold, "parsed": d.parsed}, ensure_ascii=False) + "\n")
    summ = {**ts.summary(), "task": task, "arm": arm, "k": k, "shots": [s["id"] for s in shots],
            "gen": GEN, "proc": p, "seconds": time.time() - t0}
    (out_dir / "scores.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    vol.commit()
    return summ


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=3 * 3600)
def p03_job(task: str, k: int) -> dict:
    """One (task, arm) of P-03 per container (no in-process concurrency: Triton autotuner race)."""
    arm = "base" if k == 0 else f"icl{k}"
    out = VOL / "pilots/p03_ocr4x" / arm / task
    if (out / "scores.json").exists():
        return json.loads((out / "scores.json").read_text(encoding="utf-8"))
    model, proc = _load()
    model.eval()
    return _eval_task(model, proc, task, arm, out, PROC_4X, k=k)


def _lm_targets(model) -> list[str]:
    names = [n for n, _ in model.named_modules()
             if n.endswith(TARGETS_B) and not any(m in n.lower() for m in VISION_MARKERS)]
    if len(names) != EXPECTED_B:
        raise RuntimeError(f"target set B: expected {EXPECTED_B} modules, got {len(names)}")
    return names


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=8 * 3600)
def p01_train(tasks: list[str]) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    out = VOL / "pilots/p01_multitask"
    if (out / "train_log.json").exists():
        return json.loads((out / "train_log.json").read_text(encoding="utf-8")) | {"reused": True}
    model, proc = _load()
    targets = _lm_targets(model)
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"],
                                             lora_dropout=LORA["lora_dropout"], target_modules=targets))
    examples = []
    for t in tasks:
        tdir = G0_TASKS / t
        prompt = json.loads((tdir / "task.json").read_text(encoding="utf-8"))["prompt"]
        examples += [(tdir, prompt, d) for d in _read_jsonl(tdir / "train.jsonl")]

    def encode(tdir, prompt, doc):
        pm = [_user_turn(tdir / doc["image"], prompt)]
        fm = pm + [{"role": "assistant", "content": [{"type": "text", "text": json.dumps(doc["label"], ensure_ascii=False)}]}]
        p = proc.apply_chat_template(pm, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **PROC_16X)
        f = proc.apply_chat_template(fm, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt", **PROC_16X)
        labels = f["input_ids"].clone()
        labels[:, : p["input_ids"].shape[1]] = -100
        f["labels"] = labels
        return f

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LORA["lr"], weight_decay=0.0)
    steps = LORA["epochs"] * len(examples) // LORA["grad_accum"]
    sched = get_cosine_schedule_with_warmup(opt, int(LORA["warmup_ratio"] * steps), max(steps, 1))
    model.train()
    rng, losses, t0, i = random.Random(P01_SEED), [], time.time(), 0
    for _ in range(LORA["epochs"]):
        order = examples[:]
        rng.shuffle(order)
        for tdir, prompt, doc in order:
            if time.time() - t0 > MAX_TRAIN_SECONDS:  # protocol: stop if training exceeds 3 GPU-hours
                raise RuntimeError(f"P-01 training exceeded {MAX_TRAIN_SECONDS} s after {i} examples")
            batch = encode(tdir, prompt, doc).to(model.device)
            loss = model(**batch, downsample_mode=PROC_16X["downsample_mode"]).loss / LORA["grad_accum"]
            loss.backward()
            losses.append(loss.item() * LORA["grad_accum"])
            i += 1
            if i % LORA["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
    model.save_pretrained(out / "adapter")
    log = {"n_examples": len(examples), "optimizer_steps": steps, "losses": losses, "lora": LORA, "targets": targets,
           "tasks": tasks, "seconds": time.time() - t0}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    vol.commit()
    return {k: v for k, v in log.items() if k != "losses"}


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=3 * 3600)
def p01_eval(task: str) -> dict:
    from peft import PeftModel

    vol.reload()
    out = VOL / "pilots/p01_multitask/eval" / task
    if (out / "scores.json").exists():
        return json.loads((out / "scores.json").read_text(encoding="utf-8"))
    model, proc = _load()
    model = PeftModel.from_pretrained(model, str(VOL / "pilots/p01_multitask/adapter"))
    model.eval()
    return _eval_task(model, proc, task, "lora_mt", out, PROC_16X)


@app.function(volumes={str(VOL): vol}, timeout=1800, cpu=1, memory=4096)
def per_doc(paths: list[str]) -> dict:
    """Per-doc score rows (no outputs) for each predictions.jsonl path, plus its scores.json."""
    vol.reload()
    res = {}
    for p in paths:
        d = VOL / p
        if not (d / "scores.json").exists():
            res[p] = None
            continue
        rows = [{k: r[k] for k in ("id", "tp", "n_pred", "n_gold", "parsed")} for r in _read_jsonl(d / "predictions.jsonl")]
        res[p] = {"scores": json.loads((d / "scores.json").read_text(encoding="utf-8")), "docs": rows}
    return res


@app.function(volumes={str(VOL): vol}, secrets=secrets, timeout=1800, cpu=2, memory=8192)
def check_proc(task: str = "cord_total") -> dict:
    """CPU only: does downsample_mode change the number of input tokens? (processor, no model)."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(OCR)
    tdir = G0_TASKS / task
    doc = _read_jsonl(tdir / "eval.jsonl")[0]
    prompt = json.loads((tdir / "task.json").read_text(encoding="utf-8"))["prompt"]
    msgs = [_user_turn(tdir / doc["image"], prompt)]
    out = {}
    for name, p in (("16x", PROC_16X), ("4x", PROC_4X)):
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **p)
        out[name] = {"input_ids": int(x["input_ids"].shape[1]),
                     "keys": {k: list(v.shape) if hasattr(v, "shape") else str(type(v)) for k, v in x.items()}}
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                 processor_kwargs={"downsample_mode": "4x", "max_slice_nums": 9}, enable_thinking=False)
    out["4x_via_processor_kwargs"] = {"input_ids": int(x["input_ids"].shape[1])}
    return out


# --------------------------------------------------------------------------- local entrypoints

def _guarded(run: str, usd: float, fn):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("overhead", usd, run=run)
    t0 = time.time()
    try:
        result = fn()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    return result


@app.local_entrypoint()
def p03(usd: float = 7.5, tasks: str = ""):
    from minicpm_t2l.tasks import TASKS

    names = tasks.split(",") if tasks else list(TASKS)
    # Long tasks first so the tail is short.
    order = sorted(names, key=lambda t: (not t.startswith("xfund"), t))
    jobs = [(t, k) for t in order for k in (0, 1, 4)]

    def go():
        out = []
        for r in p03_job.starmap(jobs, return_exceptions=True):
            if isinstance(r, Exception):
                print(f"FAILED: {r!r}", flush=True)
                out.append({"error": repr(r)})
            else:
                print(f"{r['task']:18s} {r['arm']:6s} f1={r['f1']:.3f} parse_fail={r['parse_fail_rate']:.2f} s={r['seconds']:.0f}", flush=True)
                out.append(r)
        return out

    res = _guarded("p03-ocr4x", usd, go)
    print(f"done: {sum('f1' in r for r in res)} / {len(jobs)}")


@app.local_entrypoint()
def p01(usd: float = 4.5):
    from minicpm_t2l.tasks import TASKS

    names = list(TASKS)

    def go():
        log = p01_train.remote(names)
        print("trained:", json.dumps(log)[:600], flush=True)
        out = []
        for r in p01_eval.map(names, return_exceptions=True):
            if isinstance(r, Exception):
                print(f"FAILED: {r!r}", flush=True)
                out.append({"error": repr(r)})
            else:
                print(f"{r['task']:18s} lora_mt f1={r['f1']:.3f} parse_fail={r['parse_fail_rate']:.2f} s={r['seconds']:.0f}", flush=True)
                out.append(r)
        return out

    res = _guarded("p01-multitask-lora", usd, go)
    print(f"done: {sum('f1' in r for r in res)} / {len(names)}")


@app.local_entrypoint()
def pull(pilot: str = "p03"):
    from minicpm_t2l.tasks import TASKS

    g0 = [f"gate0/runs_a10g/{m}/{a}/{t}" for m in ("ocr", "v47") for a in ("base", "icl1", "icl4", "loraB") for t in TASKS]
    mine = ([f"pilots/p03_ocr4x/{a}/{t}" for a in ("base", "icl1", "icl4") for t in TASKS] if pilot == "p03"
            else [f"pilots/p01_multitask/eval/{t}" for t in TASKS])
    res = per_doc.remote(g0 + mine)
    out = Path(f"studies/pilots/{ 'p03_ocr4x' if pilot == 'p03' else 'p01_multitask'}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_doc.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"pulled {sum(v is not None for v in res.values())} / {len(res)} runs into {out}/per_doc.json")


@app.local_entrypoint()
def proc_check():
    print(json.dumps(_guarded("pilot-proc-check", 0.3, lambda: check_proc.remote()), indent=1))
