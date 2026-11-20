"""Gate 1 on Modal. Protocol: studies/gate1/protocol.md (frozen 2026-11-20).

    modal run modal_app/gate1.py::smoke                    # step 0: shapes, hidden states, 3 train steps, emit + generate, PEFT round trip
    modal run modal_app/gate1.py::embed                    # step 1: cache example embeddings for every train-split doc
    modal run modal_app/gate1.py::dev --runs lr1e-4,...    # step 2: short runs + dev evaluation
    modal run modal_app/gate1.py::final --run-config ...   # step 3 (+ step 4 multi-task LoRA in parallel)
    modal run modal_app/gate1.py::evaluate                 # steps 5-6: oracles + all arms on the 78 held-out tasks
    modal run modal_app/gate1.py::pull                     # per-doc scores -> studies/gate1/

Reads task_pool/ (never writes there). Writes under gate1/. A run directory counts as finished
only when its scores.json (eval) or final.pt / adapter (training) exists.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import modal

APP = os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-gate1"
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")
POOL = VOL / "task_pool"
OUT = VOL / "gate1"
GPU = "A10G"

# Named image built by modal_app/images.py: torch 2.10 + causal-conv1d kernel (2026-11-20, protocol amendment 3).
image = (
    modal.Image.from_name(os.environ.get("T2L_GPU_IMAGE", "minicpm-t2l-gpu"))
    .add_local_python_source("minicpm_t2l", "t2l")
    .add_local_file("studies/task_pool/manifest.json", "/cfg/manifest.json")
    .add_local_file("studies/task_pool/heldout_2026-11-19.json", "/cfg/heldout.json")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


# --------------------------------------------------------------------------- shared helpers (remote)

def _splits():
    from minicpm_t2l import gate1 as g

    return g.make_splits(json.loads(Path("/cfg/manifest.json").read_text(encoding="utf-8")),
                         json.loads(Path("/cfg/heldout.json").read_text(encoding="utf-8")))


def _load():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from minicpm_t2l import gate1 as g

    proc = AutoProcessor.from_pretrained(g.MODEL)
    model = AutoModelForImageTextToText.from_pretrained(g.MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


def _targets(model):
    from minicpm_t2l import gate1 as g
    from t2l import find_targets

    return find_targets(model, g.TARGET_PATTERNS, exclude=g.TARGET_EXCLUDE, expected=g.N_TARGETS)


def _user(row: dict, prompt: str) -> dict:
    return {"role": "user", "content": [{"type": "image", "path": str(POOL / row["image"])}, {"type": "text", "text": prompt}]}


def _asst(label) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": json.dumps(label, ensure_ascii=False)}]}


def _encode(proc, row: dict, prompt: str):
    """Full example (prompt + gold JSON) with labels on the answer tokens only."""
    from minicpm_t2l import gate1 as g

    pm = [_user(row, prompt)]
    p = proc.apply_chat_template(pm, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **g.PROC)
    f = proc.apply_chat_template(pm + [_asst(row["label"])], tokenize=True, add_generation_prompt=False, return_dict=True,
                                 return_tensors="pt", **g.PROC)
    labels = f["input_ids"].clone()
    labels[:, : p["input_ids"].shape[1]] = -100
    f["labels"] = labels
    return f


def _forward(model, batch, **kw):
    from minicpm_t2l import gate1 as g

    return model(**batch, downsample_mode=g.PROC["downsample_mode"], **kw)


def _final_hidden(model, batch):
    """Last language-model hidden state [1, T, d] for a batch (labels removed)."""
    b = {k: v for k, v in batch.items() if k != "labels"}
    out = _forward(model, b, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("model did not return hidden_states")
    return hs[-1]


def _generate(model, proc, msgs) -> str:
    import torch

    from minicpm_t2l import gate1 as g

    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", **g.PROC).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, downsample_mode=g.PROC["downsample_mode"], **g.GEN)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _eval_rows(model, proc, prompt: str, rows: list[dict], shots: list[dict], out_dir: Path, meta: dict) -> dict:
    from minicpm_t2l.scoring import TaskScore, score_doc

    if (out_dir / "scores.json").exists():
        return json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, t0 = TaskScore(), time.time()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            msgs = []
            for s in shots:
                msgs += [_user(s, prompt), _asst(s["label"])]
            text = _generate(model, proc, msgs + [_user(row, prompt)])
            d = score_doc(text, row["label"])
            ts.add(d)
            fh.write(json.dumps({"id": row["id"], "output": text, "tp": d.tp, "n_pred": d.n_pred, "n_gold": d.n_gold,
                                 "parsed": d.parsed}, ensure_ascii=False) + "\n")
    summ = {**ts.summary(), **meta, "shots": [s["id"] for s in shots], "seconds": time.time() - t0}
    (out_dir / "scores.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    vol.commit()
    return summ


def _load_emb(task: str):
    import torch

    d = torch.load(OUT / "emb" / f"{task}.pt", map_location="cpu")
    return d["ids"], d["emb"]


def _task_emb(task: str, ids_wanted: list[str], variant: str):
    import torch

    from t2l import task_embedding

    ids, emb = _load_emb(task)
    pos = {i: j for j, i in enumerate(ids)}
    e = task_embedding(emb[[pos[i] for i in ids_wanted]].float())
    return _variant(e, variant)


def _variant(e, variant: str):
    if variant == "both":
        return e
    if variant == "answer":
        return e[e.shape[-1] // 2:]
    raise ValueError(variant)


def _emb_dim(variant: str) -> int:
    return 2048 if variant == "both" else 1024


def _new_generator(targets, variant: str):
    from minicpm_t2l import gate1 as g
    from t2l import HyperLoRA

    return HyperLoRA(targets, emb_dim=_emb_dim(variant), rank=g.RANK, alpha=g.ALPHA).cuda()


def _guarded(stage: str, run: str, usd: float, fn):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve(stage, usd, run=run)
    t0 = time.time()
    try:
        result = fn()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    return result


# --------------------------------------------------------------------------- step 0: smoke

@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=3600)
def smoke_job() -> dict:
    import torch

    from minicpm_t2l import gate1 as g
    from t2l import LoRAInjector, example_embedding, load_peft_adapter, save_peft_adapter

    sp = _splits()
    model, proc = _load()
    targets = _targets(model)
    shapes = sorted({(t.kind, t.in_features, t.out_features) for t in targets})
    task = sp.dev[0]
    card, train, ev = g.read_task(POOL, task)
    batch = _encode(proc, train[0], card["prompt"]).to(model.device)
    with torch.no_grad():
        h = _final_hidden(model, batch)
    e = example_embedding(h, batch["attention_mask"], (batch["labels"] != -100).long())
    gen = _new_generator(targets, "both")
    n_params = sum(p.numel() for p in gen.parameters())
    opt = torch.optim.AdamW(gen.parameters(), lr=g.LR)
    losses = []
    with LoRAInjector(model, targets, gen.scale) as inj:
        for i in range(3):
            inj.set(gen(e.cuda()))
            loss = _forward(model, _encode(proc, train[1 + i], card["prompt"]).to(model.device)).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(loss.item())
        with torch.no_grad():
            w = gen(e.cuda())
        inj.set(w)
        text = _generate(model, proc, [_user(ev[0], card["prompt"])])
    p = OUT / "smoke" / "adapter"
    save_peft_adapter(w, targets, p, rank=g.RANK, alpha=g.ALPHA, base_model=g.MODEL)
    w2, _ = load_peft_adapter(p, targets)
    vol.commit()
    return {"targets": len(targets), "shapes": shapes, "hidden": list(h.shape), "emb": list(e.shape),
            "generator_params": n_params, "losses": losses, "sample_output": text[:300], "gold": ev[0]["label"],
            "peft_roundtrip_ok": all(torch.allclose(w[n][1].cpu().half().float(), w2[n][1].float()) for n in w),
            "tasks": {"train": len(sp.train), "dev": len(sp.dev), "test": len(sp.test)}}


# --------------------------------------------------------------------------- step 1: embeddings

@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=6 * 3600)
def embed_job(tasks: list[str]) -> dict:
    import torch

    from minicpm_t2l import gate1 as g
    from t2l import example_embedding

    model, proc = _load()
    done, t0 = {}, time.time()
    for task in tasks:
        f = OUT / "emb" / f"{task}.pt"
        if f.exists():
            done[task] = "cached"
            continue
        card, train, _ = g.read_task(POOL, task)
        embs = []
        with torch.no_grad():
            for row in train:
                b = _encode(proc, row, card["prompt"]).to(model.device)
                embs.append(example_embedding(_final_hidden(model, b), b["attention_mask"], (b["labels"] != -100).long()).half().cpu())
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": [r["id"] for r in train], "emb": torch.stack(embs)}, f)
        vol.commit()
        done[task] = len(train)
    return {"done": done, "seconds": time.time() - t0}


# --------------------------------------------------------------------------- steps 2-3: hypernetwork training

@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=20 * 3600)
def train_hyper(run: str, lr: float, steps: int, variant: str = "both", max_hours: float = 14.0) -> dict:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    from minicpm_t2l import gate1 as g
    from t2l import LoRAInjector, task_embedding

    rdir = OUT / "runs" / run
    if (rdir / "final.pt").exists():
        return json.loads((rdir / "train_log.json").read_text(encoding="utf-8")) | {"reused": True}
    sp = _splits()
    model, proc = _load()
    targets = _targets(model)
    gen = _new_generator(targets, variant)
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=g.WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(opt, g.WARMUP, steps)
    data = {}
    for t in sp.train:
        card, train, _ = g.read_task(POOL, t)
        ids, emb = _load_emb(t)
        assert ids == [r["id"] for r in train], t
        data[t] = (card["prompt"], train, emb)
    plan = g.training_samples({t: len(v[1]) for t, v in data.items()}, steps, seed=0)
    rdir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    with LoRAInjector(model, targets, gen.scale) as inj:
        for step, batch in enumerate(plan, 1):
            tot = 0.0
            for s in batch:
                prompt, rows, emb = data[s.task]
                e = _variant(task_embedding(emb[list(s.support)].float()), variant).cuda()
                inj.set(gen(e))
                loss = _forward(model, _encode(proc, rows[s.query], prompt).to(model.device)).loss / len(batch)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            log.append(tot)
            if step % 50 == 0:
                print(f"{run} step {step}/{steps} loss(avg50) {sum(log[-50:]) / 50:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % 500 == 0:
                torch.save({"state": gen.state_dict(), "step": step, "config": gen.config(), "variant": variant}, rdir / "ckpt.pt")
                (rdir / "losses.json").write_text(json.dumps(log), encoding="utf-8")
                vol.commit()
            if time.time() - t0 > max_hours * 3600:
                raise RuntimeError(f"{run}: exceeded {max_hours} h at step {step}")
    torch.save({"state": gen.state_dict(), "step": steps, "config": gen.config(), "variant": variant}, rdir / "final.pt")
    info = {"run": run, "lr": lr, "steps": steps, "variant": variant, "train_tasks": len(sp.train),
            "query_examples": steps * g.TASKS_PER_STEP, "seconds": time.time() - t0,
            "loss_first100": sum(log[:100]) / max(1, len(log[:100])), "loss_last100": sum(log[-100:]) / max(1, len(log[-100:]))}
    (rdir / "losses.json").write_text(json.dumps(log), encoding="utf-8")
    (rdir / "train_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    vol.commit()
    return info


def _load_generator(run: str, targets):
    import torch

    ck = torch.load(OUT / "runs" / run / "final.pt", map_location="cpu")
    gen = _new_generator(targets, ck["variant"])
    gen.load_state_dict(ck["state"])
    gen.eval()
    return gen, ck["variant"]


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=6 * 3600)
def dev_eval(run: str) -> dict:
    """k=4 emitted adapter (or `run == "base"`: no adapter) on the first DEV_EVAL_DOCS eval docs of each dev task."""
    import torch

    from minicpm_t2l import gate1 as g
    from t2l import LoRAInjector

    vol.reload()
    sp = _splits()
    model, proc = _load()
    targets = _targets(model)
    res = {}
    gen = variant = None
    if run != "base":
        gen, variant = _load_generator(run, targets)
    with LoRAInjector(model, targets, g.ALPHA / g.RANK) as inj:
        for task in sp.dev:
            card, train, ev = g.read_task(POOL, task)
            if gen is not None:
                with torch.no_grad():
                    inj.set(gen(_task_emb(task, g.support_for([r["id"] for r in train], 4), variant).cuda()))
            res[task] = _eval_rows(model, proc, card["prompt"], ev[: g.DEV_EVAL_DOCS], [], OUT / "dev" / run / task,
                                   {"task": task, "arm": "emit4" if gen is not None else "base", "run": run})["f1"]
    mean = sum(res.values()) / len(res)
    (OUT / "dev" / run / "summary.json").write_text(json.dumps({"mean_f1": mean, "per_task": res}, indent=2), encoding="utf-8")
    vol.commit()
    return {"run": run, "mean_f1": mean}


# --------------------------------------------------------------------------- step 4: multi-task LoRA (comparison 3)

def _peft_train(model, proc, examples: list[tuple[str, dict]], targets, epochs: int, out_dir: Path, seed: int = 0) -> dict:
    """Gate 0 LoRA settings on a list of (prompt, row) examples; saves a PEFT adapter."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    from minicpm_t2l import gate1 as g

    c = g.LORA_MT
    for p in model.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(model, LoraConfig(r=c["r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                                          target_modules=[t.name for t in targets]))
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.0)
    steps = epochs * len(examples) // c["grad_accum"]
    sched = get_cosine_schedule_with_warmup(opt, int(c["warmup_ratio"] * steps), max(steps, 1))
    pm.train()
    rng, losses, i, t0 = random.Random(seed), [], 0, time.time()
    for _ in range(epochs):
        order = examples[:]
        rng.shuffle(order)
        for prompt, row in order:
            loss = _forward(pm, _encode(proc, row, prompt).to(pm.device)).loss / c["grad_accum"]
            loss.backward()
            losses.append(loss.item() * c["grad_accum"])
            i += 1
            if i % c["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if i % 2000 == 0:
                print(f"peft step {i}/{epochs * len(examples)} loss(avg) {sum(losses[-2000:]) / 2000:.4f}", flush=True)
    pm.save_pretrained(out_dir)
    info = {"examples": len(examples), "epochs": epochs, "optimizer_steps": steps, "seconds": time.time() - t0,
            "loss_last100": sum(losses[-100:]) / max(1, len(losses[-100:]))}
    (out_dir / "train_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    pm.unload()
    return info


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=20 * 3600)
def train_mt() -> dict:
    """One LoRA on the same 48,000 query examples the hypernetwork sees (same sampler, seed 0), one pass."""
    from minicpm_t2l import gate1 as g

    out = OUT / "mt" / "adapter"
    if (out / "adapter_model.safetensors").exists():
        return json.loads((out / "train_log.json").read_text(encoding="utf-8")) | {"reused": True}
    sp = _splits()
    model, proc = _load()
    targets = _targets(model)
    data = {t: g.read_task(POOL, t) for t in sp.train}
    plan = g.training_samples({t: len(v[1]) for t, v in data.items()}, g.STEPS, seed=0)
    examples = [(data[s.task][0]["prompt"], data[s.task][1][s.query]) for batch in plan for s in batch]
    out.mkdir(parents=True, exist_ok=True)
    info = _peft_train(model, proc, examples, targets, epochs=1, out_dir=out)
    vol.commit()
    return info


# --------------------------------------------------------------------------- steps 5-6: held-out evaluation

@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=6 * 3600)
def oracle_job(task: str) -> dict:
    from minicpm_t2l import gate1 as g
    from t2l import LoRAInjector, load_peft_adapter

    vol.reload()
    out = OUT / "eval" / "oracle" / task
    if (out / "scores.json").exists():
        return json.loads((out / "scores.json").read_text(encoding="utf-8"))
    model, proc = _load()
    targets = _targets(model)
    card, train, ev = g.read_task(POOL, task)
    ad = OUT / "oracle" / task / "adapter"
    if not (ad / "adapter_model.safetensors").exists():
        ad.mkdir(parents=True, exist_ok=True)
        _peft_train(model, proc, [(card["prompt"], r) for r in train], targets, epochs=g.ORACLE_EPOCHS, out_dir=ad)
        model.eval()
        vol.commit()
    w, cfg = load_peft_adapter(ad, targets, device="cuda")
    with LoRAInjector(model, targets, cfg["lora_alpha"] / cfg["r"]) as inj:
        inj.set(w)
        return _eval_rows(model, proc, card["prompt"], ev, [], out, {"task": task, "arm": "oracle"})


@app.function(gpu=GPU, volumes={str(VOL): vol}, secrets=secrets, timeout=8 * 3600)
def eval_job(task: str, run: str, arms: tuple = ("base", "icl1", "icl4", "mt", "emit1", "emit4", "emit8")) -> dict:
    """Non-oracle arms for one held-out task (default: all of base, icl1, icl4, mt, emit1, emit4, emit8)."""
    import torch

    from minicpm_t2l import gate1 as g
    from t2l import LoRAInjector, load_peft_adapter

    vol.reload()
    model, proc = _load()
    targets = _targets(model)
    card, train, ev = g.read_task(POOL, task)
    by_id = {r["id"]: r for r in train}
    sup = g.support_draw([r["id"] for r in train])
    res = {}
    with LoRAInjector(model, targets, g.ALPHA / g.RANK) as inj:
        if "base" in arms:
            res["base"] = _eval_rows(model, proc, card["prompt"], ev, [], OUT / "eval" / "base" / task, {"task": task, "arm": "base"})
        for k in [k for k in (1, 4) if f"icl{k}" in arms]:
            res[f"icl{k}"] = _eval_rows(model, proc, card["prompt"], ev, [by_id[i] for i in sup[:k]],
                                        OUT / "eval" / f"icl{k}" / task, {"task": task, "arm": f"icl{k}"})
        if "mt" in arms:
            w, cfg = load_peft_adapter(OUT / "mt" / "adapter", targets, device="cuda")
            inj.scale = cfg["lora_alpha"] / cfg["r"]
            inj.set(w)
            res["mt"] = _eval_rows(model, proc, card["prompt"], ev, [], OUT / "eval" / "mt" / task, {"task": task, "arm": "mt"})
        gen, variant = _load_generator(run, targets)
        inj.scale = gen.scale
        for k in [k for k in g.KS_EVAL if f"emit{k}" in arms]:
            with torch.no_grad():
                inj.set(gen(_task_emb(task, sup[:k], variant).cuda()))
            res[f"emit{k}"] = _eval_rows(model, proc, card["prompt"], ev, [], OUT / "eval" / f"emit{k}" / task,
                                         {"task": task, "arm": f"emit{k}", "run": run, "support": sup[:k]})
    return {a: r["f1"] for a, r in res.items()} | {"task": task}


@app.function(volumes={str(VOL): vol}, timeout=1800, cpu=2, memory=8192)
def per_doc(paths: list[str]) -> dict:
    vol.reload()
    res = {}
    for p in paths:
        d = VOL / p
        if not (d / "scores.json").exists():
            res[p] = None
            continue
        rows = [json.loads(l) for l in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        res[p] = {"scores": json.loads((d / "scores.json").read_text(encoding="utf-8")),
                  "docs": [{k: r[k] for k in ("id", "tp", "n_pred", "n_gold", "parsed")} for r in rows]}
    return res


@app.function(volumes={str(VOL): vol}, timeout=600, cpu=1, memory=2048)
def read_json(paths: list[str]) -> dict:
    vol.reload()
    return {p: (json.loads((VOL / p).read_text(encoding="utf-8")) if (VOL / p).exists() else None) for p in paths}


# --------------------------------------------------------------------------- local entrypoints

@app.local_entrypoint()
def smoke():
    print(json.dumps(_guarded("hypernetwork", "gate1-smoke", 1.0, lambda: smoke_job.remote()), indent=1, ensure_ascii=False))


@app.local_entrypoint()
def embed(chunks: int = 10, usd: float = 6):
    from minicpm_t2l import gate1 as g

    sp = g.make_splits(json.loads(Path("studies/task_pool/manifest.json").read_text(encoding="utf-8")),
                       json.loads(Path("studies/task_pool/heldout_2026-11-19.json").read_text(encoding="utf-8")))
    tasks = list(sp.train + sp.dev + sp.test)
    parts = [tasks[i::chunks] for i in range(chunks)]

    def go():
        out = []
        for r in embed_job.map(parts, return_exceptions=True):
            print(("FAILED " + repr(r)) if isinstance(r, Exception) else f"chunk done: {len(r['done'])} tasks in {r['seconds']:.0f}s", flush=True)
            out.append(r)
        return out

    res = _guarded("hypernetwork", "gate1-embed", usd, go)
    print(f"failed chunks: {sum(isinstance(r, Exception) for r in res)}")


DEV_RUNS = {"lr3e-5": (3e-5, "both"), "lr1e-4": (1e-4, "both"), "lr3e-4": (3e-4, "both"), "lr1e-4-answer": (1e-4, "answer")}


@app.local_entrypoint()
def dev(runs: str = ",".join(DEV_RUNS), usd: float = 20):
    from minicpm_t2l import gate1 as g

    names = runs.split(",")

    def one(name):
        if name == "base":
            return dev_eval.remote("base")
        lr, variant = DEV_RUNS[name]
        info = train_hyper.remote(f"dev_{name}", lr, g.DEV_STEPS, variant, 6.0)
        print(f"trained {name}: {json.dumps(info)[:300]}", flush=True)
        return dev_eval.remote(f"dev_{name}")

    def go():
        out = {}
        for n, r in zip(names + ["base"], _run_parallel(one, names + ["base"])):
            print(f"{n}: {r}", flush=True)
            out[n] = r
        return out

    res = _guarded("hypernetwork", "gate1-dev", usd, go)
    Path("studies/gate1/logs").mkdir(parents=True, exist_ok=True)
    Path("studies/gate1/dev_results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


def _run_parallel(fn, items):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futs = [ex.submit(fn, i) for i in items]
        for f in futs:
            try:
                yield f.result()
            except Exception as e:  # report and keep going
                yield {"error": repr(e)}


@app.local_entrypoint()
def final(lr: float = 1e-4, variant: str = "both", usd: float = 45):
    from minicpm_t2l import gate1 as g

    def go():
        h = train_hyper.spawn("final", lr, g.STEPS, variant, 16.0)
        m = train_mt.spawn()
        out = {}
        for name, c in (("hyper", h), ("mt", m)):
            try:
                out[name] = c.get()
            except Exception as e:
                out[name] = {"error": repr(e)}
            print(f"{name}: {json.dumps(out[name], default=str)[:400]}", flush=True)
        return out

    _guarded("hypernetwork", "gate1-final-train", usd, go)


@app.local_entrypoint()
def evaluate(run: str = "final", usd: float = 38, arms: str = "base,icl1,icl4,mt,emit1,emit4,emit8", oracle: bool = True):
    from minicpm_t2l import gate1 as g

    sp = g.make_splits(json.loads(Path("studies/task_pool/manifest.json").read_text(encoding="utf-8")),
                       json.loads(Path("studies/task_pool/heldout_2026-11-19.json").read_text(encoding="utf-8")))
    tasks = list(sp.test)

    def go():
        arm_t = tuple(arms.split(","))
        ev = eval_job.starmap([(t, run, arm_t) for t in tasks], return_exceptions=True)
        orc = oracle_job.map(tasks, return_exceptions=True) if oracle else []
        out = {"eval": [], "oracle": []}
        for r in ev:
            print(("FAILED " + repr(r)) if isinstance(r, Exception) else json.dumps(r), flush=True)
            out["eval"].append(r if not isinstance(r, Exception) else {"error": repr(r)})
        for r in orc:
            print(("FAILED oracle " + repr(r)) if isinstance(r, Exception) else f"oracle {r['task']} f1={r['f1']:.3f}", flush=True)
            out["oracle"].append(r if not isinstance(r, Exception) else {"error": repr(r)})
        return out

    _guarded("evaluation", "gate1-evaluate", usd, go)


@app.local_entrypoint()
def pull():
    from minicpm_t2l import gate1 as g

    sp = g.make_splits(json.loads(Path("studies/task_pool/manifest.json").read_text(encoding="utf-8")),
                       json.loads(Path("studies/task_pool/heldout_2026-11-19.json").read_text(encoding="utf-8")))
    arms = ("base", "icl1", "icl4", "mt", "oracle", "emit1", "emit4", "emit8")
    res = per_doc.remote([f"gate1/eval/{a}/{t}" for a in arms for t in sp.test])
    Path("studies/gate1").mkdir(parents=True, exist_ok=True)
    Path("studies/gate1/per_doc.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"pulled {sum(v is not None for v in res.values())} / {len(res)}")
