"""Gate 1 recipe: task splits, support-page draws and training samples.

Frozen in studies/gate1/protocol.md. Pure Python; the model-side code is in modal_app/gate1.py
and the model-agnostic hypernetwork in the `t2l` package (openT2L).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEV_SEED = 20261121
DEV_DOMAINS = ("gym_membership", "utility_bill")     # random.Random(DEV_SEED).sample(training domains, 2)
SUPPORT_SEED = 7
KS_EVAL = (1, 4, 8)
KS_TRAIN = (1, 2, 4, 8)
TASKS_PER_STEP = 8
STEPS = 6000
DEV_STEPS = 2000
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP = 300
DEV_EVAL_DOCS = 20
MODEL = "openbmb/MiniCPM-V-4.7-1B"
PROC = dict(downsample_mode="16x", max_slice_nums=9, enable_thinking=False)
GEN = dict(max_new_tokens=768, do_sample=False, repetition_penalty=1.0)
TARGET_PATTERNS = ("self_attn.q_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj")
TARGET_EXCLUDE = ("vision", "vpm", "visual", "resampler", "merger", "siglip", "mtp")
N_TARGETS = 48
RANK, ALPHA = 8, 16
LORA_MT = dict(r=8, lora_alpha=16, lora_dropout=0.05, lr=2e-4, grad_accum=8, warmup_ratio=0.05)  # Gate 0 settings
ORACLE_EPOCHS = 3


@dataclass(frozen=True)
class Splits:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]


def is_dev(task_id: str, card: dict | None = None) -> bool:
    if card is not None and card.get("source") == "synth":
        return card["template"]["domain"] in DEV_DOMAINS
    return any(task_id.startswith(f"syn_{d}_") for d in DEV_DOMAINS)


def make_splits(manifest: dict, heldout: dict) -> Splits:
    """Training = pool training tasks minus the dev domains; test = the frozen held-out list."""
    test = tuple(sorted(heldout["heldout_tasks"]))
    kept = manifest["tasks"]
    if not set(test) <= set(kept):
        raise ValueError("held-out list names tasks the manifest does not have")
    pool_train = sorted(k for k, v in kept.items() if not v["heldout"])
    if set(pool_train) & set(test):
        raise ValueError("a task is both training and held out")
    dev = tuple(t for t in pool_train if is_dev(t))
    train = tuple(t for t in pool_train if not is_dev(t))
    return Splits(train, dev, test)


def support_draw(train_ids: list[str], k_max: int = max(KS_EVAL), seed: int = SUPPORT_SEED) -> list[str]:
    """One ordered draw; the support set for k is the first k ids (so k=1 ⊂ k=4 ⊂ k=8)."""
    if len(train_ids) < k_max:
        raise ValueError(f"need {k_max} train docs, have {len(train_ids)}")
    return random.Random(seed).sample(list(train_ids), k_max)


def support_for(train_ids: list[str], k: int, seed: int = SUPPORT_SEED) -> list[str]:
    return support_draw(train_ids, seed=seed)[:k]


@dataclass(frozen=True)
class Sample:
    task: str
    k: int
    support: tuple[int, ...]   # indices into the task's train list
    query: int


def training_samples(task_sizes: dict[str, int], n_steps: int, seed: int = 0,
                     tasks_per_step: int = TASKS_PER_STEP) -> list[list[Sample]]:
    """Per step: `tasks_per_step` tasks (uniform), each with k ~ KS_TRAIN, a support set and a disjoint query."""
    rng = random.Random(seed)
    names = sorted(task_sizes)
    steps = []
    for _ in range(n_steps):
        batch = []
        for t in rng.sample(names, tasks_per_step) if len(names) >= tasks_per_step else rng.choices(names, k=tasks_per_step):
            n = task_sizes[t]
            k = min(rng.choice(KS_TRAIN), n - 1)
            idx = rng.sample(range(n), k + 1)
            batch.append(Sample(t, k, tuple(idx[:k]), idx[k]))
        steps.append(batch)
    return steps


def read_task(root: Path, task_id: str) -> tuple[dict, list[dict], list[dict]]:
    """(task.json, train rows, eval rows) of a pool task; image paths are relative to `root`."""
    d = Path(root) / "tasks" / task_id
    rows = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads((d / "task.json").read_text(encoding="utf-8")), rows(d / "train.jsonl"), rows(d / "eval.jsonl")
