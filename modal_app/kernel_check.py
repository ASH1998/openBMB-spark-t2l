"""Does installing causal-conv1d speed up MiniCPM-V-4.7 and leave outputs unchanged?

    modal run modal_app/kernel_check.py

Builds an image with causal-conv1d compiled against torch 2.10 (CUDA devel base), then on the
same A10G: times the example-embedding forward on 3 pool pages and one training step with the
kernel, compares the embeddings against the cached ones (made without the kernel), and reports.
Writes nothing to the volume.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "ninja", "packaging", "wheel")
    .pip_install("flash-linear-attention", "transformers>=5.21", "peft>=0.19", "accelerate", "pillow",
                 "huggingface_hub>=1.10", "safetensors")
    .run_commands("TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 pip install --no-build-isolation causal-conv1d", gpu="A10G")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("minicpm_t2l", "t2l")
)
app = modal.App(os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-kernel-check", image=image)
vol = modal.Volume.from_name(VOLUME)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


@app.function(gpu="A10G", volumes={str(VOL): vol}, secrets=secrets, timeout=3600)
def check(task: str) -> dict:
    import causal_conv1d  # noqa: F401  (fails loudly if the build did not work)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from minicpm_t2l import gate1 as g
    from t2l import example_embedding

    proc = AutoProcessor.from_pretrained(g.MODEL)
    model = AutoModelForImageTextToText.from_pretrained(g.MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    root = VOL / "task_pool"
    card, train, _ = g.read_task(root, task)
    cached = torch.load(VOL / "gate1" / "emb" / f"{task}.pt", map_location="cpu")

    def enc(row):
        user = [{"role": "user", "content": [{"type": "image", "path": str(root / row["image"])}, {"type": "text", "text": card["prompt"]}]}]
        p = proc.apply_chat_template(user, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **g.PROC)
        f = proc.apply_chat_template(user + [{"role": "assistant", "content": [{"type": "text", "text": json.dumps(row["label"], ensure_ascii=False)}]}],
                                     tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt", **g.PROC)
        lab = f["input_ids"].clone()
        lab[:, : p["input_ids"].shape[1]] = -100
        f["labels"] = lab
        return f.to("cuda")

    times, diffs, enc_times = [], [], []
    with torch.no_grad():
        for row in train[:4]:
            t1 = time.time()
            b = enc(row)
            enc_times.append(time.time() - t1)
            torch.cuda.synchronize(); t0 = time.time()
            out = model(**{k: v for k, v in b.items() if k != "labels"}, downsample_mode="16x", output_hidden_states=True)
            torch.cuda.synchronize(); times.append(time.time() - t0)
            e = example_embedding(out.hidden_states[-1], b["attention_mask"], (b["labels"] != -100).long()).half().cpu()
            j = cached["ids"].index(row["id"])
            diffs.append(float(torch.nn.functional.cosine_similarity(e.float(), cached["emb"][j].float(), dim=0)))
    b = enc(train[5])
    for p in model.parameters():
        p.requires_grad_(False)
    torch.cuda.synchronize(); t0 = time.time()
    model(**b, downsample_mode="16x").loss  # forward with labels
    torch.cuda.synchronize()
    return {"preprocess_seconds": enc_times, "forward_seconds": times, "cosine_vs_cached_no_kernel": diffs, "train_forward_seconds": time.time() - t0}


@app.local_entrypoint()
def main(task: str = ""):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("hypernetwork", 1.5, run="gate1-kernel-check")
    t0 = time.time()
    try:
        res = check.remote(task)
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    print(json.dumps(res, indent=1))
