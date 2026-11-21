"""Run the Hugging Face model card's usage code, as published, on held-out Gate 1 tasks, and
compare against the Gate 1 emit4 outputs for the same pages.

    modal run modal_app/verify_hf_card.py            # reserves USD 2 (stage "evaluation")

The image installs openT2L from GitHub (`import t2l` = the published library, not this repo's
package), downloads the hypernetwork and README from the Hub, and executes the README's Python
block: everything up to the example prompt, then the emit + PEFT lines, with real support pages
in place of the card's placeholder list. Scoring happens locally with minicpm_t2l.scoring.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import modal

HF_REPO = "ASHu2/openT2L-MiniCPM-V-4.7-1B-docs"
LIB = "git+https://github.com/ASH1998/openT2L"
TASKS = ("syn_purchase_order_zh_00", "fintabnet_full", "inv_items")
N_DOCS = 10
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")

image = (modal.Image.from_name(os.environ.get("T2L_GPU_IMAGE", "minicpm-t2l-gpu"))
         .pip_install(f"opent2l[hf] @ {LIB}"))
app = modal.App(os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-verify-hf", image=image)
vol = modal.Volume.from_name(VOLUME)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


def _card_code() -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    readme = Path(hf_hub_download(HF_REPO, "README.md")).read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    head, rest = code.split("prompt = (", 1)
    tail = rest[rest.index("emit_adapter("):]
    tail = tail.replace('"my_adapter/"', "ADAPTER_DIR")
    return head, tail


@app.function(gpu="A10G", volumes={str(VOL): vol}, secrets=secrets, timeout=3600)
def run() -> dict:
    import t2l

    head, tail = _card_code()
    out, t0 = {"t2l_file": t2l.__file__, "card_head_lines": head.count("\n"), "tasks": {}}, time.time()
    ns: dict = {}
    exec(head, ns)  # imports, model + processor, load_hypernet from the Hub, embed()
    model0 = ns["model"]
    for task in TASKS:
        d = VOL / "task_pool" / "tasks" / task
        card = json.loads((d / "task.json").read_text(encoding="utf-8"))
        rows = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        train, ev = rows(d / "train.jsonl"), rows(d / "eval.jsonl")
        ids = random.Random(7).sample([r["id"] for r in train], 8)[:4]  # Gate 1 support draw (seed 7, nested, k=4)
        by_id = {r["id"]: r for r in train}
        examples = [(str(VOL / "task_pool" / by_id[i]["image"]), by_id[i]["label"]) for i in ids]
        ns.update(model=model0, prompt=card["prompt"], examples=examples, ADAPTER_DIR=f"/tmp/adapter_{task}")
        ns["support"] = ns["torch"].stack([ns["embed"](img, card["prompt"], ans) for img, ans in examples])
        exec(tail, ns)  # emit_adapter(...) then model = PeftModel.from_pretrained(...)
        model, proc, torch = ns["model"], ns["proc"], ns["torch"]
        preds = []
        for r in ev[:N_DOCS]:
            msgs = [{"role": "user", "content": [{"type": "image", "path": str(VOL / "task_pool" / r["image"])},
                                                  {"type": "text", "text": card["prompt"]}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", **ns["PROC"]).to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, downsample_mode=ns["PROC"]["downsample_mode"], max_new_tokens=768,
                                   do_sample=False, repetition_penalty=1.0)
            preds.append({"id": r["id"], "output": proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0],
                          "label": r["label"]})
        stored = [json.loads(l) for l in (VOL / "gate1" / "eval" / "emit4" / task / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        base = [json.loads(l) for l in (VOL / "gate1" / "eval" / "base" / task / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        sid = {s["id"]: s["output"] for s in stored}
        bid = {s["id"]: s["output"] for s in base}
        for p in preds:
            p["gate1_emit4"], p["gate1_base"] = sid.get(p["id"]), bid.get(p["id"])
        out["tasks"][task] = {"support": ids, "adapter_files": sorted(os.listdir(f"/tmp/adapter_{task}")), "preds": preds}
        ns["model"] = model.unload() if hasattr(model, "unload") else model0  # drop this adapter before the next task
    out["seconds"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(usd: float = 2.0):
    from minicpm_t2l.budget import Budget
    from minicpm_t2l.scoring import TaskScore, score_doc

    b = Budget()
    rid = b.reserve("evaluation", usd, run="verify-hf-card", note=f"{HF_REPO} card code on {len(TASKS)} held-out tasks x {N_DOCS} docs")
    t0 = time.time()
    try:
        res = run.remote()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    summary = {"hf_repo": HF_REPO, "library": LIB, "t2l_file": res["t2l_file"], "seconds": res["seconds"], "tasks": {}}
    for task, r in res["tasks"].items():
        sc = {k: TaskScore() for k in ("card", "gate1_emit4", "gate1_base")}
        same = 0
        for p in r["preds"]:
            sc["card"].add(score_doc(p["output"], p["label"]))
            sc["gate1_emit4"].add(score_doc(p["gate1_emit4"] or "", p["label"]))
            sc["gate1_base"].add(score_doc(p["gate1_base"] or "", p["label"]))
            same += p["output"].strip() == (p["gate1_emit4"] or "").strip()
        summary["tasks"][task] = {"support": r["support"], "adapter_files": r["adapter_files"], "n_docs": len(r["preds"]),
                                  **{f"f1_{k}": v.summary()["f1"] for k, v in sc.items()}, "identical_to_gate1_emit4": same}
    Path("studies/gate1/hf_card_check.json").write_text(json.dumps({"summary": summary, "raw": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
