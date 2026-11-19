"""Gate 0 llama.cpp check (protocol amendment 2).

Does a PEFT LoRA trained in PyTorch apply correctly in llama.cpp to MiniCPM-OCR-1.0?
Converts the HF checkpoint (16x) and one adapter to GGUF with a pinned llama.cpp,
runs base and base+LoRA on 20 eval docs with llama-mtmd-cli, and compares with the
PyTorch predictions already on the volume.

    modal run modal_app/llamacpp_check.py            # budget-guarded
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath

import modal

LLAMA_COMMIT = "51a77c4f247ce96fb932e29c9f81625ca9d3b0b7"
MODEL_ID = "openbmb/MiniCPM-OCR-1.0"
MODEL_KEY = "ocr"
ARM = "loraB"
TASK = "cord_total"
N_DOCS = 20
RUNS = "runs_a10g"
VOLUME = os.environ.get("T2L_MODAL_VOLUME_NAME", "minicpm-t2l-artifacts")
VOL = Path("/vol")
OUT = VOL / "gate0/llamacpp" / f"{MODEL_KEY}_{ARM}_{TASK}"
LLAMA = PurePosixPath("/opt/llama.cpp")  # POSIX: these strings run in the Linux container (built on Windows)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .run_commands(
        f"git clone https://github.com/ggml-org/llama.cpp {LLAMA}",
        f"cd {LLAMA} && git checkout {LLAMA_COMMIT}",
        # portable CPU build (the image builder's CPU may differ from the runner's)
        f"cd {LLAMA} && cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release"
        " && cmake --build build -j 16 --target llama-mtmd-cli",
    )
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(f"pip install -r {LLAMA}/requirements/requirements-convert_hf_to_gguf.txt "
                  f"&& pip install -e {LLAMA}/gguf-py")
    # the OCR tokenizer config needs transformers 5.x (TokenizersBackend); llama.cpp's requirements pin 4.57
    .pip_install("transformers>=5.21", "huggingface_hub>=1.10", "pillow", "python-dotenv")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("minicpm_t2l", "t2l")
)
app = modal.App("minicpm-t2l-gate0-llamacpp", image=image)
vol = modal.Volume.from_name(VOLUME)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


def _sh(cmd: list[str], log: Path, timeout: int = 3600) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr[-20000:]}\n")
    if r.returncode != 0:
        vol.commit()
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[:3]}\nSTDERR TAIL:\n{r.stderr[-3000:]}\nSTDOUT TAIL:\n{r.stdout[-1000:]}")
    return r.stdout


def _hf_dir_16x(work: Path) -> Path:
    """Local copy of the HF checkpoint with downsample_mode forced to 16x (weights symlinked)."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(MODEL_ID))
    dst = work / "hf16x"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            if f.suffix == ".json":
                shutil.copy(f, dst / f.name)
            elif not (dst / f.name).exists():
                os.symlink(f.resolve(), dst / f.name)
    for name in ("config.json", "preprocessor_config.json"):
        c = json.loads((dst / name).read_text())
        if "downsample_mode" in c:
            c["downsample_mode"] = "16x"
        (dst / name).write_text(json.dumps(c, indent=2))
    return dst


def _gen(model: Path, mmproj: Path, img: Path, prompt: str, log: Path, lora: Path | None) -> str:
    cmd = [str(LLAMA / "build/bin/llama-mtmd-cli"), "-m", str(model), "--mmproj", str(mmproj),
           "--image", str(img), "-p", prompt, "--temp", "0", "-n", "768", "--repeat-penalty", "1.0",
           # the model's chat template defaults enable_thinking=false (same prompt as PyTorch);
           # mtmd-cli has no --chat-template-kwargs
           "--jinja", "-t", "16",
           "--seed", "0", "--no-warmup"]
    if lora is not None:
        cmd += ["--lora", str(lora)]
    return _sh(cmd, log, timeout=1800).strip()


@app.function(volumes={VOL: vol}, secrets=secrets, cpu=16, memory=32768, timeout=4 * 3600)
def check() -> dict:
    from minicpm_t2l.scoring import TaskScore, score_doc

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "commands.log"
    log.write_text(f"llama.cpp {LLAMA_COMMIT}\n")
    work = Path("/tmp/work")
    t0 = time.time()

    # 1. convert base (16x) and adapter
    _sh(["bash", "-c", f"ls -la {LLAMA}/convert_hf_to_gguf.py {LLAMA}/build/bin/llama-mtmd-cli && which python && python --version"], log)
    hf = _hf_dir_16x(work)
    lm, mm, lora = work / "lm-f16.gguf", work / "mmproj-f16-16x.gguf", work / "lora-f16.gguf"
    conv = str(LLAMA / "convert_hf_to_gguf.py")
    # Unfused layout: by default the converter fuses in_proj_qkv+in_proj_z into attn_qkv_gate,
    # but a LoRA on in_proj_qkv converts to blk.N.attn_qkv and llama.cpp refuses LoRA tensors
    # with no base match. qwen35.cpp also accepts separate attn_qkv / attn_gate tensors, so we
    # switch the merge off. Same weights, split into two tensors (amendment 3).
    unfused = (f"import sys, runpy; sys.path.insert(0, '{LLAMA}'); import conversion.qwen as q; "
               f"q.Qwen3_5TextModel._SHARED_INPUT_MERGES = (); "
               f"sys.argv = ['convert_hf_to_gguf.py', '{hf}', '--outtype', 'f16', '--outfile', '{lm}']; "
               f"runpy.run_path('{conv}', run_name='__main__')")
    _sh(["python", "-c", unfused], log)
    _sh(["python", "-c", f"import gguf; r = gguf.GGUFReader('{lm}'); n = [t.name for t in r.tensors]; "
         "print('qkv_gate', sum('attn_qkv_gate' in x for x in n), 'attn_qkv', sum(x.endswith('attn_qkv.weight') for x in n), "
         "'attn_gate', sum(x.endswith('attn_gate.weight') for x in n))"], log)
    _sh(["python", conv, str(hf), "--outtype", "f16", "--mmproj", "--outfile", str(mm)], log)
    adapter = VOL / "gate0" / RUNS / MODEL_KEY / ARM / TASK / "adapter"
    _sh(["python", str(LLAMA / "convert_lora_to_gguf.py"), str(adapter), "--base", str(hf),
         "--outtype", "f16", "--outfile", str(lora)], log)
    for p in (lm, mm, lora):
        shutil.copy(p, OUT / p.name)  # keep the artefacts for the on-device agent
    vol.commit()

    # 2. generate on the same docs, base and +LoRA
    tdir = VOL / "gate0/tasks" / TASK
    prompt = json.loads((tdir / "task.json").read_text(encoding="utf-8"))["prompt"]
    docs = [json.loads(l) for l in (tdir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:N_DOCS]
    outs = {"llama_base": [], "llama_lora": []}
    for d in docs:
        img = tdir / d["image"]
        outs["llama_base"].append(_gen(lm, mm, img, prompt, log, None))
        outs["llama_lora"].append(_gen(lm, mm, img, prompt, log, lora))

    # 3. compare with PyTorch predictions on the same docs
    def torch_preds(arm):
        rows = {json.loads(l)["id"]: json.loads(l)["output"] for l in
                (VOL / "gate0" / RUNS / MODEL_KEY / arm / TASK / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        return [rows[d["id"]] for d in docs]

    outs["torch_base"], outs["torch_lora"] = torch_preds("base"), torch_preds(ARM)
    summary = {}
    for name, texts in outs.items():
        ts = TaskScore()
        for t, d in zip(texts, docs):
            ts.add(score_doc(t, d["label"]))
        summary[name] = ts.summary()
    f1 = {k: v["f1"] for k, v in summary.items()}
    torch_gain = f1["torch_lora"] - f1["torch_base"]
    llama_gain = f1["llama_lora"] - f1["llama_base"]
    agree = sum(a.strip() == b.strip() for a, b in zip(outs["llama_lora"], outs["torch_lora"])) / len(docs)
    res = {
        "commit": LLAMA_COMMIT, "task": TASK, "arm": ARM, "model": MODEL_ID, "n_docs": len(docs),
        "f1": f1, "torch_gain": torch_gain, "llama_gain": llama_gain,
        "exact_agreement_lora": agree,
        "pass_f1": f1["llama_lora"] >= f1["torch_lora"] - 0.05,
        "pass_gain": llama_gain >= 0.5 * torch_gain,
        "seconds": time.time() - t0,
    }
    res["pass"] = res["pass_f1"] and res["pass_gain"]
    (OUT / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with open(OUT / "outputs.jsonl", "w", encoding="utf-8") as fh:
        for i, d in enumerate(docs):
            fh.write(json.dumps({"id": d["id"], **{k: v[i] for k, v in outs.items()}}, ensure_ascii=False) + "\n")
    vol.commit()
    return res


@app.local_entrypoint()
def main(usd: float = 2.0):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("gate0", usd, run="gate0-llamacpp-check")
    t0 = time.time()
    try:
        res = check.remote()
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    print(json.dumps(res, indent=2))


@app.function(cpu=1, timeout=300)
def inspect_image() -> str:
    return subprocess.run(["bash", "-c", "ls -la /opt; ls /opt/llama.cpp | head; ls /opt/llama.cpp/build/bin 2>&1 | head; "
                           "pip show gguf 2>&1 | head -3; cat /proc/mounts | grep -i opt"],
                          capture_output=True, text=True).stdout


@app.local_entrypoint()
def inspect():
    print(inspect_image.remote())


# ---------------------------------------------------------------------------
# Amendment 4: larger check, CUDA build on A10G, 3 tasks x 50 docs, pooled rule.
#     modal run modal_app/llamacpp_check.py::main_v2 --usd 4
# ---------------------------------------------------------------------------
TASKS_V2 = ("cord_total", "sroie_all", "wr_money")
BASE_GGUF_DIR = OUT  # lm-f16.gguf + mmproj-f16-16x.gguf from amendment 2 (unfused, 16x)
OUT_V2 = VOL / "gate0/llamacpp" / f"{MODEL_KEY}_{ARM}_v2"

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev", "libgomp1")
    .run_commands(
        f"git clone https://github.com/ggml-org/llama.cpp {LLAMA}",
        f"cd {LLAMA} && git checkout {LLAMA_COMMIT}",
        f"cd {LLAMA} && cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"
        " -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
        " && cmake --build build -j 16 --target llama-mtmd-cli",
    )
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(f"pip install -r {LLAMA}/requirements/requirements-convert_hf_to_gguf.txt "
                  f"&& pip install -e {LLAMA}/gguf-py")
    .pip_install("transformers>=5.21", "huggingface_hub>=1.10", "pillow", "python-dotenv")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("minicpm_t2l", "t2l")
)


@app.function(image=cuda_image, gpu="A10G", volumes={VOL: vol}, secrets=secrets, cpu=8, memory=32768,
              timeout=3 * 3600)
def check_task_v2(task: str) -> list[dict]:
    out = OUT_V2 / task
    out.mkdir(parents=True, exist_ok=True)
    log = out / "commands.log"
    log.write_text(f"llama.cpp {LLAMA_COMMIT} CUDA\n")
    work = Path("/tmp/work")
    hf = _hf_dir_16x(work)
    lm, mm = BASE_GGUF_DIR / "lm-f16.gguf", BASE_GGUF_DIR / "mmproj-f16-16x.gguf"
    lora = work / f"lora-{task}-f16.gguf"
    adapter = VOL / "gate0" / RUNS / MODEL_KEY / ARM / task / "adapter"
    _sh(["python", str(LLAMA / "convert_lora_to_gguf.py"), str(adapter), "--base", str(hf),
         "--outtype", "f16", "--outfile", str(lora)], log)
    shutil.copy(lora, out / lora.name)

    tdir = VOL / "gate0/tasks" / task
    prompt = json.loads((tdir / "task.json").read_text(encoding="utf-8"))["prompt"]
    docs = [json.loads(l) for l in (tdir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def gen(img: Path, adapter_gguf: Path | None) -> str:
        cmd = [str(LLAMA / "build/bin/llama-mtmd-cli"), "-m", str(lm), "--mmproj", str(mm),
               "--image", str(img), "-p", prompt, "--temp", "0", "-n", "768", "--repeat-penalty", "1.0",
               "--jinja", "-ngl", "99", "--seed", "0", "--no-warmup"]
        if adapter_gguf is not None:
            cmd += ["--lora", str(adapter_gguf)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n--- stdout\n{r.stdout}\n--- stderr tail\n{r.stderr[-4000:]}\n")
        if r.returncode != 0:
            vol.commit()
            raise RuntimeError(f"mtmd-cli failed ({r.returncode}):\n{r.stderr[-3000:]}")
        return r.stdout.strip()

    # mtmd-cli does not log devices by default; ask llama.cpp directly and refuse to run on CPU
    dev = subprocess.run([str(LLAMA / "build/bin/llama-mtmd-cli"), "--list-devices"], capture_output=True, text=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ --list-devices\n{dev.stdout}\n{dev.stderr[-2000:]}\n")
    if "CUDA0" not in dev.stdout + dev.stderr:
        vol.commit()
        raise RuntimeError(f"llama.cpp sees no CUDA device:\n{dev.stdout}\n{dev.stderr[-2000:]}")

    def torch_preds(arm):
        p = VOL / "gate0" / RUNS / MODEL_KEY / arm / task / "predictions.jsonl"
        return {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    tb, tl = torch_preds("base"), torch_preds(ARM)
    rows = []
    t0 = time.time()
    for d in docs:
        img = tdir / d["image"]
        rows.append({"task": task, "id": d["id"], "label": d["label"],
                     "llama_base": gen(img, None), "llama_lora": gen(img, lora),
                     "torch_base": tb[d["id"]], "torch_lora": tl[d["id"]]})
    with open(out / "outputs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "timing.json").write_text(json.dumps({"gen_seconds": time.time() - t0, "n_docs": len(docs)}))
    vol.commit()
    return rows


def score_v2(rows: list[dict]) -> dict:
    from minicpm_t2l.scoring import TaskScore, score_doc

    arms = ("llama_base", "llama_lora", "torch_base", "torch_lora")

    def block(rs):
        f1 = {}
        for a in arms:
            ts = TaskScore()
            for r in rs:
                ts.add(score_doc(r[a], r["label"]))
            f1[a] = ts.summary()["f1"]
        return {"n_docs": len(rs), "f1": f1,
                "torch_gain": f1["torch_lora"] - f1["torch_base"],
                "llama_gain": f1["llama_lora"] - f1["llama_base"],
                "exact_agreement_lora": sum(r["llama_lora"].strip() == r["torch_lora"].strip() for r in rs) / len(rs),
                "exact_agreement_base": sum(r["llama_base"].strip() == r["torch_base"].strip() for r in rs) / len(rs)}

    pooled = block(rows)
    pooled["pass_f1"] = pooled["f1"]["llama_lora"] >= pooled["f1"]["torch_lora"] - 0.05
    pooled["pass_gain"] = pooled["llama_gain"] >= 0.5 * pooled["torch_gain"]
    pooled["pass"] = pooled["pass_f1"] and pooled["pass_gain"]
    per_task = {t: block([r for r in rows if r["task"] == t]) for t in TASKS_V2}
    return {"commit": LLAMA_COMMIT, "build": "CUDA sm_86, A10G", "model": MODEL_ID, "arm": ARM,
            "pooled": pooled, "per_task": per_task}


@app.local_entrypoint()
def main_v2(usd: float = 4.0):
    from minicpm_t2l.budget import Budget

    b = Budget()
    rid = b.reserve("gate0", usd, run="gate0-llamacpp-check-v2", note="amendment 4")
    t0 = time.time()
    try:
        rows = [r for part in check_task_v2.map(TASKS_V2) for r in part]
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__, wall_seconds=time.time() - t0)
        raise
    b.finish(rid, "completed", wall_seconds=time.time() - t0)
    res = score_v2(rows)
    Path("data/llamacpp").mkdir(parents=True, exist_ok=True)
    Path("data/llamacpp/v2_outputs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    Path("studies/gate0/llamacpp_v2_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
