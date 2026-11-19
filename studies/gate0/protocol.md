# Gate 0 protocol: does ordinary LoRA give headroom?

Status: **FROZEN 2026-11-19** (approved by the user, together with the budget cap and the data downloads). Changes go in the dated amendments section at the bottom; the sections above are not edited.

Code: `t2l/tasks.py` (tasks, splits), `t2l/scoring.py` (metric), `modal_app/gate0.py` (runs), `t2l/gate0_report.py` (decision). Constants in those files must match this document.

## Question

On narrow document-to-JSON tasks, does an ordinary per-task LoRA on a frozen MiniCPM base clearly beat both the base model and the same model given examples in the prompt? If it doesn't, the hypernetwork has nothing to emit, and PLAN.md says stop.

## Models

| Key | Model | Notes |
| --- | --- | --- |
| `v47` | `openbmb/MiniCPM-V-4.7-1B` | General VLM. Loads with `AutoModelForImageTextToText` (transformers ≥ 5.18 per its config) |
| `ocr` | `openbmb/MiniCPM-OCR-1.0` | Built on MiniCPM-V 4.6. Needs transformers ≥ 5.21. Its README only documents a layout-crop pipeline with per-region prompts, so it may not follow free-form JSON instructions. If that shows up as a parse-failure rate above 50% at base, it's reported as such; we don't add special prompting for it |

**Architecture finding (2026-11-19, from config + meta-device build).** The language model is a Qwen3.5 hybrid with 24 layers. Only 6 of them (every 4th) are full attention with `q_proj/k_proj/v_proj/o_proj`. The other 18 are linear attention (Gated DeltaNet) with `in_proj_qkv`, `in_proj_z`, `in_proj_a/b` and `out_proj`. PLAN.md's "rank-8 LoRA on q and v" therefore touches only 6 layers, so Gate 0 tests two target sets:

| Set | Modules | Count |
| --- | --- | --- |
| A (plan as written) | `self_attn.q_proj`, `self_attn.v_proj` | 12 |
| B | A + `linear_attn.in_proj_qkv`, `linear_attn.out_proj` | 48 |

Vision tower, resampler and MTP modules are never adapted. The module paths are checked by `probe` before any training.

## Tasks (20)

Each task is one source plus one output schema. Prompts come from `TaskSpec.prompt()`.

| Source (Hub / URL) | Tasks | Licence (as stated; not legally reviewed) |
| --- | --- | --- |
| CORD v2 `naver-clova-ix/cord-v2` | `cord_menu`, `cord_subtotal`, `cord_total`, `cord_full` | CC-BY-4.0 |
| SROIE `jsdnrs/ICDAR2019-SROIE` | `sroie_company`, `sroie_date`, `sroie_address`, `sroie_total`, `sroie_all` | CC-BY-4.0 per uploader; original release unclear |
| WildReceipt `podbilabs/wildreceipt-donut` | `wr_store`, `wr_datetime`, `wr_products`, `wr_money` | Unstated; unclear |
| FUNSD `davidle7/funsd-json` | `funsd_qa`, `funsd_headers` | Original: non-commercial research only |
| XFUND (GitHub release v1.0) | `xfund_{zh,ja,es,fr,de}_qa` | CC BY-NC-SA 4.0 |

EPHOIE is left out: there's no Hub copy, and access requires a signed agreement.

Licence note: FUNSD and XFUND are non-commercial. That's fine for a research gate. Before anything ships, PLAN.md's licence record must say whether any weights trained on them are released.

## Splits

- **Official test splits are never used** in Gate 0. They stay untouched, so later gates can still treat whole datasets as held out.
- Eval: 50 docs per task. They come from the official validation split when one exists (CORD, WildReceipt, XFUND). Otherwise 50 are carved from train with seed `20261119` (SROIE, FUNSD).
- Train: the remaining train docs, shuffled with the same seed and capped at 200.
- Tasks that share a source share the same doc ids. A doc's label can be empty for a task, e.g. a CORD receipt with no subtotal block.

## Arms (same eval docs, same decoding for all)

1. `base`: zero-shot.
2. `icl1`, `icl4`: k = 1 or 4 (image, JSON) example turns before the query. Examples are drawn once per task from train with seed 7. k = 8 is left out on cost grounds.
3. `loraA`, `loraB`: one LoRA per task and target set, trained on that task's train split.

Rows 3 and 5 of PLAN.md's comparison (multi-task LoRA, emitted adapter) belong to Gate 1. Row 4 (oracle LoRA) is what `loraA/B` measures here.

## Settings (frozen with this document)

- Processor: `downsample_mode="16x"`, `max_slice_nums=9`, `enable_thinking=False`.
- Decoding: greedy, `max_new_tokens=768`, repetition penalty 1.0.
- LoRA:
  - rank 8, alpha 16, dropout 0.05
  - AdamW, lr 2e-4, no weight decay, cosine schedule with 5% warmup, grad-clip 1.0
  - 3 epochs, batch 1 × grad-accum 8, bf16 weights
  - loss on assistant tokens only
  - one configuration for every task, with no per-task tuning
- Hardware: Modal L4 for every arm.

## Metric

Field-level micro-F1 over (path, value) pairs (`t2l/scoring.py`):

- Values are normalised with NFKC, casefold, collapsed whitespace and stripped edge punctuation.
- Lists are compared as multisets.
- Unparseable output scores as an empty prediction.

Exact match and parse-failure rate are also reported.

## Pass rule (per model × target set)

A task is a **win** when the LoRA's F1 minus the best of {base, icl1, icl4} on that task is at least **+0.05**, and the paired bootstrap 95% CI (2000 resamples over docs) lies above 0.

**Gate 0 passes** for a (model, target set) when there are at least **11 wins out of 20** and no task is missing.

The base model for later gates is the passing model with more wins, ties broken by mean gain. The target set chosen is the one that passes with more wins; when the two are within 2 wins, the smaller set A is preferred.

The llama.cpp check from PLAN.md is a separate step, run only after a pass: convert one adapter to GGUF and compare its outputs with PyTorch on 20 eval docs. It's recorded here as an amendment.

## Run order and cost

| Step | Entrypoint | GPU-h (estimate) | USD (estimate) |
| --- | --- | ---: | ---: |
| Probe module names | `probe` | CPU | < 0.5 |
| Build tasks onto volume | `build` | CPU | < 1 |
| v47 baselines (3,000 generations) | `baselines --model v47` | 3–7 | 3–6 |
| v47 LoRA A, B (40 trainings) | `loras --model v47 --targets A/B` | 10–14 | 8–12 |
| ocr baselines | `baselines --model ocr` | 3–7 | 3–6 |
| ocr LoRA A, B | `loras --model ocr --targets A/B` | 10–14 | 8–12 |

Estimates assume about USD 0.80/h for an L4; that price is unverified. The total is about USD 25–37 against the USD 40 Gate 0 cap. If spend after the v47 runs projects past the cap, the ocr runs drop to set A only, and the user decides before anything runs.

Every entrypoint reserves through `t2l.budget`, which refuses until the user confirms the cap.

## Before running

- [x] User approves the budget cap (set `confirmed_by_user` in `config/budget.json`)
- [x] User approves this protocol, which freezes it
- [x] User OKs the data downloads into the Modal volume, about 4.5 GB in total (CORD about 2.3 GB, SROIE about 0.5 GB, WildReceipt about 0.2 GB, FUNSD < 0.1 GB, XFUND zh/ja/es/fr/de about 1.2 GB)
- [x] `probe` confirms target-set module counts (12 / 48) on both models

## Amendments

### 1. 2026-11-19: hardware L4 -> A10G for every arm
Modal L4 supply was too low to run more than one container at a time (a trivial L4 job took about 6 minutes to start), while A10G started in seconds. All arms now run on A10G, which is the same 24 GB class. The few results that had already finished on L4 (some MiniCPM-V `cord_menu` baselines) are kept apart under `gate0/runs` on the volume and are not used. Every reported result comes from `gate0/runs_a10g`. Nothing else changed. This decision was made before any LoRA result existed.

### 2. 2026-11-19: llama.cpp check, fixed before running it
Written while the OCR LoRA runs were still finishing, before any llama.cpp output existed.

- **What:** OCR model (`openbmb/MiniCPM-OCR-1.0`), set B adapter for `cord_total`, the first 20 docs of its eval split. `cord_total` was picked for its large PyTorch gain and short outputs, so 768-token truncation can't affect it.
- **llama.cpp:** commit `51a77c4f247ce96fb932e29c9f81625ca9d3b0b7`, CPU build, `llama-mtmd-cli`.
- **Model files:** language model and mmproj converted from the HF checkpoint with that commit's `convert_hf_to_gguf.py` at F16. `downsample_mode` is set to `16x` to match the PyTorch runs; the official GGUF repo only ships a 4x mmproj. The adapter is converted with `convert_lora_to_gguf.py` at F16.
- **Decoding:** the same prompt text, chat template from the GGUF (`--jinja`, `enable_thinking=false`), temperature 0, `-n 768`, repeat penalty 1.0.
- **Arms:** llama.cpp base and llama.cpp + LoRA. They're compared with the PyTorch `base` and `loraB` predictions on the same 20 docs, scored with `t2l/scoring.py`.
- **Pass:**
  - llama.cpp+LoRA F1 ≥ PyTorch LoRA F1 − 0.05, and
  - the llama.cpp gain (LoRA − base) is at least half the PyTorch gain on those docs.

  Exact-output agreement with PyTorch is reported but is not part of the pass rule, because the numerics (bf16 GPU vs F16 CPU) differ.

Caveat found while preparing this: the OCR model's own default is `downsample_mode: 4x` (its `config.json` and `preprocessor_config.json`). The frozen protocol ran both models at 16x. The checkpoint does contain trained 16x merger weights (`vision_tower.vit_merger.*`), but 16x is not its default, which may lower its absolute scores. This is not changed for Gate 0. It is noted for the base-model choice in later gates.

### 3. 2026-11-19: llama.cpp check uses an unfused base GGUF
Made before any llama.cpp generation had run.

By default, `convert_hf_to_gguf.py` fuses `in_proj_qkv` and `in_proj_z` into one `attn_qkv_gate` tensor (and `in_proj_b`/`in_proj_a` into `ssm_beta_alpha`). For a LoRA it skips that fusion, so the set B adapter converts to `blk.N.attn_qkv` tensors, and llama.cpp refuses a LoRA tensor with no matching base tensor. `src/models/qwen35.cpp` also accepts the unfused layout (separate `attn_qkv` / `attn_gate`), so the base GGUF for the check is converted with the merge switched off. The weights are identical, only split into two tensors.

Deployment consequence: our adapters cannot be applied to the official fused GGUFs as-is. The agent must ship its own unfused base GGUF, or each adapter must be fused (zero-padded `lora_b` rows for the gate part) before use.


### 4. 2026-11-19: larger llama.cpp check, fixed before running it
Written after amendment 2's check failed `pass_f1` by 0.0001 F1. That check had 20 docs, and all of its gap came from one doc. Amendment 2's result stands as recorded. This is a second, larger check, reported next to it, and it does not overwrite it.

- **What:** OCR model, set B adapters, **all 50 eval docs** of three tasks from three datasets: `cord_total`, `sroie_all` and `wr_money`. 150 docs in total.
- **llama.cpp:** same commit `51a77c4`, but a **CUDA build** (`GGML_CUDA=ON`, sm_86) running on A10G with all layers offloaded (`-ngl 99`). This is needed for cost, since CPU took about 100 s per generation. The GPU numerics differ from an on-device CPU or Metal build, so this tests that the adapter is applied correctly, not the device's numerics.
- **Model files:** the same base and mmproj GGUFs as amendment 2 (unfused, F16, 16x), reused from the volume. Each task's adapter is converted the same way as in amendment 2.
- **Decoding and prompts:** unchanged from amendment 2.
- **Pass (decided on all 150 docs pooled, micro-F1 via `t2l/scoring.py`):**
  - pooled llama.cpp+LoRA F1 ≥ pooled PyTorch LoRA F1 − 0.05, and
  - pooled llama.cpp gain ≥ 0.5 × pooled PyTorch gain.

  Per-task F1, gains and exact-output agreement are reported but do not decide the result.
