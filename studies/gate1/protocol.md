# Gate 1 protocol: can a hypernetwork emit a useful adapter from a few example pages?

Status: **FROZEN 2026-11-20.** The user approved this protocol and the USD 140 reservation (80 hypernetwork + 60 evaluation) in chat on 2026-11-20, before any Gate 1 run. Later changes go in dated amendments at the bottom; the sections above are not edited.

Inputs: the frozen task pool (`studies/task_pool/`, held-out list `heldout_2026-11-19.json`), the base-model decision D-01 (MiniCPM-V-4.7-1B) and the pragmatic stance in PLAN.md (2026-11-19).

## Question

Given k example pages with their correct JSON (k = 1, 4, 8) of a document type it has **never seen**, can a hypernetwork emit a LoRA in one forward pass that makes MiniCPM-V-4.7-1B better at that document type?

## Pass rule (pragmatic stance)

- **Primary, decides "ship":** on the 78 held-out tasks, the emitted adapter at **k = 4** beats the base model (comparison 1). The mean per-task F1 difference must be positive, with a 95% CI from a task-level bootstrap (2000 resamples, seed 0) above 0.
- **Full bar, reported but not deciding:** PLAN.md's original rule, which is beating comparisons 1 and 3 and recovering at least half of the oracle's gain (comparison 4 minus comparison 1), pooled over tasks.
- Every number is reported separately for the 64 synthetic held-out tasks and the 14 public held-out tasks (FinTabNet, ChartQA, invoices), and per task.

If the primary rule fails, PLAN.md's fallback applies: ship the agent with a library of trained adapters plus a router, and report the hypernetwork result as negative.

## Model and adapter

- Base: `openbmb/MiniCPM-V-4.7-1B`, bf16, frozen. Processor `downsample_mode="16x"` and `max_slice_nums=9` (checked 2026-11-20: both are this model's own defaults), `enable_thinking=False`.
- Adapter: LoRA **target set B**: `self_attn.q_proj` and `v_proj` on the 6 full-attention layers, `linear_attn.in_proj_qkv` and `out_proj` on the 18 linear-attention layers, 48 modules in all. Rank 8, alpha 16. The exact module shapes are read by a probe before any training and recorded here as an amendment.

## Hypernetwork (Text-to-LoRA "M" style, conditioned on pages)

1. **Example embedding.** For each support example (page image + prompt + gold JSON), run the frozen base model once. Take the last hidden layer (d = 1024) and concatenate two means: over all tokens, and over the gold-JSON tokens only. That gives a 2048-d vector, computed once per document and cached.
2. **Task embedding.** The mean of the k example embeddings, then a LayerNorm.
3. **Generator.** A shared MLP takes [task embedding, layer embedding (24), module-type embedding (4)]. It has 3 layers of width 512 with SiLU. One output head per module type emits the flattened A (8 × in) and B (out × 8).
   - The output is **shared part + task part**. The shared part is a learned per-module bias, which plays the role of a multi-task LoRA inside the hypernetwork; P-01 showed shared structure is strong. The task part's B head starts at zero, so training begins from "no task-specific change".
   - Size: about 35–70M parameters, depending on the probed shapes.
4. **Training (end to end, real labels only).**
   - Each step samples 8 training tasks.
   - Per task: k from {1, 2, 4, 8}, a support set and one disjoint query doc, both from that task's train split.
   - The adapter is emitted and applied to the frozen base, with loss on the query's gold JSON tokens only.
   - Optimiser: AdamW, lr 1e-4, weight decay 0.01, 300 warmup steps then cosine, grad-clip 1.0. **6000 optimizer steps** (48,000 query examples).
   - Training may use a larger GPU (A100) for speed. All evaluation runs on A10G.

## Data

- **Training:** the pool's 365 training tasks minus a dev set: train splits only, never their eval splits.
- **Dev set (for tuning only):** all templates of 2 training synthetic domains, picked with `random.Random(20261121)`: **gym_membership, utility_bill** (32 tasks). They are excluded from hypernetwork training and from the multi-task LoRA, and are used to choose learning rate, steps and embedding variant. **That leaves 333 training tasks.**
- **Test:** the 78 held-out tasks (`heldout_2026-11-19.json`). Nothing about them is used before the single final evaluation.
- **Licence:** 32 of the training tasks are non-commercial or unclear (XFUND, DVQA, SROIE, WildReceipt, HME100K), so the resulting hypernetwork weights are marked non-commercial. A commercial-only retrain on the 301 commercially usable training tasks is a possible later step and isn't part of this gate.

## The five comparisons (same decoding, same docs, same A10G class)

Decoding for all arms: greedy, `max_new_tokens=768`, repetition penalty 1.0. Per held-out task, all arms are scored on the same 50 eval docs (fewer for 3 tasks; see the pool results).

| # | Arm | Details |
| --- | --- | --- |
| 1 | `base` | No adapter, zero-shot |
| 2 | `icl1`, `icl4` | The same support pages as the emitted arms, placed in the prompt |
| 3 | `mt` | One multi-task LoRA (set B, Gate 0 LoRA settings) trained on the same 333 training tasks with the **same number of query examples** (48,000) as the hypernetwork |
| 4 | `oracle` | Per held-out task: an ordinary LoRA trained on that task's train split (≤ 200 docs), Gate 0 settings |
| 5 | `emit1`, `emit4`, `emit8` | The adapter emitted from k = 1, 4, 8 support pages |

Support pages: one draw per (task, k) from the held-out task's train split, seed 7, with nested draws (the k = 1 set is inside k = 4, which is inside k = 8). ICL and emission use identical pages.

## Run order, cost and budget

Prices assume A10G at about USD 1.10/h and A100 at about USD 2.50/h (not verified against the bill). Speeds are extrapolated from Gate 0 and the pilots, where generation dominated eval time.

| Step | What | GPU-h (est.) | USD (est.) | Stage |
| --- | --- | ---: | ---: | --- |
| 0 | Probe module shapes; smoke test (20 steps of training, emit + eval on 3 dev docs) | 0.5 | 1 | hypernetwork |
| 1 | Cache example embeddings for every train-split doc (training, dev, held-out tasks), about 90k forward passes | 4 | 5 | hypernetwork |
| 2 | Dev runs: up to 4 short runs (2000 steps each), evaluated on the 32 dev tasks | 14 | 20 | hypernetwork |
| 3 | Final hypernetwork, 6000 steps, chosen config; one rerun allowed if it crashes | 9–18 | 15–30 | hypernetwork |
| 4 | Comparison 3: multi-task LoRA, 48,000 examples | 9 | 10 | hypernetwork |
| 5 | Comparison 4: 78 oracle LoRAs | 10 | 12 | evaluation |
| 6 | Evaluation: 8 arms × 78 tasks × ≤ 50 docs ≈ 31,000 generations | 35 | 40 | evaluation |
| | **Total** | | **about 100–120** | caps: hypernetwork 350, evaluation 80 |

Budget proposal: reserve **USD 80** in the hypernetwork stage for steps 0–4 and **USD 60** in the evaluation stage for steps 5–6. The work stops and comes back to the user before either would be exceeded.

## Known limits (stated now)

- One support draw per (task, k): results include sampling noise from which pages were shown.
- Synthetic held-out tasks are 64 of 78. Held-out domains share the generator's layout families with training domains, so the public held-out group (14 tasks) is the harder and more honest test, and it is reported separately.
- The `mt` arm is matched on examples, not tuned further.

## Amendments

### 1. 2026-11-20: probed shapes (step 0), as the protocol requires
From the step-0 smoke run on A10G (`logs/smoke.log`), before any training run:
- 48 target modules. `self_attn.q_proj` is 1024→4096 and `self_attn.v_proj` is 1024→512 (6 layers each); `linear_attn.in_proj_qkv` is 1024→6144 and `linear_attn.out_proj` is 2048→1024 (18 layers each).
- The last hidden state has d = 1024, so example embeddings are 2048-d, as planned.
- The generator has **72.8M parameters**, slightly above the 35–70M estimate. That comes from the larger q_proj output (4096); no design change.
- The emitted adapter saved in PEFT format reloads identically.

### 2. 2026-11-20: step 1 costs more than estimated; user approved
Embedding ran at about 1.1 s per page on A10G, not the assumed 0.15–0.2 s. transformers logs that the `causal_conv1d` kernel isn't installed and falls back to a slower reference implementation. At that rate step 1 costs about USD 25–30 instead of 5. The run was stopped after 73 of 443 tasks, and the user told it to continue ("does not matter for a mere 30 dollar amount, run it"). It is relaunched with the cached tasks skipped, under a USD 30 reservation in the hypernetwork stage. Method and outputs are unchanged. Later steps will be slower than estimated in the same proportion; their costs are re-estimated from measured step-1 and dev-run speeds before they start.

### 3. 2026-11-20: runtime image with the `causal-conv1d` kernel (user-requested)
Made after amendment 2, before any training run. All Gate 1 jobs now run on a named Modal image (`modal_app/images.py`, published as `minicpm-t2l-gpu`). It has the same packages plus `causal-conv1d` 1.7.0, compiled for the GPU.

The check (`modal_app/kernel_check.py`, log `logs/kernel_check.log`) measured the following on A10G:
- The forward pass takes 0.16–0.18 s per page with the kernel, after a one-time warm-up. Preprocessing takes about 0.1 s. Before, a whole page took about 1.1 s.
- Example embeddings match those computed without the kernel at cosine similarity 0.99997 (4 pages).

The math is unchanged, so the 99 tasks embedded before the switch are kept, and method, data and decoding are unchanged. The cost estimates in the table above are expected to be close to correct again.

### 4. 2026-11-20: held-out evaluation cut to the primary comparison (user decision)
Decided while step 6 was running, before any held-out result had been looked at. At that point about 4–13 tasks per arm had finished, and none of their scores had been read.

The user judged that the project already had enough evidence and asked to cut the evaluation to essentials. Step 6 was stopped and is rerun with only two arms on all 78 held-out tasks: **`base`** and **`emit4`**. Finished arm outputs on the volume are reused.

Not run:
- `icl1`, `icl4`, `emit1`, `emit8`;
- `mt` (comparison 3);
- the 78 `oracle` LoRAs (comparison 4; step 5).

Consequences, stated now:
- The primary pass rule (emit4 beats base, task-level bootstrap CI above 0) is unchanged and still decides "ship".
- The full bar (beating the multi-task LoRA and recovering half the oracle's gain) is **not evaluated**. Results must say so, and must not claim anything about comparisons 2–4 on held-out tasks.
- CLAUDE.md asks for all five comparisons on every result. This gate reports two, at the user's explicit request. The trained multi-task LoRA is kept on the volume (`gate1/mt/adapter`), so comparison 3 can still be run later.

