# Pilot P-01 protocol: one multi-task LoRA on the 20 Gate 0 tasks

Status: **FROZEN 2026-11-19**, before any run. Approved by the user in chat on 2026-11-19, with a USD 12 cap for P-01 and P-03 together, drawn from the overhead hold. Changes go in dated amendments at the bottom.

Code: `modal_app/pilots.py` (`p01`), `t2l/pilot_report.py` (analysis).

## Question

Gate 1 has to beat comparison 3, one multi-task LoRA trained on all training tasks. How strong is that bar? On the 20 Gate 0 tasks, does one shared LoRA keep most of the gain of the 20 per-task LoRAs?

This pilot is in-distribution: the multi-task LoRA has seen every task it is scored on. That makes it an optimistic bound for comparison 3, which in Gate 1 is scored on held-out tasks it has never seen. The leave-one-dataset-out variant from the log is **not** part of this pilot; it needs its own approval.

## Setup (Gate 0's passing combination)

- Model: `openbmb/MiniCPM-OCR-1.0`, bf16, **`downsample_mode="16x"`**, `max_slice_nums=9`, `enable_thinking=False`, A10G.
- LoRA: target set B (48 modules, checked at load time). Rank 8, alpha 16, dropout 0.05. AdamW, lr 2e-4, no weight decay, cosine schedule with 5% warmup, grad-clip 1.0, **3 epochs**, batch 1 × grad-accum 8, loss on assistant tokens only. These are the same settings as every Gate 0 LoRA, with no tuning.
- Training data: the union of the 20 Gate 0 tasks' `train.jsonl`, about 3,900 examples. Each example keeps its own task's prompt. Each epoch is shuffled with `random.Random(0)`.
- Evaluation: each task's 50 Gate 0 eval docs with greedy decoding and `max_new_tokens=768`, one container per task.

## Analysis (fixed now)

Per task, compared with Gate 0's OCR 16x runs on the same docs:

1. `mt` minus the best of {base, icl1, icl4}: wins by Gate 0's rule (≥ +0.05, paired bootstrap CI above 0), out of 20. Also the mean gain.
2. `mt` minus per-task `loraB`: the paired bootstrap difference and CI. Each task is counted as **better** (CI above 0), **worse** (CI below 0) or **no clear difference**.
3. **Retention:** (sum of `mt` gains) / (sum of `loraB` gains) over the tasks where `loraB` was a Gate 0 win, with gains measured over the best baseline.

How it will be read (fixed before seeing results):
- **Retention ≥ 0.8 with ≤ 3 tasks worse:** the multi-task LoRA is a strong comparison 3. The hypernetwork's case rests on held-out tasks; it can't rest on seen tasks.
- **Retention < 0.5:** tasks interfere in one shared adapter. That favours per-task adapters, emitted or from a library.

## Cost

Training: about 11,700 forward/backward passes. Gate 0 OCR set B ran at about 0.45 s per example, so this is about 1.5 GPU-hours. Evaluation: 20 × 50 generations, about 1.7 GPU-hours spread over containers. Estimate **about USD 4**; reservation USD 4.5 (stage `overhead`). If training alone exceeds 3 GPU-hours, it is stopped and reported.

## Amendments

### 1. 2026-11-19: cap raised
After P-03 overran its estimate, the user approved finishing P-01 under a combined pilots cap of USD 14, up from 12. This was decided while P-01 was still training, before any P-01 result existed. Method and analysis are unchanged.
