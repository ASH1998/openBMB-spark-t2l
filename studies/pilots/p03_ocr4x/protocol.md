# Pilot P-03 protocol: MiniCPM-OCR-1.0 baselines at its default 4x downsampling

Status: **FROZEN 2026-11-19**, before any run. Approved by the user in chat on 2026-11-19, with a USD 12 cap for P-01 and P-03 together, drawn from the overhead hold. Changes go in dated amendments at the bottom.

Code: `modal_app/pilots.py` (`p03`), `t2l/pilot_report.py` (analysis).

## Question

Gate 0 ran MiniCPM-OCR-1.0 at `downsample_mode="16x"`, but the model's own default is 4x. How much higher are OCR-1.0's baselines at 4x? Does a set-B LoRA (trained and run at 16x in Gate 0) still beat OCR-1.0's best 4x baseline on most tasks? The answer feeds the user's choice of Gate 1 base model: OCR-1.0 16x, OCR-1.0 4x or V-4.7-1B.

## Setup (everything as Gate 0 except the downsampling)

- Model: `openbmb/MiniCPM-OCR-1.0`, bf16, A10G, one job per container.
- Processor: **`downsample_mode="4x"`**, `max_slice_nums=9`, `enable_thinking=False`.
- Decoding: greedy, `max_new_tokens=768`, repetition penalty 1.0.
- Tasks and docs: Gate 0's 20 tasks, the same 50 eval docs in the same order (`gate0/tasks/<task>/eval.jsonl`), the same prompts.
- Arms: `base`, `icl1`, `icl4`. The in-prompt examples are the same docs as Gate 0 (drawn from the task's train split with seed 7).
- Metric: field-level micro-F1 (`t2l/scoring.py`), plus parse-failure rate and exact match.

No training happens in this pilot.

## Analysis (fixed now)

Per task, `best4x` is the highest F1 among base, icl1 and icl4 at 4x. The comparison numbers come from Gate 0's per-doc predictions (`gate0/runs_a10g`).

1. **Baseline shift:** `best4x` minus Gate 0's OCR `best16x`, with a paired bootstrap 95% CI. The bootstrap uses 2000 resamples over docs, seed 0, and compares the arm that is best per task in each setting.
2. **Headroom against the 4x baseline:** Gate 0's OCR `loraB` (16x) minus `best4x`. A **win** uses Gate 0's rule: at least +0.05 F1, with the CI above 0. Report wins out of 20. Note that this compares a 16x adapter with a 4x baseline. It answers "does the adapter path beat the model's own default setting", not "is there headroom at 4x".
3. **Absolute standing:** `best4x` against V-4.7-1B's best baseline and V-4.7-1B `loraB`, per task. Report the counts where OCR 4x is higher.

How the results will be read (fixed before seeing them; the user makes the final call):
- If `loraB` (16x) still wins on **≥ 11 of 20** tasks against `best4x`, OCR-1.0's headroom holds even against its default setting, and Gate 1 can keep OCR-1.0 at 16x.
- Otherwise, Gate 1 at 16x would be measuring gains over a deliberately weakened baseline. The base should then be either OCR-1.0 at 4x, which needs its own LoRA headroom check first because no 4x adapter has been trained, or V-4.7-1B. That choice goes to the user.

## Cost

60 evaluation jobs: 20 tasks × 3 arms × 50 docs, 3000 generations. Gate 0's OCR baseline jobs took 135–883 s each at 16x on A10G. At 4x the prompts are longer, so the estimate is 5–7 GPU-hours, **about USD 6–7.5**. Reservation: USD 7.5 (stage `overhead`). If spend reaches the reservation before all jobs finish, the remaining jobs are stopped and reported as missing, and nothing more runs without the user.

## Amendments

### 1. 2026-11-19: cost over the estimate (no change to method or analysis)
Written after all 60 jobs had finished and after the analysis had been run. It records cost only; nothing in the method or reading changes. The estimate of USD 6–7.5 was low: at 4x the XFUND jobs took 716–1126 s each, and the jobs used the GPUs for roughly 8–9 hours in total. The pilots' combined spend reached about USD 9.8 with P-01 still running. P-03 was not stopped early, because all of its jobs had finished by the time the overrun was noticed. On 2026-11-19 the user raised the combined cap for P-01 + P-03 from USD 12 to USD 14.
