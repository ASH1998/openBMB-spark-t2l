# Pilot P-01 results: one multi-task LoRA on the 20 Gate 0 tasks

Run on 2026-11-19 under the frozen [protocol](protocol.md), with amendment 1 (cap raised to USD 14). The adapter trained on 3,543 examples (1,328 optimizer steps) and all 20 evaluations finished. Per-doc data: [per_doc.json](per_doc.json); analysis: [report.json](report.json) (`python -m t2l.pilot_report p01`); adapter and predictions on the volume under `pilots/p01_multitask/`.

## Result: the shared adapter is *better* than the per-task ones

| | Value |
| --- | --- |
| Multi-task LoRA wins over the best baseline (Gate 0 rule) | **16 / 20** (per-task LoRA B: 14 / 20) |
| Mean gain over the best baseline | **+0.215** (per-task LoRA B: +0.149) |
| Multi-task vs per-task LoRA B | better on 9, worse on **0**, no clear difference on 11 |
| Retention (multi-task gain / per-task gain, over LoRA B's win tasks) | **1.26** |

By the protocol's reading this is a **strong comparison 3**. Pooling the 20 tasks helps instead of interfering:
- The biggest gains are on the small form sets. All five XFUND languages gain +0.07 to +0.18 F1 over their per-task LoRA, and `funsd_qa` goes from 0.131 to 0.319.
- Those per-task LoRAs had only 99–149 training docs each.
- No task gets significantly worse.

## What it means for Gate 1

- **On tasks it has seen, one shared adapter is hard to beat.** The hypernetwork's case rests entirely on held-out tasks, which is the comparison Gate 1 already runs on the frozen held-out list.
- **This result is in-distribution.** Scored on held-out tasks it never saw, a multi-task LoRA has no task-specific signal, so its Gate 1 numbers will be lower than these.
- **A practical point for the hypernetwork:** shared structure across tasks is clearly learnable. Two designs follow from that and would likely make training easier:
  - start the emitted adapter from the multi-task LoRA and learn a per-task delta;
  - include the multi-task LoRA as a fixed part of every emitted adapter.

  That is a design decision for Gate 1's protocol and is not settled here.
- Under the pragmatic stance in PLAN.md (2026-11-19), Gate 1 still ships if the emitted adapter beats the base model on held-out tasks, even when it doesn't beat the multi-task LoRA.

## Per task (OCR-1.0, 16x, set B; 50 Gate 0 eval docs each)

| Task | Best baseline (arm) | Per-task LoRA B | Multi-task LoRA | MT − best [95% CI] | Win | MT − LoRA B [CI] |
| --- | ---: | ---: | ---: | --- | --- | --- |
| cord_menu | 0.520 (base) | 0.732 | 0.789 | +0.269 [+0.20, +0.35] | yes | +0.057 [-0.01, +0.11] |
| cord_subtotal | 0.214 (icl1) | 0.612 | 0.748 | +0.534 [+0.42, +0.65] | yes | +0.137 [+0.04, +0.25] |
| cord_total | 0.544 (icl1) | 0.818 | 0.763 | +0.220 [+0.13, +0.32] | yes | -0.055 [-0.13, +0.01] |
| cord_full | 0.425 (icl1) | 0.658 | 0.755 | +0.331 [+0.26, +0.41] | yes | +0.098 [+0.04, +0.16] |
| sroie_company | 0.620 (base) | 0.780 | 0.740 | +0.120 [-0.02, +0.26] | no | -0.040 [-0.12, +0.04] |
| sroie_date | 0.780 (base) | 0.900 | 0.960 | +0.180 [+0.08, +0.30] | yes | +0.060 [-0.02, +0.14] |
| sroie_address | 0.100 (base) | 0.340 | 0.460 | +0.360 [+0.20, +0.52] | yes | +0.120 [+0.02, +0.24] |
| sroie_total | 0.760 (base) | 0.900 | 0.900 | +0.140 [+0.00, +0.28] | no | +0.000 [-0.08, +0.08] |
| sroie_all | 0.570 (base) | 0.700 | 0.736 | +0.166 [+0.09, +0.24] | yes | +0.036 [-0.02, +0.09] |
| wr_store | 0.200 (icl1) | 0.235 | 0.231 | +0.031 [-0.04, +0.11] | no | -0.004 [-0.06, +0.05] |
| wr_datetime | 0.457 (base) | 0.467 | 0.455 | -0.003 [-0.10, +0.09] | no | -0.012 [-0.09, +0.06] |
| wr_products | 0.253 (icl4) | 0.439 | 0.437 | +0.184 [+0.13, +0.24] | yes | -0.002 [-0.05, +0.05] |
| wr_money | 0.293 (icl4) | 0.584 | 0.614 | +0.321 [+0.23, +0.41] | yes | +0.030 [-0.02, +0.09] |
| funsd_qa | 0.211 (icl1) | 0.131 | 0.319 | +0.108 [+0.04, +0.18] | yes | +0.189 [+0.12, +0.26] |
| funsd_headers | 0.143 (base) | 0.315 | 0.380 | +0.237 [+0.17, +0.31] | yes | +0.064 [-0.01, +0.14] |
| xfund_zh_qa | 0.058 (base) | 0.185 | 0.348 | +0.290 [+0.18, +0.38] | yes | +0.164 [+0.08, +0.25] |
| xfund_ja_qa | 0.055 (base) | 0.104 | 0.225 | +0.171 [+0.09, +0.25] | yes | +0.122 [+0.06, +0.19] |
| xfund_es_qa | 0.118 (base) | 0.247 | 0.319 | +0.201 [+0.12, +0.29] | yes | +0.072 [+0.00, +0.15] |
| xfund_fr_qa | 0.136 (icl1) | 0.195 | 0.309 | +0.173 [+0.09, +0.25] | yes | +0.114 [+0.06, +0.18] |
| xfund_de_qa | 0.159 (base) | 0.249 | 0.433 | +0.273 [+0.17, +0.37] | yes | +0.184 [+0.08, +0.30] |

## Caveats

- **The baseline is OCR at 16x.** P-03 shows that OCR's best baseline at its default 4x is higher (mean 0.412 against 0.331 here), so the gains above are measured against a weakened baseline. The *relative* finding, multi-task ≥ per-task, does not depend on that.
- **SROIE:** the training data includes Gate 0's SROIE train splits, which contain same-store near-duplicates of eval receipts (`studies/task_pool/results.md`). This affects both LoRA columns equally.
- **Cost:** the pilots app billed USD 12.92 in total for P-01 and P-03 (Modal billing, 23:41; the current hour may still settle), within the raised USD 14 cap.
