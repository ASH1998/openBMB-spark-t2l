# Pilot P-03 results: MiniCPM-OCR-1.0 at its default 4x

Run on 2026-11-19 under the frozen [protocol](protocol.md), with amendment 1 (cost only). All 60 jobs finished and none failed. Per-doc data: [per_doc.json](per_doc.json); analysis: [report.json](report.json) (`python -m t2l.pilot_report p03`); predictions on the volume under `pilots/p03_ocr4x/`. A processor check run beforehand (log row P-03a) confirmed that `downsample_mode` takes effect: one CORD page is 280 input tokens at 16x and 848 at 4x.

## Result: the headroom does **not** hold against the 4x baseline

| | Value |
| --- | --- |
| Gate 0 OCR LoRA B (16x) wins against OCR's best 4x baseline | **7 / 20** (the rule needed ≥ 11) |
| Tasks where the 4x best baseline is significantly above the 16x one | 12 / 20 (none significantly lower) |
| Mean best-baseline F1, 16x → 4x | 0.331 → **0.412** |
| OCR 4x best baseline above V-4.7 best baseline | 8 / 20 tasks |
| OCR 4x best baseline above V-4.7 LoRA B | 2 / 20 tasks |

By the reading fixed in the protocol, Gate 0's OCR-1.0 pass at 16x partly measured gains over a deliberately weakened baseline. Gate 1 should therefore not simply keep OCR-1.0 at 16x. The base should be either:
- **OCR-1.0 at 4x,** which first needs its own LoRA headroom check, because no 4x adapter has been trained; or
- **MiniCPM-V-4.7-1B,** whose LoRA B beats OCR's best 4x baseline on 18 of 20 tasks in absolute F1, and which missed Gate 0's rule by one win.

That choice is the user's.

## Per task

F1 on Gate 0's 50 eval docs per task. "Win" uses Gate 0's rule: at least +0.05 F1 with the CI above 0. Note that LoRA B ran at 16x and the baseline at 4x.

| Task | Best 16x (arm) | Best 4x (arm) | Shift 4x − 16x [95% CI] | LoRA B 16x | LoRA B − best 4x [CI] | Win | V-4.7 best base | V-4.7 LoRA B |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: |
| cord_menu | 0.520 (base) | 0.679 (base) | +0.159 [+0.08, +0.25] | 0.732 | +0.053 [-0.07, +0.18] | no | 0.642 | 0.767 |
| cord_subtotal | 0.214 (icl1) | 0.296 (base) | +0.082 [+0.02, +0.14] | 0.612 | +0.316 [+0.16, +0.47] | yes | 0.326 | 0.709 |
| cord_total | 0.544 (icl1) | 0.685 (base) | +0.142 [+0.07, +0.22] | 0.818 | +0.133 [+0.04, +0.22] | yes | 0.706 | 0.817 |
| cord_full | 0.425 (icl1) | 0.501 (icl1) | +0.076 [+0.02, +0.14] | 0.658 | +0.157 [+0.03, +0.27] | yes | 0.495 | 0.770 |
| sroie_company | 0.620 (base) | 0.760 (base) | +0.140 [+0.04, +0.26] | 0.780 | +0.020 [-0.12, +0.14] | no | 0.653 | 0.880 |
| sroie_date | 0.780 (base) | 0.900 (base) | +0.120 [+0.02, +0.22] | 0.900 | +0.000 [-0.10, +0.12] | no | 0.940 | 1.000 |
| sroie_address | 0.100 (base) | 0.180 (base) | +0.080 [+0.00, +0.18] | 0.340 | +0.160 [+0.00, +0.32] | no | 0.200 | 0.560 |
| sroie_total | 0.760 (base) | 0.760 (base) | +0.000 [-0.10, +0.10] | 0.900 | +0.140 [+0.00, +0.28] | no | 0.720 | 0.980 |
| sroie_all | 0.570 (base) | 0.675 (base) | +0.105 [+0.06, +0.16] | 0.700 | +0.025 [-0.03, +0.08] | no | 0.630 | 0.800 |
| wr_store | 0.200 (icl1) | 0.325 (base) | +0.125 [+0.06, +0.20] | 0.235 | -0.090 [-0.18, +0.00] | no | 0.252 | 0.300 |
| wr_datetime | 0.457 (base) | 0.606 (base) | +0.149 [+0.06, +0.23] | 0.467 | -0.140 [-0.24, -0.03] | no | 0.469 | 0.528 |
| wr_products | 0.253 (icl4) | 0.317 (icl4) | +0.065 [-0.04, +0.17] | 0.439 | +0.122 [+0.01, +0.23] | yes | 0.432 | 0.458 |
| wr_money | 0.293 (icl4) | 0.396 (icl4) | +0.103 [+0.01, +0.20] | 0.584 | +0.187 [+0.08, +0.29] | yes | 0.481 | 0.640 |
| funsd_qa | 0.211 (icl1) | 0.284 (icl1) | +0.072 [+0.00, +0.14] | 0.131 | -0.153 [-0.25, -0.06] | no | 0.348 | 0.318 |
| funsd_headers | 0.143 (base) | 0.148 (icl1) | +0.006 [-0.02, +0.03] | 0.315 | +0.167 [+0.10, +0.24] | yes | 0.118 | 0.274 |
| xfund_zh_qa | 0.058 (base) | 0.039 (base) | -0.019 [-0.06, +0.01] | 0.185 | +0.145 [+0.05, +0.24] | yes | 0.414 | 0.382 |
| xfund_ja_qa | 0.055 (base) | 0.089 (base) | +0.034 [-0.01, +0.08] | 0.104 | +0.015 [-0.03, +0.05] | no | 0.213 | 0.276 |
| xfund_es_qa | 0.118 (base) | 0.182 (base) | +0.064 [+0.01, +0.12] | 0.247 | +0.066 [-0.02, +0.16] | no | 0.316 | 0.381 |
| xfund_fr_qa | 0.136 (icl1) | 0.191 (base) | +0.055 [-0.02, +0.13] | 0.195 | +0.004 [-0.09, +0.10] | no | 0.378 | 0.403 |
| xfund_de_qa | 0.159 (base) | 0.232 (base) | +0.073 [-0.01, +0.15] | 0.249 | +0.017 [-0.09, +0.14] | no | 0.381 | 0.416 |

## Notes

- **In-prompt examples still don't help** at 4x. The base arm is the best 4x baseline on 15 of 20 tasks, and icl4 hurts most tasks (e.g. `sroie_date` 0.90 → 0.42).
- **What this doesn't show:** whether a LoRA trained *at 4x* has headroom over the 4x baseline. That needs a new protocol and budget.
- **SROIE caveat:** the pool's duplicate check found same-store near-duplicates between Gate 0's SROIE train and eval splits (`studies/task_pool/results.md`). They affect the LoRA B column for SROIE, not the baselines.
- **Cost:** see the experiment log. The pilots app billed about USD 9.8 by the time P-03 finished, while P-01 was still running. That is well above the USD 6–7.5 estimate (amendment 1).
