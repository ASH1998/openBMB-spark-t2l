# Gate 1 results (2026-11-21)

Protocol: [protocol.md](protocol.md), frozen 2026-11-20, with amendments 1–4. Under amendment 4 the held-out evaluation compares only two arms: the base model (V-4.7-1B, no adapter) and emit4 (an adapter emitted from 4 example pages). Decoding, pages and hardware (A10G) are the same for both arms.

Sources: run G1-09 (`logs/evaluate.log`), per-page outputs in `per_doc.json`, analysis by `python -m t2l.gate1_report base,emit4` written to `report.json`.

## Primary result

| Group | Tasks | Base F1 | emit4 F1 | emit4 − base [95% CI, task bootstrap] | Better / worse tasks |
| --- | ---: | ---: | ---: | --- | --- |
| **All held-out** | 78 | 0.728 | 0.881 | **+0.154 [+0.116, +0.195]** | 71 / 2 |
| Synthetic forms (4 unseen domains) | 64 | 0.813 | 0.957 | +0.145 [+0.104, +0.191] | 58 / 1 |
| Public (ChartQA, FinTabNet, invoices) | 14 | 0.339 | 0.533 | +0.194 [+0.097, +0.305] | 13 / 1 |

"Better" and "worse" mean a change of more than 0.005 F1 on the task.

**Primary criterion (emit4 − base, lower 95% bound > 0): PASS.** The hypernetwork ships.

## Notes

- The gain comes mostly from tasks where the base model is weak: Chinese synthetic forms (for example `syn_purchase_order_zh_00`: 0.24 → 0.98) and FinTabNet tables (`fintabnet_first_col`: 0.21 → 0.79). On English synthetic forms the base model is already near 0.95 and the gain is small.
- The 2 tasks that got worse: `inv_parties` (0.622 → 0.520) and `syn_restaurant_bill_zh_02` (0.965 → 0.938).
- ChartQA gains are small (+0.01 to +0.27). Charts are the weakest area.

## Not evaluated (amendment 4)

The full bar was not tested, so it gets no pass or fail. That bar is: beat the multi-task LoRA, and recover at least 50% of the oracle gain. Comparisons 2 (in-prompt examples), 3 (multi-task LoRA) and 4 (oracle LoRA) were not run on the held-out tasks. Neither were emit1 or emit8. So these results do not show whether the emitted adapters beat a single shared adapter or in-prompt examples. The trained multi-task adapter is kept at `gate1/mt/adapter` on the volume, so comparison 3 can be added later.

Per-task scores are in `report.json` (`rows`).
