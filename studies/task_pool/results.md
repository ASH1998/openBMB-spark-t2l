# Task pool results

Built on 2026-11-19 on Modal (CPU only) under the frozen [protocol](protocol.md). There are no amendments. Data is on the volume `minicpm-t2l-artifacts` under `task_pool/`. Here: [manifest.json](manifest.json) (every kept task with sizes, licence and held-out flag), [dedup_report.json](dedup_report.json), [task_cards.json](task_cards.json) (prompts, schemas, template parameters), [heldout_2026-11-19.json](heldout_2026-11-19.json) (**the frozen held-out list**), and the logs in `logs/`.

## Outcome: PASS

| | Tasks | Target |
| --- | ---: | ---: |
| Training | **365** | ≥ 300 |
| Held out | **78** | ≥ 30 |
| Dropped by the size rule | 3 | — |

The held-out set has two parts:
- **Whole public datasets:** FinTabNet (5 tasks), ChartQA (5) and the katanaml invoices (4).
- **All templates of 4 synthetic domains** picked by seed: payslip, purchase_order, restaurant_bill and shipping_label (64 tasks, 32 en / 32 zh).

The Gate 0 datasets are training only, as decided.

| Family | Training | Held out |
| --- | ---: | ---: |
| Synthetic forms (en / zh) | 320 (160 / 160) | 64 (32 / 32) |
| Receipts (CORD, SROIE, WildReceipt) | 17 | — |
| Forms (XFUND ×5 languages) | 15 | — |
| Tables (PubTabNet / FinTabNet) | 5 | 5 |
| Charts (DVQA / ChartQA) | 5 | 5 |
| Formulas (UniMER, HME100K) | 3 | — |
| Invoices | — | 4 |

By language, training tasks are 189 en, 164 zh, and 3 each of ja, es, fr and de. Held-out tasks are 46 en and 32 zh.

## Build

- **Public sources:** all 16 built, one container each.
- **Gate 0 tasks reproduce exactly.** For all 20 Gate 0 tasks, the rebuilt eval ids and first 200 train ids are identical to `gate0/tasks/` (`gate0_split_checks` in `logs/build_result_*.json`).
- **Synthetic templates:** 384 templates in 64 chunks, 290 docs each (50 eval + 240 train candidates), 111,360 rendered pages.
  - One chunk (6 shipping_label templates) failed with an empty `RemoteError`. The same templates render fine locally. It was re-run alone and succeeded (`logs/build_retry_chunk49.log`).
  - Every font role resolved to its intended Debian font; the build checks this.
- **Labels:** no label comes from a model. Public labels are the datasets' own annotations. Synthetic labels are the strings the renderer printed, and a test checks that every gold value appears among the drawn strings.

## Duplicate checks

117,214 documents were indexed.

| Check | Pairs found | Acted on |
| --- | ---: | ---: |
| Exact pixel hash | 8 | 2 SROIE pairs that cross train/eval |
| Near-duplicate text (Jaccard ≥ 0.8 on 5-gram shingles; 111,718 texts checked, 5,496 too short) | 251 crossing a boundary | yes |

266 actions in total, all "same task, train vs eval". Removals by source: SROIE 215 (43 per task × 5), XFUND 21, CORD 18, WildReceipt 12.

- **No duplicate crossed the training/held-out boundary.** No held-out doc was found in any training task, and no held-out split was changed.
- **The remaining 6 exact pairs cross no boundary,** so they're reported only: 1 SROIE pair, 4 ChartQA pairs and 1 FinTabNet pair inside validation.
- **Synthetic templates had no duplicates.**

**What this says about Gate 0.** Its train splits held near-duplicates of eval docs:
- **SROIE:** 43 of the 240 train candidates per task. Most are receipts from the same store as an eval receipt (same company and address), plus 2 exact image duplicates of eval receipts.
- **XFUND, CORD and WildReceipt:** a few each.

Gate 0's SROIE LoRA gains may therefore be somewhat inflated. Gate 0's results stand as recorded (`studies/gate0/` is read-only), but this is a caveat on its SROIE numbers. The pool's training splits for these tasks now exclude those docs.

## Tasks smaller than the nominal 200 / 50

- **Dropped (fewer than 100 train docs):** `funsd_qa`, `funsd_headers` and `funsd_questions`. FUNSD has 149 train docs; carving 50 for eval leaves 99. Gate 0 used these tasks with 99 train docs, but the pool's rule is 100, so they are out of the pool.
- **XFUND, all 15 tasks:** 145–149 train docs. The dataset is small, and 1–4 docs per language were removed as near-duplicates.
- **SROIE:** 197 train docs per task after the removals.
- **Short eval sets:** `chartqa_table_oecd` (held out) has 40 eval and 155 train docs, `inv_items` (held out) 48 eval, `dvqa_most_preferred` 46 eval. These sources have too few eligible docs; the held-out eval splits are as sampled.

## Licences (see the protocol for sources)

- **Commercially usable:** 333 of 365 training tasks (synthetic, CORD, PubTabNet, UniMER).
- **Non-commercial:** 20 (XFUND, DVQA).
- **Unclear:** 12 (SROIE, WildReceipt, HME100K).

An adapter or hypernetwork trained on any non-commercial or unclear task has to be marked non-commercial, or retrained without those tasks before a commercial release. The held-out ChartQA tables come from a GPL-3.0 dataset; they are used only for evaluation.

## Cost

Modal billing for 2026-11-19 shows USD 0.62 for the task-pool app (the current hour may still settle). The ledger books USD 0.90 against the USD 40 stage cap.
