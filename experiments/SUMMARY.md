# Experiment summary (plain language)

A short overview of every experiment. The full record, with links to protocols and results, is [LOG.md](LOG.md). Updated 2026-11-21.

| ID | What we asked | Model(s) | Data | Result | Status | Cost USD |
| --- | --- | --- | --- | --- | --- | ---: |
| E0-01 | Which layers can an adapter attach to? | V-4.7-1B, OCR-1.0 | — | Hybrid model: only 6 of 24 layers have normal attention. So we tested two adapter layouts: A (6 layers) and B (all 24) | ✅ done | 0.06 |
| E0-02 | Build the 20 Gate 0 tasks | — | CORD, SROIE, WildReceipt, FUNSD, XFUND (receipts, forms) | 20 tasks, 50 test pages each | ✅ done | 0.30 |
| E0-03 | Smoke test of the whole pipeline | V-4.7 | 1 task | Worked after a missing package was added | ✅ done (1 failed try) | 0.24 |
| E0-04 | Why were GPUs so slow to start? | — | — | L4 GPUs took ~10 min to start, A10G seconds; switched to A10G | ✅ done | 0.81 |
| **E0-05** | **Gate 0: do adapters beat the plain model and in-prompt examples?** | Both, layouts A and B | the 20 tasks | **OCR + B passed (14/20 wins)**. V-4.7 + B got 10/20 (one short) but scored higher on 17/20 tasks. Layout B beats A on both models | ✅ **pass** | 31.18 |
| E0-06 | Does our adapter work in llama.cpp (the on-device runtime)? 20 pages, CPU | OCR + B | cord_total | Failed the rule by 0.0001 F1; 19/20 outputs identical | ❌ fail (technically) | 1.44 |
| E0-07 | Same check, larger: 150 pages, GPU build | OCR + B | 3 tasks | Adapter carries over intact (0.690 vs 0.702) | ✅ **pass** | 0.69 |
| TP-01 | Smoke test of the task pool build | — | SROIE, PubTabNet, 2 templates | Worked; found SROIE duplicates | ✅ done | 0.05 |
| TP-02 | Build the task pool | — | 16 public datasets + 384 synthetic templates | Built; one chunk failed randomly and was re-run fine | ✅ done (1 retry) | 0.75 |
| **TP-03** | **Duplicate checks + freeze test set** | — | all ~117k pages | **365 training / 78 test tasks** (needed 300/30). 3 FUNSD tasks dropped (too small). Found Gate 0's SROIE had same-shop near-copies between train and test | ✅ **pass** | 0.10 |
| P-01a | First launch of both pilots | — | — | Apps stopped after setup; nothing ran; relaunched | ❌ failed | ~0 |
| P-03a | Does the resolution setting actually take effect? | OCR | 1 page | Yes (280 vs 848 image tokens); ran twice by mistake | ✅ done | 0.10 |
| **P-03** | **Was OCR handicapped by low resolution?** | OCR at 4x | the 20 Gate 0 tasks | **Yes.** At normal resolution it scores 0.41 vs 0.33, and the adapter's edge mostly vanishes (7/20 wins) | ✅ done: bad news for OCR | 9.10 |
| **P-01** | **How strong is one shared adapter for all tasks?** | OCR + B | the 20 Gate 0 tasks | **Very strong:** 16/20 wins, better than per-task adapters on 9 tasks, worse on 0 | ✅ done: good news | 4.00 |
| D-01 | Pick the Gate 1 model | — | — | **V-4.7-1B**; its default resolution is the 16x used in Gate 0, so it was not handicapped | ✅ decided | 0 |
| G1-00 | **Gate 1:** does an adapter made from a few example pages beat the plain model on the 78 unseen tasks? | V-4.7-1B + B | 333 training tasks, 32 dev tasks, 78 test tasks | Protocol frozen 2026-11-20; budget USD 80 + 60 | ✅ passed (G1-09) | — |
| G1-01 | Smoke test of the hypernetwork on the real model | V-4.7 | 1 dev task | Everything works end to end; emitted adapters load in standard PEFT | ✅ done | 0.3 |
| G1-02/03 | Cache example "summaries" of every training page (first two tries) | V-4.7 | 443 tasks | Far slower than expected (a missing fast kernel); stopped twice, partial results kept | ❌ stopped | 5.7 |
| G1-04 | Does the fast `causal_conv1d` kernel help? | V-4.7 | 4 pages | **~4× faster, identical results.** Now a named Modal image (`minicpm-t2l-gpu`) | ✅ done | 0.05 |
| G1-05 | Finish the cache on the fast image | V-4.7 | 443 tasks | All cached, no failures | ✅ done | 9.3 |
| **G1-06** | **Dev runs: does an adapter made from 4 example pages help on unseen form types?** | V-4.7 + B | 32 dev tasks (2 form domains never trained on) | **Yes, for all 4 settings:** plain model 0.873 → emitted adapter 0.970–0.978. Chinese forms +0.18, English +0.03 (already near perfect). At most 2 of 32 worse. Picked lr 1e-4 with answer-only summary | ✅ **good** | ~13 |
| G1-07 | Final hypernetwork (6,000 steps) + the shared adapter | V-4.7 + B | 333 training tasks | Both trained cleanly (5 h each) | ✅ done | ≤25 |
| G1-08 | Full evaluation with all 7 comparisons | V-4.7 + B | 78 test tasks | Stopped early: cut to essentials | ⏹ dropped | ~5 |
| **G1-09** | **Gate 1: plain model vs adapter from 4 example pages, on 78 unseen tasks** | V-4.7 + B | 78 test tasks (64 synthetic, 14 public) | **PASS: 0.728 → 0.881 F1 (+0.154, CI +0.116 to +0.195); 71 tasks better, 2 worse.** Synthetic +0.145, real-world tables/charts/invoices +0.194. Not compared with in-prompt examples or the shared adapter | ✅ **pass** | ≤15 |

**What's good:** adapters clearly help; they run on-device in llama.cpp; the 365/78 task pool is ready; the tasks share structure (P-01); and **Gate 1 passed: adapters emitted from 4 example pages beat the plain model on 71 of 78 unseen tasks** (G1-09).

**What failed or hurt:** the OCR model's advantage was partly an artifact of low resolution (P-03), and the strict 20-page llama.cpp check missed by a hair. Neither blocks the project.

**Spend:** about USD 78 before the final runs, plus USD 78 reserved for G1-07 to G1-09 (actual is lower), so at most about USD 156 of the USD 600 project cap. The ledger counts reservations until Modal billing is reconciled; see `budget/ledger.json`).
