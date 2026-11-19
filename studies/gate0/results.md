# Gate 0 results: does ordinary LoRA give headroom?

Run on 2026-11-19 on Modal under the frozen protocol in [protocol.md](protocol.md), with amendments 1 (A10G instead of L4) and 2 (llama.cpp check). The raw decision data is in [report.json](report.json); predictions, adapters and logs are on the Modal volume `minicpm-t2l-artifacts` under `gate0/runs_a10g/`. All 200 jobs completed with none failed.

## Decision

| Model | LoRA targets | Wins / 20 | Mean gain over best baseline | Gate 0 |
| --- | --- | ---: | ---: | --- |
| MiniCPM-OCR-1.0 | **B** (q/v + linear-attention in/out) | **14** | +0.149 | **PASS** |
| MiniCPM-OCR-1.0 | A (q/v only, 6 layers) | 7 | +0.075 | fail |
| MiniCPM-V-4.7-1B | B | 10 | +0.127 | fail (one short) |
| MiniCPM-V-4.7-1B | A | 10 | +0.085 | fail (one short) |

A win means the LoRA beats the best of {no examples, 1 example, 4 examples in the prompt} by at least 0.05 F1, with the paired-bootstrap 95% CI above 0. The rule needs 11 of 20.

**Gate 0 passes with MiniCPM-OCR-1.0 + target set B.** By the protocol's selection rule, that is the base model and target set for the next gates. Set B beats set A for both models, so adapting the linear-attention layers matters. The plan's "q and v only" would have reached only 6 of 24 layers.

**llama.cpp:**
- The 20-doc CPU check (amendment 2) failed its frozen rule by 0.0001 F1, with 19 of 20 outputs identical to PyTorch.
- The larger 150-doc CUDA check (amendment 4) **passes**: pooled LoRA F1 is 0.690 in llama.cpp vs 0.702 in PyTorch, and the gain is +0.300 vs +0.252.
- A PEFT adapter trained on set B applies correctly in llama.cpp, provided the base GGUF is unfused. See the section below.

## Per task: MiniCPM-OCR-1.0, set B (the passing combination)

F1 on 50 eval docs per task. The last three columns are for comparison only.

| Task | Best baseline (arm) | LoRA B | Gain [95% CI] | Win | OCR LoRA A | V-4.7 best baseline | V-4.7 LoRA B |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| cord_menu | 0.520 (base) | 0.732 | +0.212 [+0.13, +0.31] | yes | 0.695 | 0.642 | 0.767 |
| cord_subtotal | 0.214 (icl1) | 0.612 | +0.398 [+0.28, +0.53] | yes | 0.536 | 0.326 | 0.709 |
| cord_total | 0.544 (icl1) | 0.818 | +0.274 [+0.20, +0.35] | yes | 0.741 | 0.706 | 0.817 |
| cord_full | 0.425 (icl1) | 0.658 | +0.233 [+0.16, +0.31] | yes | 0.621 | 0.495 | 0.770 |
| sroie_company | 0.620 (base) | 0.780 | +0.160 [+0.04, +0.28] | yes | 0.620 | 0.653 | 0.880 |
| sroie_date | 0.780 (base) | 0.900 | +0.120 [+0.02, +0.22] | yes | 0.840 | 0.940 | 1.000 |
| sroie_address | 0.100 (base) | 0.340 | +0.240 [+0.12, +0.38] | yes | 0.242 | 0.200 | 0.560 |
| sroie_total | 0.760 (base) | 0.900 | +0.140 [+0.02, +0.26] | yes | 0.860 | 0.720 | 0.980 |
| sroie_all | 0.570 (base) | 0.700 | +0.130 [+0.07, +0.19] | yes | 0.620 | 0.630 | 0.800 |
| wr_store | 0.200 (icl1) | 0.235 | +0.035 [-0.03, +0.11] | no | 0.165 | 0.252 | 0.300 |
| wr_datetime | 0.457 (base) | 0.467 | +0.009 [-0.06, +0.08] | no | 0.474 | 0.469 | 0.528 |
| wr_products | 0.253 (icl4) | 0.439 | +0.187 [+0.14, +0.23] | yes | 0.337 | 0.432 | 0.458 |
| wr_money | 0.293 (icl4) | 0.584 | +0.290 [+0.19, +0.38] | yes | 0.521 | 0.481 | 0.640 |
| funsd_qa | 0.211 (icl1) | 0.131 | −0.080 [−0.16, −0.00] | no | 0.163 | 0.348 | 0.318 |
| funsd_headers | 0.143 (base) | 0.315 | +0.173 [+0.10, +0.25] | yes | 0.194 | 0.118 | 0.274 |
| xfund_zh_qa | 0.058 (base) | 0.185 | +0.126 [+0.03, +0.23] | yes | 0.073 | 0.414 | 0.382 |
| xfund_ja_qa | 0.055 (base) | 0.104 | +0.049 [+0.00, +0.10] | no | 0.125 | 0.213 | 0.276 |
| xfund_es_qa | 0.118 (base) | 0.247 | +0.130 [+0.04, +0.21] | yes | 0.130 | 0.316 | 0.381 |
| xfund_fr_qa | 0.136 (icl1) | 0.195 | +0.059 [−0.03, +0.15] | no | 0.076 | 0.378 | 0.403 |
| xfund_de_qa | 0.159 (base) | 0.249 | +0.089 [−0.01, +0.19] | no | 0.081 | 0.381 | 0.416 |

## What the numbers say

1. **LoRA headroom is real on receipts.** Both models gain clearly on CORD and SROIE: 9 of 9 tasks win for OCR set B and 8 of 9 for V-4.7 set B. Most gains are +0.1 to +0.4 F1.
2. **In-prompt examples rarely help these 1B–2B models,** and often hurt. For example, OCR `sroie_*` with 4 examples drops by 0.3–0.5. So "show a few examples in the prompt" is a weak alternative to an adapter, which supports the project idea.
3. **Forms (FUNSD, XFUND) are where LoRAs struggle.** V-4.7 won 1 of 7 form tasks. OCR set B won 4 of 7, but from very low baselines. `funsd_qa` gets worse with a LoRA for both models.
4. **Set B is better than set A** for both models (14 vs 7 wins for OCR, and a higher mean gain for V-4.7). The hypernetwork should emit adapters for the linear-attention projections as well as q/v.

## Caveats that matter for the next decision

- **OCR passes partly because its baseline is weak.** In absolute F1, **V-4.7 + LoRA B beats OCR + LoRA B on 17 of 20 tasks**. Gate 0 measures *headroom* (gain over the same model's baselines), not which model ends up best. The protocol's rule picks the passing model, OCR. If the deciding question is the best absolute quality on the device, V-4.7 + LoRA is stronger, even though it missed the headroom bar by one task.
- **OCR ran at 16x, which is not its default.** MiniCPM-OCR-1.0's own config uses `downsample_mode: 4x`. The frozen protocol used 16x for both models. The checkpoint has trained 16x merger weights, so the runs are valid, but its baseline at 4x may be much higher, and the headroom correspondingly smaller. This is untested.
- **The form-QA metric scores questions and answers separately,** not as linked pairs (the multiset rule in `t2l/scoring.py`). This treats every arm the same, but it inflates absolute F1 on `funsd_qa` and `xfund_*`.
- **The long-output tasks hit the 768-token limit.** XFUND documents have 26–46 pairs. Some outputs repeat themselves until the limit and then fail to parse. Parse-failure rates stay low overall (1–2.5% mean per arm), but greedy decoding with no repetition penalty is a known weak spot.
- **Only 50 eval docs per task.** CIs are wide (often ±0.1), and several "no" verdicts have positive point gains.
- **Licences:** FUNSD and XFUND are non-commercial. Any adapter trained on them must not ship in a commercial product.

## llama.cpp check (amendment 2)

**Result: FAIL under the frozen rule, by 0.0001 F1.** In practice the LoRA applies correctly.

Setup: MiniCPM-OCR-1.0 (16x) and the `loraB` adapter for `cord_total`, converted to f16 GGUF with llama.cpp `51a77c4`. The base GGUF is unfused (amendment 3). It ran with `llama-mtmd-cli` (greedy, 768 tokens, CPU) on the first 20 eval docs and was compared with the PyTorch predictions for the same docs. Data: `gate0/llamacpp/ocr_loraB_cord_total/result.json` and `outputs.jsonl` on the volume.

| | Base F1 | LoRA F1 | Gain |
| --- | ---: | ---: | ---: |
| PyTorch (bf16, A10G) | 0.488 | 0.907 | +0.419 |
| llama.cpp (f16 GGUF, CPU) | 0.457 | 0.857 | +0.400 |

- `pass_gain` passes: the llama.cpp gain is 96% of the PyTorch gain, and the rule needs 50% or more.
- `pass_f1` **fails**: llama.cpp+LoRA is 0.85714 and the threshold is 0.90722 − 0.05 = 0.85722.
- 19 of 20 LoRA outputs are byte-identical to PyTorch. All of the F1 gap comes from one doc, `cord_validation_23`. PyTorch matched both gold fields. llama.cpp misread the total and wrote the cash amount as "20,000" where the gold has "20.000", so it scored 0 of 3.
- Without the adapter, only 13 of 20 outputs agree. So the differences come from base inference (f16 vs bf16, CPU kernels, image preprocessing in mtmd), not from how the LoRA is applied.

The protocol is frozen, so this is recorded as a fail. My reading is that llama.cpp applies a PEFT LoRA trained on target set B correctly. What remains is the general numeric drift between llama.cpp and PyTorch, and with 20 docs one doc moves F1 by about 0.05. A larger check (more docs and tasks) would settle it. Any such check needs a new protocol written before it runs, not a change to this rule.

### Larger check (amendment 4): **PASS**

The larger check used a CUDA build of the same commit on A10G with all layers on the GPU (`--list-devices`: CUDA0, NVIDIA A10). It ran all 50 eval docs of three tasks, 150 docs in total, and the decision is on pooled micro-F1. Data: `studies/gate0/llamacpp_v2_result.json`, plus `gate0/llamacpp/ocr_loraB_v2/` on the volume.

| | Base F1 | LoRA F1 | Gain |
| --- | ---: | ---: | ---: |
| PyTorch, pooled (150 docs) | 0.450 | 0.702 | +0.252 |
| llama.cpp CUDA, pooled | 0.390 | 0.690 | +0.300 |

- `pass_f1` passes: llama.cpp+LoRA is 0.690, against a threshold of 0.702 − 0.05 = 0.652.
- `pass_gain` passes: the llama.cpp gain is 119% of the PyTorch gain, against the required 50%.

Per task (reported only, not part of the decision):

| Task | Torch base → LoRA | llama.cpp base → LoRA | Same output as PyTorch, LoRA / base |
| --- | --- | --- | --- |
| cord_total | 0.530 → 0.818 | 0.511 → 0.788 | 90% / 72% |
| sroie_all | 0.570 → 0.700 | 0.511 → 0.695 | 40% / 10% |
| wr_money | 0.219 → 0.584 | 0.158 → 0.582 | 34% / 16% |

Reading:
- The adapter carries over intact. With it, llama.cpp lands within 0.03 F1 of PyTorch on every task.
- The llama.cpp base model is 0.02–0.06 F1 weaker than the PyTorch base, so its measured gain is slightly larger.
- Exact output agreement is low on SROIE and WildReceipt. Those outputs are longer, so small numeric differences (f16 GGUF and mtmd image preprocessing vs bf16 PyTorch) change more tokens. Agreement is higher with the adapter than without it.
- This tests a GPU build. Quantised on-device builds (CPU or Metal) are still untested; that is pilot P-05.

Deployment consequence (amendment 3): the adapter needs the **unfused** base GGUF. The official fused GGUFs (`attn_qkv_gate`) reject LoRA tensors on `in_proj_qkv`. The other options are to fuse the adapter into the fused layout or to merge it into the weights.

## Cost

Modal billing for 2026-11-19: USD 31.18 for the Gate 0 app, plus USD 0.81 for GPU-availability tests and USD 0.09 for the first llama.cpp attempts, plus USD 1.35 for the llama.cpp check itself (16 CPU, about 80 min), plus USD 0.69 for the CUDA check (0.16 for an aborted attempt, 0.53 for the full run). Gate 0 total: USD 35.64 of the USD 40 cap. The ledger (`budget/ledger.json`) has these reconciled against the USD 40 Gate 0 cap.
