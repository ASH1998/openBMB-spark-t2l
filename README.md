# openT2L for MiniCPM-V: MiniCPM Spark Program entry

Show the model 4 pages of a document type it has never seen (receipts, payslips, shipping labels, financial tables) with the fields you want. It writes a LoRA adapter for MiniCPM-V-4.7-1B in one forward pass, with no fine-tuning on the device.

This repository holds the competition work: the document task pool, the experiments, and the Modal GPU runs. The model-agnostic hypernetwork is a separate library, [openT2L](https://github.com/ASH1998/openT2L) (import name `t2l`), which this code uses.

## Headline result (Gate 1)

The test set is 78 tasks never seen in training, held out as whole tasks: 64 synthetic Chinese and English forms from 4 unseen domains, plus 14 public tasks (charts, financial tables, invoices). Each adapter was emitted from 4 example pages. The metric is field-level F1. Decoding, pages and hardware (A10G) are identical across rows.

| Held-out tasks | Base MiniCPM-V-4.7-1B | Emitted adapter | Gain [95% CI, task bootstrap] | Better / worse |
| --- | ---: | ---: | --- | --- |
| All (78) | 0.728 | 0.881 | +0.154 [+0.116, +0.195] | 71 / 2 |
| Synthetic forms (64) | 0.813 | 0.957 | +0.145 [+0.104, +0.191] | 58 / 1 |
| Public (14) | 0.339 | 0.533 | +0.194 [+0.097, +0.305] | 13 / 1 |

Limits, stated plainly:
- On the held-out tasks we compared only against the base model. Comparisons with in-prompt examples, one multi-task LoRA, and a per-task oracle LoRA were cut to save budget (amendment 4 of the protocol).
- Chart gains are small.
- English synthetic forms were already near 0.95 for the base model.

Details: [studies/gate1/results.md](studies/gate1/results.md).

## Released weights

The hypernetwork behind these numbers is on Hugging Face: [ASHu2/openT2L-MiniCPM-V-4.7-1B-docs](https://huggingface.co/ASHu2/openT2L-MiniCPM-V-4.7-1B-docs) (CC-BY-NC-4.0).

| | |
| --- | --- |
| What it is | The final Gate 1 hypernetwork (run `final`, 6,000 steps), converted without changes to openT2L's safetensors + JSON format |
| Size | 72.2M parameters (fp32, 289 MB) |
| What it emits | A rank-8, alpha-16 LoRA for 48 modules of MiniCPM-V-4.7-1B's language model: q/v in the 6 full-attention layers, plus in_proj_qkv/out_proj in the 18 linear-attention layers. That is 1.8M parameters, about 3.6 MB in fp16, in standard PEFT format. |
| Input | 1–8 labelled example pages. Each is embedded by the frozen base model as the last LM hidden state averaged over the answer tokens, with processor settings `downsample_mode="16x"`, `max_slice_nums=9`. |
| Trained on | 333 tasks: 16 public datasets and synthetic forms. Real labels only. |

The model card has a complete example: embed 4 pages, emit an adapter, load it with PEFT.

The card's code was run as published: openT2L installed from GitHub, the weights downloaded from the Hub. It was run on 3 held-out tasks × 10 pages with the Gate 1 support pages (run G1-10, [modal_app/verify_hf_card.py](modal_app/verify_hf_card.py)).
- **Chinese purchase order:** F1 0.987, outputs identical to Gate 1 on 10/10 pages.
- **FinTabNet table:** F1 0.913, outputs identical to Gate 1 on 10/10 pages.
- **Invoice line items:** F1 0.721 vs 0.699 in Gate 1. 8/10 outputs are identical; PEFT applies the LoRA with slightly different numerics than the hooks used in Gate 1.

On the same pages the base model scores 0.26, 0.37 and 0.41. Results: [studies/gate1/hf_card_check.json](studies/gate1/hf_card_check.json). Because the training data includes non-commercial datasets (XFUND, DVQA) and some with unclear licences (SROIE, WildReceipt, HME100K), the weights and every adapter they emit are for non-commercial use.

## What we ran

Each gate has a protocol, frozen before its runs, plus its results. Later changes are dated amendments. Every experiment is in [experiments/LOG.md](experiments/LOG.md), failed and stopped runs included. A plain-language table is in [experiments/SUMMARY.md](experiments/SUMMARY.md).

| Stage | Question | Outcome |
| --- | --- | --- |
| [Gate 0](studies/gate0/results.md) | Does an ordinary per-task LoRA help MiniCPM on documents? Does it survive llama.cpp? | Yes, the adapters help. Layout B (all 24 layers) beats A. The adapter carries over to llama.cpp (150-page check). |
| [Task pool](studies/task_pool/results.md) | Build enough distinct tasks, with real labels only | 365 training and 78 held-out tasks from 16 public datasets and 384 synthetic templates, with duplicate checks across splits |
| [Pilot P-03](studies/pilots/p03_ocr4x/results.md) | Was MiniCPM-OCR handicapped by resolution? | Yes, so we chose V-4.7-1B as the base |
| [Pilot P-01](studies/pilots/p01_multitask/results.md) | How strong is one shared LoRA? | Very strong, so the tasks share structure |
| [Gate 1](studies/gate1/results.md) | Do emitted adapters beat the base model on unseen tasks? | Pass (table above) |

## Layout

```
minicpm_t2l/   task pool (pool, tasks, synth, synth_fields, dedup), scoring, Gate 1 recipe, reports, budget guard
modal_app/     the GPU and CPU jobs we ran on Modal (task pool, gate 0, pilots, gate 1, image build, llama.cpp check)
studies/       frozen protocols, results, reports and per-page outputs for each stage
experiments/   LOG.md (every run) and SUMMARY.md
budget/        ledger of every paid run; config/budget.json holds the caps
tests/
```

Notes on paths:
- The protocols and results are kept exactly as frozen, so they use the paths from when they were written. There, `t2l/<module>.py` and `python -m t2l.<module>` now mean `minicpm_t2l/<module>.py` and `python -m minicpm_t2l.<module>`, and `t2l.hyper` is now the `t2l` package from openT2L.
- Raw run logs (`studies/**/logs/*.log`, about 60 MB) are not included.

## Setup

```bash
git clone https://github.com/ASH1998/openT2L && pip install -e openT2L
pip install -e ".[data,modal,dev]"
cp .env.example .env    # Modal, Hugging Face and W&B credentials; never commit it
pytest
```

The GPU stages run on Modal. Build the GPU image once with `python -m modal_app.images`, then run, for example, `modal run modal_app/gate1.py::evaluate --arms base,emit4 --no-oracle`. Every paid entrypoint first reserves money in `budget/ledger.json` against the caps in `config/budget.json`, and refuses to start a run above them.

## Models, data and licences

- Base model: [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), under the MiniCPM model licence. It isn't redistributed here.
- Adapters target 48 modules: q/v in the 6 attention layers, plus the input and output projections in the 18 linear-attention layers. They are rank 8, alpha 16.
- Licences for each dataset are in [studies/task_pool/results.md](studies/task_pool/results.md) and `studies/task_pool/manifest.json`.
  - 333 of the 365 training tasks are commercially usable.
  - XFUND and DVQA are non-commercial.
  - SROIE, WildReceipt and HME100K have unclear licences.
  - Any hypernetwork or adapter trained on those tasks is released for **non-commercial use only**.
  - ChartQA (GPL-3.0) is used for evaluation only.
- Labels come from the datasets, or from the generator for synthetic forms. No teacher-model labels are used anywhere.
- Code: Apache-2.0.
