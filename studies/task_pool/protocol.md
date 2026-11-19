# Task pool protocol: build, split, check and freeze the held-out list

Status: **FROZEN 2026-11-19**, before any paid run of this stage. The user decided the hold-out policy, the pilots and the licence in chat on 2026-11-19. Changes go in the dated amendments section at the bottom; the sections above are not edited.

Code: `t2l/pool.py` (sources, tasks, splits), `t2l/synth.py` + `t2l/synth_fields.py` (synthetic templates), `t2l/dedup.py` (duplicate checks, final splits), `modal_app/task_pool.py` (runs). Constants in those files must match this document.

## Goal

Build the tasks the hypernetwork is trained and tested on:
- at least **300 training tasks**;
- at least **30 held-out tasks**, from whole datasets and whole synthetic templates;
- real labels only, with a licence recorded for every source.

Pass condition (PLAN.md): 300+ training tasks and 30+ held-out tasks survive the duplicate checks below.

## Task definition

A task is one document type plus one output schema: a fixed prompt (instruction + JSON shape) and a label per document. Prompts come from `PoolTask.prompt()` and use the same wording as Gate 0's `TaskSpec.prompt()`. Synthetic tasks use the instruction "Extract the listed fields from this document." plus their schema.

Families: receipt, form, table, chart, formula, invoice, synthetic_form.

## Sources and licences

Licences as stated by the source or uploader; none has been legally reviewed. `commercial` is our reading: **no** = adapters trained on it must be marked non-commercial; **unclear** = treat as no until checked.

| Source | Hub / URL | Split use | Role | Licence | Commercial |
| --- | --- | --- | --- | --- | --- |
| CORD v2 | `naver-clova-ix/cord-v2` | train / validation | training | CC-BY-4.0 | yes |
| SROIE | `jsdnrs/ICDAR2019-SROIE` | train (carve) | training | CC-BY-4.0 per uploader; original unclear | unclear |
| WildReceipt | `podbilabs/wildreceipt-donut` | train / validation | training | unstated | unclear |
| FUNSD | `davidle7/funsd-json` | train (carve) | training | non-commercial research | no |
| XFUND zh/ja/es/fr/de | GitHub release v1.0 | train / val | training | CC BY-NC-SA 4.0 | no |
| PubTabNet | `apoidea/pubtabnet-html` | train / validation, first 4000 / 1500 rows | training | CDLA-Permissive-1.0 | yes |
| DVQA | `vikhyatk/dvqa` | train (carve), first 6000 rows | training | CC BY-NC 4.0 (authors); Hub card empty | no |
| UniMER-1M | `deepcopy/UniMER` | train (carve), first 8000 rows | training | Apache-2.0 | yes |
| HME100K | `ConstantHao/HME100K` | train (carve), first 5000 rows | training | unclear (TAL, research use); Hub card empty | unclear |
| FinTabNet (en) | `apoidea/fintabnet-html`, config `en` | train / validation, first 4000 / 1500 rows | **held out** | CDLA-Permissive-1.0; Hub card empty | yes |
| ChartQA | `ahmed-masry/ChartQA` images + `nmayorga7/chartqa-tables` tables | train / val | **held out** | GPL-3.0 (ChartQA); tables copy MIT | unclear |
| Invoices | `katanaml-org/invoices-donut-data-v1` | train / validation | **held out** | MIT | yes |
| Synthetic forms | generated here (Faker, MIT) | generated | training + **held out** | ours | yes |

Not used, and why: EPHOIE (no public copy, needs a signed agreement); FATURA (the Hub copy's tag ids have no published names); PlotQA (the only clean labels are series values that the charts don't print); CROHME (no licence on the Hub copy, and UniMER-1M already contains CROHME and HME100K images, see the duplicate checks).

"First N rows" means the first N rows of the split in the Hub's shard order, streamed. Official **test splits are never read**.

## Tasks

### Public (62)

- **Gate 0 sources (35, training).**
  - Gate 0's 20 tasks, unchanged: the same prompts, the same doc ids, the same seed `20261119` and the same carve. The build checks the ids against `gate0/tasks/` on the volume and reports any mismatch.
  - Plus `cord_menu_names`, `cord_total_price`, `wr_total`, `wr_tel` and `funsd_questions`.
  - Plus `xfund_<lang>_headers` and `xfund_<lang>_questions` for each of the five languages.
- **PubTabNet (5, training)** and **FinTabNet (5, held out):** `<src>_header`, `_first_col`, `_shape`, `_last_row` and `_full`. Labels come from the HTML annotation (`parse_table`). `_full` keeps only tables with at most 8 rows, 6 columns and 40 cells.
- **DVQA (5, training):** one task per fixed question: largest bar, smallest bar, value of the largest bar, value of the smallest bar, most preferred object. The answer has its trailing period stripped.
- **UniMER (2, training):** `unimer_latex_short` (< 60 chars) and `unimer_latex_long` (60–239 chars). **HME100K (1, training):** `hme100k_latex` (< 240 chars). Labels are the dataset's space-tokenised LaTeX.
- **ChartQA (5, held out):** `chartqa_table_{statista,owid,pew,oecd}` (the data table, split by chart source using the image name) and `chartqa_columns`.
- **Invoices (4, held out):** `inv_header`, `inv_parties`, `inv_items` and `inv_summary`.

### Synthetic forms (384)

- 24 domains (`t2l/synth_fields.py`) × 2 languages (en, zh) × 8 templates.
- Each template is sampled with a seed (`SYNTH_SEED = 20261120`):
  - 4–9 schema fields from its domain, plus 0–3 extra printed fields that are not in the schema;
  - a line-item table for some domains;
  - one of 8 layout families;
  - fonts, sizes, label punctuation, date and money formats, a scan profile (clean, scan or rough), and key style (snake_case, camelCase or the printed label).
- Every template has a distinct schema: (domain, language, field set, items, item columns, key style) is unique.
- Values come from Faker (`faker==40.54.0`, seeded per document) and a seeded RNG. The label is exactly the printed strings. A blank field prints its label with no value, and its gold is `""`.
- Rendering uses Pillow 11.2.1 with the Debian font packages in `modal_app/task_pool.py`. The build fails if any font role doesn't resolve to its first candidate.

## Splits and sizes

- Per task: **eval = 50 docs**, **train = up to 200 docs**, plus up to 40 ordered reserve candidates for refilling after the duplicate check.
- **Public sources:**
  - Doc ids are shuffled with the source seed: `20261119` for Gate 0 sources, `20261120` for the others.
  - Eval candidates come from the official validation split when one exists. Otherwise the first `carve` shuffled train ids form the eval pool: 50 for Gate 0 sources, as in Gate 0, and 500 for new sources.
  - Each task then takes the first 50 eligible eval-pool docs and the first 240 eligible train docs. A doc is eligible when the task's extractor returns a label; for example, `_full` needs a small table, and a DVQA task needs its question.
  - Tasks of one source share doc ids.
- **Synthetic:** docs 0–49 of a template are eval, docs 50–289 are train candidates. Each doc has its own seed derived from (template, index).
- A task is kept only if the final train has at least **100** docs. A training task is also dropped if its eval falls below **40** docs after the duplicate check. Dropped tasks are listed with the reason.

## Held-out list rules

- **Held-out public datasets (whole):** FinTabNet, ChartQA, Invoices. Nothing from them is ever used to train or tune anything: not the hypernetwork, not the multi-task LoRA, and no hyperparameter choice.
- **Held-out synthetic domains (whole):** 4 of the 24 domains, picked by `random.Random(20261120).sample(sorted(domains), 4)` and not by hand. The result is **payslip, purchase_order, restaurant_bill, shipping_label**, which gives 64 templates (32 en, 32 zh). Field definitions are per domain, so held-out templates also contain value kinds (for example payslip line items) that no training template prints.
- **Gate 0 datasets are training only** (CORD, SROIE, WildReceipt, FUNSD, XFUND), because they were used for tuning in Gate 0.
- A held-out task's own train split is used only for its support examples (k pages for emission) and for its oracle LoRA (comparison 4).
- After the duplicate check, the list is frozen in a dated file next to this protocol (`heldout_<date>.json`) and never changed. Any later change is a new, dated file with a reason, and every result says which file it used.

Expected counts before the duplicate check: 368 training tasks (48 public + 320 synthetic) and 78 held-out tasks (14 public + 64 synthetic).

## Duplicate checks across splits

Run once over the whole pool by `t2l/dedup.py`, after everything is built:

1. **Exact image duplicates:** equal sha256 of the decoded RGB pixels plus the image size.
2. **Near-duplicate text:** Jaccard similarity ≥ **0.8** between character 5-gram shingle sets of the normalised document text. The text is every annotated string of the doc (all fields; table cells; chart table; LaTeX), normalised as in `t2l/scoring.py`. Only texts of at least **30** characters are checked. Candidate pairs come from MinHash LSH (64 permutations, 16 bands × 4 rows) and are confirmed with the exact Jaccard.

A duplicate pair is acted on only when it crosses a boundary:
- **Same task, train vs eval:** drop the train doc from that task.
- **Training task vs held-out task:** drop the doc from every training task, from both train and eval.

Eval splits of held-out tasks are never changed. Training splits are refilled from the ordered reserve. The report gives the number of pairs found, the number of actions, examples, and per-task removal counts. Pairs that cross no boundary (two training tasks from different sources, or two docs in the same split) are not acted on.

Known overlap risk this is meant to catch: UniMER-1M's handwritten subset reuses CROHME and HME100K images. Both are training sources here, so it can only matter if a UniMER doc lands in the eval split of `hme100k_latex` or vice versa. That isn't a boundary under the rules above (different tasks), and it's reported for information.

## Outputs

On the volume `minicpm-t2l-artifacts` under `task_pool/`: `images/`, `docs/`, `tasks/<id>/{task.json, eval.jsonl, train_candidates.jsonl, train.jsonl}`, `manifest.json`, `dedup_report.json`, `build_logs/`, and a `FINALIZED` marker written last.

Locally in `studies/task_pool/`: `manifest.json`, `dedup_report.json`, `task_cards.json`, `heldout_<date>.json`, `results.md` and `logs/`.

## Run order and cost

| Step | Entrypoint | Hardware | USD (estimate) |
| --- | --- | --- | ---: |
| Smoke: SROIE, PubTabNet capped at 300 rows, 2 templates × 3 docs, into `task_pool_smoke/` | `smoke` | CPU | < 0.3 |
| Build: 16 public sources (one container each) + 384 templates (64 containers of 6) | `build` | CPU, ≤ 80 containers | 1–4 |
| Duplicate checks and final splits | `finalize` | CPU, 4 cores | < 0.5 |

Estimates assume Modal's CPU and memory prices (not verified against the bill). The stage cap is USD 40. Every entrypoint reserves through `t2l.budget` (stage `task_pool`); actual cost is reconciled from `modal billing report`. A failed source or chunk is re-run alone with `--sources` or `--templates`, and every attempt gets a row in `experiments/LOG.md`.

## Amendments

(none yet)
