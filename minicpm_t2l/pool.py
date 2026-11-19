"""Task pool: public sources split into narrow tasks, plus synthetic templates.

Frozen in studies/task_pool/protocol.md; constants here must match it.

Layout under the pool root (on the Modal volume: /vol/task_pool):

    images/<source>/<doc_id>.<ext>          one copy per document, shared by tasks
    docs/<source>.jsonl                     one row per saved doc: id, split, image,
                                            pixel_sha256, dhash, text (for duplicate checks)
    tasks/<task_id>/task.json               spec, prompt, source + licence, held-out flag
    tasks/<task_id>/train_candidates.jsonl  ordered; final train = first TRAIN_CAP kept after dedup
    tasks/<task_id>/eval.jsonl              EVAL_N docs
    tasks/<task_id>/train.jsonl             written by `finalize` (after the duplicate check)

Rows are {"id", "image": <path relative to the pool root>, "label": {...}}.

Splits never touch an official test split. Eval docs come from the official
validation split when there is one, otherwise from a seeded carve of train.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Iterator

from minicpm_t2l import tasks as g0

POOL_SEED = 20261120
EVAL_N = 50
TRAIN_CAP = 200
TRAIN_RESERVE = 40          # extra train candidates so dedup removals can be refilled
MIN_TRAIN = 100             # a task with fewer final train docs is dropped
MIN_EVAL_TRAINING_TASK = 40  # a training task whose eval falls below this after dedup is dropped
CARVE_POOL = 500            # source-level eval pool when a new source has no validation split


@dataclass(frozen=True)
class PoolTask:
    task_id: str
    source: str
    family: str            # receipt | form | table | chart | formula | invoice | synthetic_form
    lang: str
    instruction: str
    schema: str
    extract: str           # key into EXTRACTORS; returns None when a doc doesn't fit the task

    def prompt(self) -> str:
        return (
            f"{self.instruction}\n"
            f"Return only JSON with this shape: {self.schema}\n"
            "Copy text exactly as printed. Use an empty string or empty list when a field is absent."
        )


@dataclass(frozen=True)
class PoolSource:
    key: str
    family: str
    train_split: str
    eval_split: str | None
    licence: str
    commercial: str        # "yes" | "no" | "unclear"
    heldout: bool
    seed: int = POOL_SEED
    carve: int = CARVE_POOL
    scan_train: int | None = None   # read only the first N rows of the split (streamed)
    scan_eval: int | None = None
    hf: str | None = None
    hf_config: str | None = None
    url: str | None = None


# --------------------------------------------------------------------------- extractors (new)

def cord_menu_names(gt: dict) -> dict:
    return {"names": [i["nm"] for i in g0.cord_menu(gt)["menu"] if i.get("nm")]}


def cord_total_price(gt: dict) -> dict:
    return {"total_price": g0.cord_total(gt).get("total_price", "")}


def wr_total(gt: dict) -> dict:
    return {"Total_value": gt.get("Total_value", "") or ""}


def wr_tel(gt: dict) -> dict:
    return {"Tel_value": gt.get("Tel_value", "") or ""}


def form_questions(entities: list[dict]) -> dict:
    return {"questions": [e["text"].strip() for e in entities if e.get("label") == "question" and e.get("text", "").strip()]}


class _TableParser(HTMLParser):
    """HTML table -> rows of cell texts (colspan cells counted once in `rows`, fully in `width`)."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.widths: list[int] = []
        self._cell: list[str] | None = None
        self._span = 1
        self._tight = 0

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
            self.widths.append(0)
        elif tag in ("td", "th"):
            self._cell = []
            span = dict(attrs).get("colspan") or "1"
            self._span = int(span) if str(span).isdigit() else 1
        elif tag in ("sup", "sub"):
            self._tight += 1

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
                self.widths.append(0)
            self.rows[-1].append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self.widths[-1] += self._span
            self._cell = None
        elif tag in ("sup", "sub"):
            self._tight = max(0, self._tight - 1)

    def handle_data(self, data):
        if self._cell is None:
            return
        piece = data.strip()
        if not piece:
            return
        if self._cell and not self._tight:
            self._cell.append(" ")
        self._cell.append(piece)


def parse_table(html: str) -> tuple[list[list[str]], int]:
    p = _TableParser()
    p.feed(html)
    rows = [r for r in p.rows if r]
    width = max(p.widths) if p.widths else 0
    return rows, width


def tab_header(t: dict):
    rows, _ = parse_table(t["html"])
    head = [c for c in rows[0] if c] if rows else []
    return {"header": head} if head else None


def tab_first_col(t: dict):
    rows, _ = parse_table(t["html"])
    col = [r[0] for r in rows[1:] if r and r[0]]
    return {"first_column": col} if col else None


def tab_shape(t: dict):
    rows, width = parse_table(t["html"])
    return {"rows": str(len(rows)), "columns": str(width)} if rows else None


def tab_last_row(t: dict):
    rows, _ = parse_table(t["html"])
    last = [c for c in rows[-1] if c] if len(rows) > 1 else []
    return {"last_row": last} if last else None


def tab_small_full(t: dict):
    rows, width = parse_table(t["html"])
    if not rows or len(rows) > 8 or width > 6 or sum(len(r) for r in rows) > 40:
        return None
    return {"rows": rows}


def chart_table(src: str | None) -> Callable[[dict], dict | None]:
    def f(c: dict):
        if src is not None and c["chart_source"] != src:
            return None
        rows = c["table"]
        if len(rows) < 2:
            return None
        return {"columns": rows[0], "rows": rows[1:]}
    f.__name__ = f"chart_table_{src or 'all'}"
    return f


def chart_columns(c: dict):
    rows = c["table"]
    return {"columns": rows[0]} if rows else None


DVQA_QUESTIONS = {
    "dvqa_largest_bar": "Which bar has the largest value?",
    "dvqa_smallest_bar": "Which bar has the smallest value?",
    "dvqa_largest_value": "What is the value of the largest bar?",
    "dvqa_smallest_value": "What is the value of the smallest bar?",
    "dvqa_most_preferred": "Which object is the most preferred?",
}


def dvqa_answer(question: str) -> Callable[[dict], dict | None]:
    def f(rec: dict):
        for qa in rec["qa"]:
            if qa.get("question") == question:
                ans = str(qa.get("answer", "")).strip().rstrip(".")
                return {"answer": ans} if ans else None
        return None
    f.__name__ = f"dvqa_{question[:20]}"
    return f


def latex_len(lo: int, hi: int) -> Callable[[dict], dict | None]:
    def f(rec: dict):
        s = rec["latex"].strip()
        return {"latex": s} if lo <= len(s) < hi else None
    f.__name__ = f"latex_{lo}_{hi}"
    return f


def inv_header(gt: dict):
    h = gt.get("header") or {}
    keys = ("invoice_no", "invoice_date", "seller", "client", "seller_tax_id", "client_tax_id", "iban")
    return {k: h.get(k, "") or "" for k in keys}


def inv_parties(gt: dict):
    h = gt.get("header") or {}
    return {"seller": h.get("seller", "") or "", "client": h.get("client", "") or ""}


def inv_items(gt: dict):
    keys = ("item_desc", "item_qty", "item_net_price", "item_net_worth", "item_vat", "item_gross_worth")
    items = [{k: it.get(k, "") or "" for k in keys} for it in g0._as_list(gt.get("items")) if isinstance(it, dict)]
    return {"items": items} if items else None


def inv_summary(gt: dict):
    s = gt.get("summary") or {}
    return {k: s.get(k, "") or "" for k in ("total_net_worth", "total_vat", "total_gross_worth")}


EXTRACTORS: dict[str, Callable] = {
    **g0.EXTRACTORS,
    "cord_menu_names": cord_menu_names, "cord_total_price": cord_total_price, "wr_total": wr_total, "wr_tel": wr_tel,
    "form_questions": form_questions,
    "tab_header": tab_header, "tab_first_col": tab_first_col, "tab_shape": tab_shape, "tab_last_row": tab_last_row,
    "tab_small_full": tab_small_full,
    "chart_columns": chart_columns, **{f"chart_table_{s}": chart_table(s) for s in ("statista", "owid", "pew", "oecd")},
    **{k: dvqa_answer(q) for k, q in DVQA_QUESTIONS.items()},
    "latex_short": latex_len(1, 60), "latex_long": latex_len(60, 240), "latex_any": latex_len(1, 240),
    "inv_header": inv_header, "inv_parties": inv_parties, "inv_items": inv_items, "inv_summary": inv_summary,
}


# --------------------------------------------------------------------------- sources

_G0 = {k: v for k, v in g0.SOURCES.items()}
_G0_FAMILY = {"cord": "receipt", "sroie": "receipt", "wildreceipt": "receipt", "funsd": "form"}

SOURCES: dict[str, PoolSource] = {s.key: s for s in [
    # Gate 0 sources: training only (they were used for tuning). Same seed and carve as Gate 0.
    *[PoolSource(k, _G0_FAMILY.get(k, "form"), v["train"], v["eval"], v["licence"],
                 "no" if k == "funsd" or k.startswith("xfund") else "unclear" if k in ("sroie", "wildreceipt") else "yes",
                 heldout=False, seed=g0.SEED, carve=g0.EVAL_N, hf=v.get("hf"), url=v.get("github"))
      for k, v in _G0.items()],
    PoolSource("pubtabnet", "table", "train", "validation", "CDLA-Permissive-1.0", "yes", heldout=False,
               scan_train=4000, scan_eval=1500, hf="apoidea/pubtabnet-html"),
    PoolSource("dvqa", "chart", "train", None, "CC BY-NC 4.0 (DVQA authors); Hub copy has no card", "no", heldout=False,
               scan_train=6000, hf="vikhyatk/dvqa"),
    PoolSource("unimer", "formula", "train", None, "Apache-2.0 (UniMER-1M)", "yes", heldout=False,
               scan_train=8000, hf="deepcopy/UniMER"),
    PoolSource("hme100k", "formula", "train", None, "unclear (TAL HME100K, research use); Hub copy has no card", "unclear",
               heldout=False, scan_train=5000, hf="ConstantHao/HME100K"),
    # Held-out public sources (whole datasets, never trained or tuned on).
    PoolSource("fintabnet", "table", "train", "validation", "CDLA-Permissive-1.0 (FinTabNet); Hub copy has no card", "yes",
               heldout=True, scan_train=4000, scan_eval=1500, hf="apoidea/fintabnet-html", hf_config="en"),
    PoolSource("chartqa", "chart", "train", "val", "GPL-3.0 (ChartQA repo); tables from nmayorga7/chartqa-tables (MIT card)", "unclear",
               heldout=True, hf="ahmed-masry/ChartQA"),
    PoolSource("invoices", "invoice", "train", "validation", "MIT (Hub card)", "yes", heldout=True,
               hf="katanaml-org/invoices-donut-data-v1"),
]}

_QA = g0._QA
_TAB_INSTR = {
    "header": ("List the column headers in the first row of this table.", '{"header": [str]}', "tab_header"),
    "first_col": ("List the cells of the first column below the header row.", '{"first_column": [str]}', "tab_first_col"),
    "shape": ("Count the rows and columns of this table.", '{"rows": str, "columns": str}', "tab_shape"),
    "last_row": ("List the cells of the last row of this table.", '{"last_row": [str]}', "tab_last_row"),
    "full": ("Transcribe this table row by row.", '{"rows": [[str]]}', "tab_small_full"),
}
_CHART_TAB = ("Extract the data table behind this chart: column names, then one row per category.",
              '{"columns": [str], "rows": [[str]]}')
_LANGS = {"zh": "zh", "ja": "ja", "es": "es", "fr": "fr", "de": "de"}

PUBLIC_TASKS: dict[str, PoolTask] = {t.task_id: t for t in [
    # Gate 0's 20 tasks, unchanged.
    *[PoolTask(t.task_id, t.source, _G0_FAMILY.get(t.source, "form"), t.source.split("_")[1] if t.source.startswith("xfund") else "en",
               t.instruction, t.schema, t.extract) for t in g0.TASKS.values()],
    # More narrow tasks on the Gate 0 sources (training only).
    PoolTask("cord_menu_names", "cord", "receipt", "en", "List the name of every menu item on this receipt.", '{"names": [str]}', "cord_menu_names"),
    PoolTask("cord_total_price", "cord", "receipt", "en", "Extract the total price of this receipt.", '{"total_price": str}', "cord_total_price"),
    PoolTask("wr_total", "wildreceipt", "receipt", "en", "Extract the total amount.", '{"Total_value": str}', "wr_total"),
    PoolTask("wr_tel", "wildreceipt", "receipt", "en", "Extract the store telephone number.", '{"Tel_value": str}', "wr_tel"),
    PoolTask("funsd_questions", "funsd", "form", "en", "List every question (key) printed on this form.", '{"questions": [str]}', "form_questions"),
    *[PoolTask(f"xfund_{lang}_headers", f"xfund_{lang}", "form", lang, "List every section header on this form.", '{"headers": [str]}', "form_headers")
      for lang in _LANGS],
    *[PoolTask(f"xfund_{lang}_questions", f"xfund_{lang}", "form", lang, "List every question (key) printed on this form.", '{"questions": [str]}', "form_questions")
      for lang in _LANGS],
    # Tables.
    *[PoolTask(f"{src}_{name}", src, "table", "en", ins, sch, ex)
      for src in ("pubtabnet", "fintabnet") for name, (ins, sch, ex) in _TAB_INSTR.items()],
    # Charts.
    *[PoolTask(k, "dvqa", "chart", "en", f"Answer this question about the chart: {q}", '{"answer": str}', k) for k, q in DVQA_QUESTIONS.items()],
    *[PoolTask(f"chartqa_table_{s}", "chartqa", "chart", "en", _CHART_TAB[0], _CHART_TAB[1], f"chart_table_{s}")
      for s in ("statista", "owid", "pew", "oecd")],
    PoolTask("chartqa_columns", "chartqa", "chart", "en", "List the column names of the data table behind this chart.", '{"columns": [str]}', "chart_columns"),
    # Formulas.
    PoolTask("unimer_latex_short", "unimer", "formula", "en", "Transcribe this formula as LaTeX, with tokens separated by spaces.", '{"latex": str}', "latex_short"),
    PoolTask("unimer_latex_long", "unimer", "formula", "en", "Transcribe this formula as LaTeX, with tokens separated by spaces.", '{"latex": str}', "latex_long"),
    PoolTask("hme100k_latex", "hme100k", "formula", "zh", "Transcribe this handwritten formula as LaTeX, with tokens separated by spaces.", '{"latex": str}', "latex_any"),
    # Invoices (held out).
    PoolTask("inv_header", "invoices", "invoice", "en", "Extract the invoice header fields.",
             '{"invoice_no": str, "invoice_date": str, "seller": str, "client": str, "seller_tax_id": str, "client_tax_id": str, "iban": str}', "inv_header"),
    PoolTask("inv_parties", "invoices", "invoice", "en", "Extract the seller and the client, each with their address.", '{"seller": str, "client": str}', "inv_parties"),
    PoolTask("inv_items", "invoices", "invoice", "en", "List every line item of this invoice.",
             '{"items": [{"item_desc": str, "item_qty": str, "item_net_price": str, "item_net_worth": str, "item_vat": str, "item_gross_worth": str}]}', "inv_items"),
    PoolTask("inv_summary", "invoices", "invoice", "en", "Extract the invoice totals.", '{"total_net_worth": str, "total_vat": str, "total_gross_worth": str}', "inv_summary"),
]}

for _t in PUBLIC_TASKS.values():
    assert _t.source in SOURCES and _t.extract in EXTRACTORS, _t


def tasks_for(source: str) -> list[PoolTask]:
    return [t for t in PUBLIC_TASKS.values() if t.source == source]


# --------------------------------------------------------------------------- loading

@dataclass
class RawDoc:
    doc_id: str
    image: object          # PIL image, or bytes of an encoded image
    raw: object            # what the extractors read


def _image_bytes(img) -> bytes | None:
    if isinstance(img, dict):
        return img.get("bytes")
    return None


def _hf_rows(src: PoolSource, split: str, n: int | None) -> Iterator[dict]:
    import datasets

    kw = {"name": src.hf_config} if src.hf_config else {}
    stream = n is not None
    ds = datasets.load_dataset(src.hf, split=split, streaming=stream, **kw)
    if "image" in (ds.features or {}):
        ds = ds.cast_column("image", datasets.Image(decode=False))
    if stream:
        ds = ds.take(n)
    for row in ds:
        if isinstance(row.get("image"), dict):
            row = {**row, "image": _image_bytes(row["image"])}
        yield row


def _chart_source(imgname: str) -> str:
    stem = Path(imgname).stem
    if stem.startswith(("two_col_", "multi_col_")):
        return "statista"
    if stem.isdigit() and len(stem) >= 10:
        return "owid"
    if stem.isdigit():
        return "pew"
    if "oecd" in stem.lower():
        return "oecd"
    return "other"


def _csv_rows(text: str) -> list[list[str]]:
    import csv

    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]


def load_docs(src: PoolSource, split: str, cache: Path) -> Iterator[RawDoc]:
    """Yield the docs of one split of a source (first `scan_*` rows when capped)."""
    key = src.key
    if key in g0.SOURCES:
        for doc_id, img, label in g0.load_source(key, split, cache):
            yield RawDoc(doc_id, img, label)
        return
    n = src.scan_train if split == src.train_split else src.scan_eval
    if key in ("pubtabnet", "fintabnet"):
        for i, row in enumerate(_hf_rows(src, split, n)):
            yield RawDoc(f"{key}_{split}_{row.get('imgid', i)}", row["image"], {"html": row["html_table"]})
    elif key == "dvqa":
        for i, row in enumerate(_hf_rows(src, split, n)):
            qa = row["qa"]
            if isinstance(qa, str):
                qa = ast.literal_eval(qa)
            yield RawDoc(f"dvqa_{split}_{i}", row["image"], {"qa": qa})
    elif key in ("unimer", "hme100k"):
        for i, row in enumerate(_hf_rows(src, split, n)):
            rid = row.get("id") or Path(str(row.get("image_path", i)).replace("\\", "/")).stem
            yield RawDoc(f"{key}_{split}_{rid}", row["image"], {"latex": row.get("text") or row.get("label") or ""})
    elif key == "invoices":
        for i, row in enumerate(_hf_rows(src, split, n)):
            yield RawDoc(f"inv_{split}_{i}", row["image"], json.loads(row["ground_truth"])["gt_parse"])
    elif key == "chartqa":
        import datasets

        tsplit = {"train": "train", "val": "validation"}[split]
        tables = {r["imgname"]: r["table_csv"] for r in datasets.load_dataset("nmayorga7/chartqa-tables", split=tsplit)}
        seen = set()
        for row in _hf_rows(src, split, n):
            name = row["imgname"]
            if name in seen or name not in tables:
                continue
            seen.add(name)
            yield RawDoc(f"chartqa_{split}_{Path(name).stem}", row["image"],
                         {"table": _csv_rows(tables[name]), "chart_source": _chart_source(name)})
    else:
        raise KeyError(key)


# --------------------------------------------------------------------------- hashing / text

def pixel_hash_and_dhash(img) -> tuple[str, str, tuple[int, int]]:
    """sha256 of the decoded RGB pixels (+ size) and a 64-bit difference hash."""
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))
    rgb = img.convert("RGB")
    h = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
    g = rgb.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return h, f"{bits:016x}", rgb.size


def text_of(obj) -> str:
    out: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif x not in (None, ""):
            out.append(str(x))

    walk(obj)
    return " ".join(out)


def doc_text_for(source: str, raw) -> str:
    """Document content used by the near-duplicate text check (not shown to models)."""
    if source in ("pubtabnet", "fintabnet"):
        rows, _ = parse_table(raw["html"])
        return text_of(rows)
    if source == "chartqa":
        return text_of(raw["table"])
    if source == "dvqa":
        return text_of([qa.get("answer") for qa in raw["qa"]])
    if source in ("unimer", "hme100k"):
        return raw["latex"]
    if source in ("funsd",) or source.startswith("xfund"):
        return text_of([e.get("text") for e in raw])
    return text_of(raw)


# --------------------------------------------------------------------------- building

def order_ids(train_ids: list[str], eval_ids: list[str] | None, seed: int, carve: int) -> tuple[list[str], list[str]]:
    """Seeded orderings. Without an eval split, the first `carve` shuffled train ids become the eval pool.

    For the Gate 0 sources (seed 20261119, carve 50) the first 50 eval / first 200 train ids
    are exactly Gate 0's split.
    """
    rng = random.Random(seed)
    tr = sorted(train_ids)
    rng.shuffle(tr)
    if eval_ids is None:
        return tr[carve:], tr[:carve]
    ev = sorted(eval_ids)
    rng.shuffle(ev)
    return tr, ev


def select_for_task(order: list[str], labels: dict[str, dict | None], n: int) -> list[str]:
    return [d for d in order if labels.get(d) is not None][:n]


def _save_image(img, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path)
    else:
        path.write_bytes(img)


def _ext(img) -> str:
    from PIL import Image

    if isinstance(img, Image.Image):
        return "png"
    head = img[:4]
    return "jpg" if head[:2] == b"\xff\xd8" else "png" if head == b"\x89PNG" else "img"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def task_card(t: PoolTask, src: PoolSource, **extra) -> dict:
    return {**asdict(t), "prompt": t.prompt(), "heldout": src.heldout, "licence": src.licence,
            "commercial": src.commercial, "source_info": asdict(src), **extra}


def build_public_source(key: str, root: Path, cache: Path, log: Callable[[str], None] = print) -> dict:
    """Read one source once and write its docs, images and task candidate files."""
    src = SOURCES[key]
    specs = tasks_for(key)
    raw: dict[str, RawDoc] = {}
    train_ids: list[str] = []
    eval_ids: list[str] | None = [] if src.eval_split else None
    for split, bucket in ((src.train_split, train_ids), (src.eval_split, eval_ids)):
        if split is None:
            continue
        for d in load_docs(src, split, cache):
            if d.doc_id in raw:
                continue
            raw[d.doc_id] = d
            bucket.append(d.doc_id)
        log(f"{key}/{split}: {len(bucket)} docs read")
    tr_order, ev_order = order_ids(train_ids, eval_ids, src.seed, src.carve)
    split_of = {**{d: "train" for d in tr_order}, **{d: "eval" for d in ev_order}}
    used: set[str] = set()
    summary = {}
    for spec in specs:
        fn = EXTRACTORS[spec.extract]
        labels = {}
        for d in tr_order + ev_order:
            try:
                labels[d] = fn(raw[d].raw)
            except (KeyError, TypeError, ValueError, IndexError):
                labels[d] = None
        ev = select_for_task(ev_order, labels, EVAL_N)
        tr = select_for_task(tr_order, labels, TRAIN_CAP + TRAIN_RESERVE)
        tdir = root / "tasks" / spec.task_id
        rows = lambda ids: [{"id": d, "image": f"images/{key}/{d}.{_ext(raw[d].image)}", "label": labels[d]} for d in ids]
        write_jsonl(tdir / "eval.jsonl", rows(ev))
        write_jsonl(tdir / "train_candidates.jsonl", rows(tr))
        (tdir / "task.json").write_text(json.dumps(task_card(spec, src, n_eval_candidates=len(ev), n_train_candidates=len(tr)),
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
        used.update(ev)
        used.update(tr)
        summary[spec.task_id] = {"eval": len(ev), "train_candidates": len(tr)}
        log(f"  {spec.task_id}: eval {len(ev)}, train candidates {len(tr)}")
    docs = []
    for d in sorted(used):
        img = raw[d].image
        rel = f"images/{key}/{d}.{_ext(img)}"
        if not (root / rel).exists():
            _save_image(img, root / rel)
        sha, dh, size = pixel_hash_and_dhash(img)
        docs.append({"id": d, "source": key, "split": split_of[d], "image": rel, "pixel_sha256": sha, "dhash": dh,
                     "size": list(size), "text": doc_text_for(key, raw[d].raw)})
    write_jsonl(root / "docs" / f"{key}.jsonl", docs)
    return {"source": key, "docs_read": len(raw), "docs_saved": len(docs), "tasks": summary}


def build_synth_templates(template_ids: list[str], root: Path, log: Callable[[str], None] = print) -> dict:
    """Render every doc of the given templates (eval docs 0..49, train candidates after)."""
    from minicpm_t2l import synth

    by_id = {t.template_id: t for t in synth.make_templates()}
    held = set(synth.heldout_domains())
    out = {}
    for tid in template_ids:
        t = by_id[tid]
        tdir = root / "tasks" / tid
        rows, docs = {"eval": [], "train": []}, []
        for i in range(EVAL_N + TRAIN_CAP + TRAIN_RESERVE):
            split = "eval" if i < EVAL_N else "train"
            img, label, _ = synth.render_doc(t, i)
            did = f"{tid}_{i:04d}"
            rel = f"images/synth/{tid}/{did}.jpg"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=synth.jpeg_quality(t, i))
            p.write_bytes(buf.getvalue())
            sha, dh, size = pixel_hash_and_dhash(buf.getvalue())
            rows[split].append({"id": did, "image": rel, "label": label})
            docs.append({"id": did, "source": "synth", "split": split, "image": rel, "pixel_sha256": sha, "dhash": dh,
                         "size": list(size), "text": synth.doc_text(label)})
        write_jsonl(tdir / "eval.jsonl", rows["eval"])
        write_jsonl(tdir / "train_candidates.jsonl", rows["train"])
        card = synth.template_card(t)
        card.update(prompt=PoolTask(tid, "synth", "synthetic_form", t.lang, synth.INSTRUCTION, t.schema(), "").prompt(),
                    heldout=t.domain in held, licence="generated by this project (values from Faker, MIT)",
                    commercial="yes", n_eval_candidates=len(rows["eval"]), n_train_candidates=len(rows["train"]))
        (tdir / "task.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        write_jsonl(root / "docs" / "synth" / f"{tid}.jsonl", docs)
        out[tid] = {"eval": len(rows["eval"]), "train_candidates": len(rows["train"])}
        log(f"{tid}: done")
    return out
