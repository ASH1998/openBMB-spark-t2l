"""Gate 0 task definitions and builders.

A task is one document source plus one output schema. Each builder turns a
source record into {"id", "image" (PIL), "label" (dict)}; `build_task` writes a
task directory:

    <out>/<task_id>/
        task.json        # spec, prompt, split sizes, seed, source + licence
        train.jsonl      # {"id", "image": "images/<id>.png", "label": {...}}
        eval.jsonl
        images/

Splits never use an official test split: those stay untouched for later gates.
Eval docs come from the official validation split when there is one, otherwise
they are carved from train with a fixed seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

SEED = 20261119
EVAL_N = 50
TRAIN_CAP = 200


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    source: str            # key into SOURCES
    instruction: str       # what to extract; the schema line is appended
    schema: str            # JSON shape shown to the model
    extract: str           # key into EXTRACTORS

    def prompt(self) -> str:
        return (
            f"{self.instruction}\n"
            f"Return only JSON with this shape: {self.schema}\n"
            "Copy text exactly as printed. Use an empty string or empty list when a field is absent."
        )


# --------------------------------------------------------------------------
# Label extractors: source record (already decoded to a dict label) -> task label
# --------------------------------------------------------------------------

def _as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _pick(d: dict, keys: Iterable[str]) -> dict:
    return {k: d[k] for k in keys if k in d and d[k] not in (None, "", [], {})}


def cord_menu(gt: dict) -> dict:
    items = []
    for it in _as_list(gt.get("menu")):
        if isinstance(it, dict):
            items.append(_pick(it, ("nm", "cnt", "price")))
    return {"menu": [i for i in items if i]}


def cord_subtotal(gt: dict) -> dict:
    st = gt.get("sub_total") or {}
    return _pick(st, ("subtotal_price", "discount_price", "service_price", "tax_price", "etc")) if isinstance(st, dict) else {}


def cord_total(gt: dict) -> dict:
    t = gt.get("total") or {}
    return _pick(t, ("total_price", "cashprice", "changeprice", "creditcardprice")) if isinstance(t, dict) else {}


def cord_full(gt: dict) -> dict:
    return {"menu": cord_menu(gt)["menu"], "sub_total": cord_subtotal(gt), "total": cord_total(gt)}


def sroie_field(name: str) -> Callable[[dict], dict]:
    def f(ent: dict) -> dict:
        return {name: ent.get(name, "") or ""}
    f.__name__ = f"sroie_{name}"
    return f


def sroie_all(ent: dict) -> dict:
    return {k: ent.get(k, "") or "" for k in ("company", "date", "address", "total")}


def wr_store(gt: dict) -> dict:
    return _pick(gt, ("Store_name_value", "Store_addr_value", "Tel_value"))


def wr_datetime(gt: dict) -> dict:
    return _pick(gt, ("Date_value", "Time_value"))


def wr_products(gt: dict) -> dict:
    items = []
    for it in _as_list(gt.get("Prod_list")):
        if isinstance(it, dict):
            items.append(_pick(it, ("Prod_item_value", "Prod_quantity_value", "Prod_price_value")))
    return {"Prod_list": [i for i in items if i]}


def wr_money(gt: dict) -> dict:
    out = _pick(gt, ("Subtotal_value", "Tips_value", "Total_value"))
    taxes = [t.get("Tax_value") for t in _as_list(gt.get("Tax_list")) if isinstance(t, dict) and t.get("Tax_value")]
    if taxes:
        out["Tax_list"] = [{"Tax_value": t} for t in taxes]
    return out


def form_qa_pairs(entities: list[dict]) -> dict:
    """FUNSD/XFUND-style entities with `linking` -> question/answer pairs."""
    by_id = {e["id"]: e for e in entities}
    pairs = []
    for e in entities:
        if e.get("label") != "question":
            continue
        answers = []
        for link in e.get("linking") or []:
            a, b = link
            other = b if a == e["id"] else a
            o = by_id.get(other)
            if o is not None and o.get("label") == "answer" and o.get("text", "").strip():
                answers.append((o.get("box", [0, 0])[1], o.get("box", [0, 0])[0], o["text"].strip()))
        if e.get("text", "").strip() and answers:
            answers.sort()
            pairs.append({"question": e["text"].strip(), "answer": " ".join(t for *_, t in answers)})
    return {"pairs": pairs}


def form_headers(entities: list[dict]) -> dict:
    return {"headers": [e["text"].strip() for e in entities if e.get("label") == "header" and e.get("text", "").strip()]}


EXTRACTORS: dict[str, Callable[[dict], dict]] = {
    "cord_menu": cord_menu, "cord_subtotal": cord_subtotal, "cord_total": cord_total, "cord_full": cord_full,
    "sroie_company": sroie_field("company"), "sroie_date": sroie_field("date"),
    "sroie_address": sroie_field("address"), "sroie_total": sroie_field("total"), "sroie_all": sroie_all,
    "wr_store": wr_store, "wr_datetime": wr_datetime, "wr_products": wr_products, "wr_money": wr_money,
    "form_qa_pairs": form_qa_pairs, "form_headers": form_headers,
}


# --------------------------------------------------------------------------
# Task registry (Gate 0: 20 tasks)
# --------------------------------------------------------------------------

_QA = '{"pairs": [{"question": str, "answer": str}]}'

TASKS: dict[str, TaskSpec] = {t.task_id: t for t in [
    TaskSpec("cord_menu", "cord", "List every menu item on this receipt with its name (nm), count (cnt) and price.",
             '{"menu": [{"nm": str, "cnt": str, "price": str}]}', "cord_menu"),
    TaskSpec("cord_subtotal", "cord", "Extract the subtotal block of this receipt.",
             '{"subtotal_price": str, "discount_price": str, "service_price": str, "tax_price": str, "etc": str}', "cord_subtotal"),
    TaskSpec("cord_total", "cord", "Extract the total block of this receipt.",
             '{"total_price": str, "cashprice": str, "changeprice": str, "creditcardprice": str}', "cord_total"),
    TaskSpec("cord_full", "cord", "Parse this receipt into menu items, subtotal block and total block.",
             '{"menu": [{"nm": str, "cnt": str, "price": str}], "sub_total": {...}, "total": {...}}', "cord_full"),
    TaskSpec("sroie_company", "sroie", "Extract the company name from this receipt.", '{"company": str}', "sroie_company"),
    TaskSpec("sroie_date", "sroie", "Extract the receipt date as printed.", '{"date": str}', "sroie_date"),
    TaskSpec("sroie_address", "sroie", "Extract the company address from this receipt.", '{"address": str}', "sroie_address"),
    TaskSpec("sroie_total", "sroie", "Extract the total amount paid on this receipt.", '{"total": str}', "sroie_total"),
    TaskSpec("sroie_all", "sroie", "Extract company, date, address and total from this receipt.",
             '{"company": str, "date": str, "address": str, "total": str}', "sroie_all"),
    TaskSpec("wr_store", "wildreceipt", "Extract the store name, store address and telephone number.",
             '{"Store_name_value": str, "Store_addr_value": str, "Tel_value": str}', "wr_store"),
    TaskSpec("wr_datetime", "wildreceipt", "Extract the purchase date and time.",
             '{"Date_value": str, "Time_value": str}', "wr_datetime"),
    TaskSpec("wr_products", "wildreceipt", "List every purchased product with item name, quantity and price.",
             '{"Prod_list": [{"Prod_item_value": str, "Prod_quantity_value": str, "Prod_price_value": str}]}', "wr_products"),
    TaskSpec("wr_money", "wildreceipt", "Extract subtotal, taxes, tips and total.",
             '{"Subtotal_value": str, "Tax_list": [{"Tax_value": str}], "Tips_value": str, "Total_value": str}', "wr_money"),
    TaskSpec("funsd_qa", "funsd", "List every question/answer (key/value) pair on this form.", _QA, "form_qa_pairs"),
    TaskSpec("funsd_headers", "funsd", "List every section header on this form.", '{"headers": [str]}', "form_headers"),
    *[TaskSpec(f"xfund_{lang}_qa", f"xfund_{lang}", "List every question/answer (key/value) pair on this form.", _QA, "form_qa_pairs")
      for lang in ("zh", "ja", "es", "fr", "de")],
]}

assert len(TASKS) == 20, len(TASKS)

# Source facts checked 2026-11-19 (HF API / GitHub API). Licences as stated by the
# uploader; see studies/gate0/protocol.md for the caveats.
SOURCES: dict[str, dict] = {
    "cord": {"hf": "naver-clova-ix/cord-v2", "train": "train", "eval": "validation", "licence": "CC-BY-4.0"},
    "sroie": {"hf": "jsdnrs/ICDAR2019-SROIE", "train": "train", "eval": None,
              "licence": "CC-BY-4.0 per uploader; original ICDAR 2019 release has no clear licence"},
    "wildreceipt": {"hf": "podbilabs/wildreceipt-donut", "train": "train", "eval": "validation",
                    "licence": "unstated on Hub; MMOCR dataset, licence unclear"},
    "funsd": {"hf": "davidle7/funsd-json", "train": "train", "eval": None,
              "licence": "FUNSD: non-commercial research/education (Hub card says MIT; doubtful)"},
    **{f"xfund_{lang}": {"github": f"https://github.com/doc-analysis/XFUND/releases/download/v1.0/{lang}",
                         "train": "train", "eval": "val", "licence": "CC BY-NC-SA 4.0 (XFUND repo)"}
       for lang in ("zh", "ja", "es", "fr", "de")},
}


# --------------------------------------------------------------------------
# Source loaders -> iterables of (doc_id, PIL image, raw label dict)
# --------------------------------------------------------------------------

def _load_hf(source: str, split: str):
    from datasets import load_dataset

    ds = load_dataset(SOURCES[source]["hf"], split=split)
    for i, row in enumerate(ds):
        if source == "cord":
            yield f"cord_{split}_{i}", row["image"], json.loads(row["ground_truth"])["gt_parse"]
        elif source == "sroie":
            yield f"sroie_{row.get('key', i)}", row["image"], dict(row["entities"])
        elif source == "wildreceipt":
            yield f"wr_{split}_{i}", row["image"], json.loads(row["ground_truth"])["gt_parse"]
        elif source == "funsd":
            yield f"funsd_{split}_{i}", row["image"], json.loads(row["text_output"])["form"]
        else:
            raise KeyError(source)


def _load_xfund(source: str, split: str, cache: Path):
    import urllib.request
    import zipfile

    from PIL import Image

    base = SOURCES[source]["github"]
    cache.mkdir(parents=True, exist_ok=True)
    jpath, zpath = cache / f"{source}.{split}.json", cache / f"{source}.{split}.zip"
    for url, path in ((f"{base}.{split}.json", jpath), (f"{base}.{split}.zip", zpath)):
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    docs = json.loads(jpath.read_text(encoding="utf-8"))["documents"]
    with zipfile.ZipFile(zpath) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for d in docs:
            with zf.open(names[d["img"]["fname"]]) as fh:
                img = Image.open(fh).convert("RGB")
            yield f"{source}_{d['id']}", img, d["document"]


def load_source(source: str, split: str, cache: Path):
    if source.startswith("xfund_"):
        return _load_xfund(source, split, cache)
    return _load_hf(source, split)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

def split_ids(train_ids: list[str], eval_ids: list[str] | None, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Deterministic train/eval id lists. Without an official eval split, carve EVAL_N from train."""
    rng = random.Random(seed)
    train = sorted(train_ids)
    rng.shuffle(train)
    if eval_ids is None:
        ev, train = train[:EVAL_N], train[EVAL_N:]
    else:
        ev = sorted(eval_ids)
        rng.shuffle(ev)
        ev = ev[:EVAL_N]
    return train[:TRAIN_CAP], ev


def build_source(source: str, out: Path, cache: Path) -> None:
    """Build every task that uses `source` (each source is read once)."""
    src = SOURCES[source]
    specs = [t for t in TASKS.values() if t.source == source]
    raw: dict[str, tuple] = {}
    train_ids, eval_ids = [], ([] if src["eval"] else None)
    for split, bucket in ((src["train"], train_ids), (src["eval"], eval_ids)):
        if split is None:
            continue
        for doc_id, img, label in load_source(source, split, cache):
            raw[doc_id] = (img, label)
            bucket.append(doc_id)
    tr, ev = split_ids(train_ids, eval_ids)
    for spec in specs:
        tdir = out / spec.task_id
        (tdir / "images").mkdir(parents=True, exist_ok=True)
        fn = EXTRACTORS[spec.extract]
        for name, ids in (("train", tr), ("eval", ev)):
            with open(tdir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for doc_id in ids:
                    img, label = raw[doc_id]
                    rel = f"images/{doc_id}.png"
                    if not (tdir / rel).exists():
                        img.convert("RGB").save(tdir / rel)
                    fh.write(json.dumps({"id": doc_id, "image": rel, "label": fn(label)}, ensure_ascii=False) + "\n")
        meta = {**asdict(spec), "prompt": spec.prompt(), "source_info": src, "seed": SEED,
                "n_train": len(tr), "n_eval": len(ev)}
        (tdir / "task.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build Gate 0 task directories")
    ap.add_argument("--out", default="data/gate0")
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--source", action="append", help="source key(s); default all")
    a = ap.parse_args()
    for s in a.source or list(SOURCES):
        print("building", s, flush=True)
        build_source(s, Path(a.out), Path(a.cache))
