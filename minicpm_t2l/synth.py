"""Synthetic form templates with exact labels.

A *template* is one task: a document domain (see `minicpm_t2l.synth_fields`), a
language (en or zh), a subset of the domain's fields that forms the output
schema, a few extra printed fields that are not in the schema, a layout family,
fonts, formats and a scan profile. Every document rendered from a template uses
the same layout and schema with new values.

Labels are exact by construction: values come from Faker and a seeded RNG, the
renderer prints them, and the label is the same strings. `render_doc` returns
the strings it drew so tests can check that every gold value was printed.

Nothing here imports Modal. Fonts are looked up by file name in the system font
folders (or `T2L_FONT_DIRS`, separated by os.pathsep).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from minicpm_t2l.synth_fields import DOMAINS, Domain, FieldDef

SYNTH_SEED = 20261120
LANGS = ("en", "zh")
TEMPLATES_PER_DOMAIN_LANG = 8
FAMILIES = ("kv_lines", "kv_two_col", "boxed_grid", "underline", "table_2col", "card", "receipt", "letterhead")
KEY_STYLES = ("snake", "camel", "label")
SCAN_PROFILES = ("clean", "scan", "rough")
INSTRUCTION = "Extract the listed fields from this document."


def stable_seed(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


# --------------------------------------------------------------------------- fonts

FONT_ROLES: dict[str, tuple[str, ...]] = {
    "sans": ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"),
    "sans_bold": ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"),
    "serif": ("LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "times.ttf"),
    "serif_bold": ("LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "timesbd.ttf"),
    "mono": ("LiberationMono-Regular.ttf", "DejaVuSansMono.ttf", "cour.ttf"),
    "italic": ("LiberationSerif-Italic.ttf", "DejaVuSerif-Italic.ttf", "timesi.ttf"),
    "cjk_sans": ("NotoSansCJK-Regular.ttc", "wqy-zenhei.ttc", "msyh.ttc"),
    "cjk_sans_bold": ("NotoSansCJK-Bold.ttc", "wqy-zenhei.ttc", "msyhbd.ttc"),
    "cjk_serif": ("NotoSerifCJK-Regular.ttc", "uming.ttc", "simsun.ttc"),
    "cjk_kai": ("ukai.ttc", "simkai.ttf", "NotoSerifCJK-Regular.ttc", "simsun.ttc"),
}
TEXT_ROLES = {"en": ("sans", "serif", "mono"), "zh": ("cjk_sans", "cjk_serif", "cjk_kai")}
BOLD_ROLES = {"en": ("sans_bold", "serif_bold"), "zh": ("cjk_sans_bold", "cjk_sans")}
HAND_ROLES = {"en": ("italic",), "zh": ("cjk_kai",)}


def _font_dirs() -> list[Path]:
    env = os.environ.get("T2L_FONT_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [Path(p) for p in ("/usr/share/fonts", "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts")]


@lru_cache(maxsize=1)
def _font_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for d in _font_dirs():
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                idx.setdefault(f.lower(), str(Path(root) / f))
    return idx


def font_path(role: str) -> str:
    """First installed file for a font role. Raises if none is installed."""
    idx = _font_index()
    for name in FONT_ROLES[role]:
        if name.lower() in idx:
            return idx[name.lower()]
    raise FileNotFoundError(f"no font for role {role!r}; looked for {FONT_ROLES[role]} in {_font_dirs()}")


@lru_cache(maxsize=256)
def _font(role: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(font_path(role), size)


# --------------------------------------------------------------------------- templates

@dataclass(frozen=True)
class Template:
    template_id: str
    domain: str
    lang: str
    schema_fields: tuple[str, ...]      # field keys in the output schema, in schema order
    printed_fields: tuple[str, ...]     # schema + distractor fields, in print order
    items: bool                         # line-item table printed and in the schema
    item_cols: tuple[str, ...]
    key_style: str
    family: str
    title: str
    label_role: str
    value_role: str
    title_role: str
    label_size: int
    value_size: int
    title_size: int
    page_w: int
    margin: int
    colon: str
    date_fmt: str
    period_fmt: str
    money_sym: str
    money_suffix: str
    money_commas: bool
    title_align: str
    borders: bool
    accent: tuple[int, int, int]
    scan: str
    blank_prob: float
    issuer: bool
    footer: bool
    max_items: int

    @property
    def domain_def(self) -> Domain:
        return DOMAINS[self.domain]

    def field(self, key: str) -> FieldDef:
        for f in self.domain_def.fields:
            if f.key == key:
                return f
        raise KeyError(key)

    def item_field(self, key: str) -> FieldDef:
        for f in self.domain_def.items:
            if f.key == key:
                return f
        raise KeyError(key)

    def label_text(self, f: FieldDef) -> str:
        return f.zh if self.lang == "zh" else f.en

    def json_key(self, f: FieldDef) -> str:
        return _style_key(f.key, self.label_text(f), self.key_style)

    def items_json_key(self) -> str:
        d = self.domain_def
        return _style_key(d.items_key, d.items_zh if self.lang == "zh" else d.items_en, self.key_style)

    def schema(self) -> str:
        parts = [f'"{self.json_key(self.field(k))}": str' for k in self.schema_fields]
        if self.items:
            cols = ", ".join(f'"{self.json_key(self.item_field(c))}": str' for c in self.item_cols)
            parts.append(f'"{self.items_json_key()}": [{{{cols}}}]')
        return "{" + ", ".join(parts) + "}"

    def to_json(self) -> dict:
        return asdict(self)


def _style_key(key: str, label: str, style: str) -> str:
    if style == "snake":
        return key
    if style == "camel":
        head, *rest = key.split("_")
        return head + "".join(p.capitalize() for p in rest)
    return re.sub(r"\s+", " ", label.replace(".", "").strip())


_DATE_EN = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%d.%m.%Y")
_DATE_ZH = ("%Y-%m-%d", "%Y年%m月%d日", "%Y/%m/%d", "%Y.%m.%d")
_PERIOD_EN = ("%b %Y", "%Y-%m", "%m/%Y")
_PERIOD_ZH = ("%Y年%m月", "%Y-%m", "%Y.%m")
_MONEY_EN = (("$", ""), ("USD ", ""), ("", ""), ("€", ""), ("£", ""), ("", " USD"))
_MONEY_ZH = (("¥", ""), ("", "元"), ("￥", ""), ("", ""), ("人民币", "元"))
_ACCENTS = ((20, 60, 140), (150, 30, 30), (20, 110, 70), (60, 60, 60), (120, 60, 140), (0, 100, 130))


def sample_template(domain: str, lang: str, k: int, seed: int = SYNTH_SEED) -> Template:
    d = DOMAINS[domain]
    rng = random.Random(stable_seed(seed, domain, lang, k))
    keys = [f.key for f in d.fields]
    n_schema = rng.randint(4, min(9, len(keys)))
    schema = rng.sample(keys, n_schema)
    rest = [x for x in keys if x not in schema]
    distract = rng.sample(rest, min(len(rest), rng.randint(0, 3)))
    printed = schema + distract
    rng.shuffle(printed)
    if rng.random() < 0.7:  # schema in reading order most of the time
        schema = [x for x in printed if x in schema]
    family = rng.choice(FAMILIES)
    has_items = bool(d.items) and (family in ("letterhead", "receipt") or rng.random() < 0.6)
    item_cols = tuple(f.key for f in d.items)
    if has_items and len(item_cols) > 2 and rng.random() < 0.4:
        drop = rng.choice(item_cols[1:])
        item_cols = tuple(c for c in item_cols if c != drop)
    titles = d.title_zh if lang == "zh" else d.title_en
    label_role = rng.choice(BOLD_ROLES[lang] + TEXT_ROLES[lang])
    value_role = rng.choice(HAND_ROLES[lang] if family == "underline" and rng.random() < 0.6 else TEXT_ROLES[lang])
    vsize = rng.randint(17, 24)
    sym, suf = rng.choice(_MONEY_ZH if lang == "zh" else _MONEY_EN)
    return Template(
        template_id=f"syn_{domain}_{lang}_{k:02d}", domain=domain, lang=lang,
        schema_fields=tuple(schema), printed_fields=tuple(printed), items=has_items,
        item_cols=item_cols if has_items else (), key_style=rng.choice(KEY_STYLES), family=family,
        title=rng.choice(titles), label_role=label_role, value_role=value_role, title_role=rng.choice(BOLD_ROLES[lang]),
        label_size=max(14, vsize + rng.randint(-3, 1)), value_size=vsize, title_size=rng.randint(28, 42),
        page_w=rng.randint(560, 660) if family == "receipt" else rng.randint(880, 1000) if family == "card" else rng.randint(1000, 1240),
        margin=rng.randint(36, 70), colon=rng.choice(("：", ":", "") if lang == "zh" else (":", "", " :")),
        date_fmt=rng.choice(_DATE_ZH if lang == "zh" else _DATE_EN), period_fmt=rng.choice(_PERIOD_ZH if lang == "zh" else _PERIOD_EN),
        money_sym=sym, money_suffix=suf, money_commas=rng.random() < 0.6, title_align=rng.choice(("left", "center")),
        borders=rng.random() < 0.6, accent=rng.choice(_ACCENTS), scan=rng.choice(SCAN_PROFILES),
        blank_prob=rng.choice((0.0, 0.0, 0.1)), issuer=rng.random() < 0.5, footer=rng.random() < 0.6,
        max_items=rng.randint(2, 6),
    )


def _signature(t: Template) -> tuple:
    return (t.domain, t.lang, frozenset(t.schema_fields), t.items, t.item_cols, t.key_style)


def make_templates(seed: int = SYNTH_SEED, per_domain_lang: int = TEMPLATES_PER_DOMAIN_LANG) -> list[Template]:
    """All templates, each with a distinct output schema (field set + key style)."""
    out, seen = [], set()
    for domain in sorted(DOMAINS):
        for lang in LANGS:
            k, attempt = 0, 0
            while k < per_domain_lang:
                t = sample_template(domain, lang, k, seed=stable_seed(seed, attempt) if attempt else seed)
                sig = _signature(t)
                if sig in seen:
                    attempt += 1
                    if attempt > 50:
                        raise RuntimeError(f"cannot find a distinct schema for {domain}/{lang}/{k}")
                    continue
                seen.add(sig)
                out.append(t)
                k, attempt = k + 1, 0
    return out


def heldout_domains(seed: int = SYNTH_SEED, n: int = 4) -> list[str]:
    """Whole domains held out for testing, picked by a fixed seed (not by hand)."""
    return sorted(random.Random(seed).sample(sorted(DOMAINS), n))


# --------------------------------------------------------------------------- values

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_PRODUCTS_EN = ("Cable", "Bracket", "Filter", "Notebook", "Valve", "Desk Lamp", "Sensor", "Pump", "Charger", "Mug", "Printer Paper", "Screw Set", "Headset", "Backpack", "Kettle")
_PRODUCTS_ZH = ("数据线", "支架", "滤芯", "笔记本", "阀门", "台灯", "传感器", "水泵", "充电器", "水杯", "打印纸", "螺丝套装", "耳机", "双肩包", "电水壶")
_ADJ_ZH = ("不锈钢", "加厚", "便携式", "无线", "家用", "工业", "高速", "迷你", "多功能", "节能")
_DISHES_EN = ("Caesar Salad", "Beef Burger", "Fish and Chips", "Margherita Pizza", "Tomato Soup", "Grilled Salmon", "Iced Tea", "Espresso", "Fried Rice", "Pad Thai", "Lemonade", "Cheesecake")
_DISHES_ZH = ("宫保鸡丁", "麻婆豆腐", "鱼香肉丝", "清蒸鲈鱼", "红烧肉", "酸辣汤", "扬州炒饭", "小笼包", "回锅肉", "糖醋排骨", "米饭", "酸梅汤")
_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"


def fmt_date(y: int, m: int, d: int, fmt: str) -> str:
    return (fmt.replace("%Y", f"{y:04d}").replace("%m", f"{m:02d}").replace("%d", f"{d:02d}")
            .replace("%b", _MONTHS[m - 1]))


def _code(pattern: str, rng: random.Random) -> str:
    out = []
    for ch in pattern:
        if ch == "#":
            out.append(str(rng.randint(0, 9)))
        elif ch == "?":
            out.append(chr(ord("A") + rng.randint(0, 25)))
        else:
            out.append(ch)
    return "".join(out)


def fmt_money(v: float, t: Template) -> str:
    s = f"{v:,.2f}" if t.money_commas else f"{v:.2f}"
    return f"{t.money_sym}{s}{t.money_suffix}"


def gen_value(f: FieldDef, t: Template, rng: random.Random, fake) -> str:
    """One printed value for a field. `fake` is a Faker for the template's language."""
    zh = t.lang == "zh"
    k, a = f.kind, f.args
    if k == "name":
        v = fake.name()
    elif k == "first_name":
        v = fake.first_name()
    elif k == "last_name":
        v = fake.last_name()
    elif k == "company":
        v = fake.company()
    elif k == "address":
        v = fake.address()
    elif k == "city":
        v = fake.city()
    elif k == "country":
        v = fake.country()
    elif k == "phone":
        v = fake.phone_number()
    elif k == "email":
        v = fake.email()
    elif k == "job":
        v = fake.job()
    elif k == "idcard":
        v = fake.ssn()
    elif k == "date":
        y0, y1 = (1950, 2008) if a == ("dob",) else (2015, 2026)
        v = fmt_date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28), t.date_fmt)
    elif k == "period":
        v = fmt_date(rng.randint(2018, 2026), rng.randint(1, 12), 1, t.period_fmt)
    elif k == "time":
        v = f"{rng.randint(7, 23):02d}:{rng.randint(0, 59):02d}"
    elif k == "money":
        v = fmt_money(round(rng.uniform(a[0], a[1]), 2), t)
    elif k == "int":
        v = str(rng.randint(a[0], a[1]))
    elif k == "dec":
        v = f"{rng.uniform(a[0], a[1]):.{a[2]}f}"
    elif k == "code":
        v = _code(a[0], rng)
    elif k == "choice":
        v = rng.choice(a)[1 if zh else 0]
    elif k == "plate":
        v = (rng.choice(_PROVINCES) + _code("?·#####", rng)) if zh else _code("???-####", rng)
    elif k == "product":
        v = (rng.choice(_ADJ_ZH) + rng.choice(_PRODUCTS_ZH)) if zh else f"{fake.word().title()} {rng.choice(_PRODUCTS_EN)}"
    elif k == "dish":
        v = rng.choice(_DISHES_ZH if zh else _DISHES_EN)
    else:
        raise KeyError(f"unknown value kind {k!r}")
    return re.sub(r"\s*\n\s*", ", " if not zh else "", str(v)).strip()


@lru_cache(maxsize=4)
def _faker(lang: str):
    from faker import Faker

    return Faker("zh_CN" if lang == "zh" else "en_US")


@dataclass
class DocValues:
    fields: dict[str, str]                       # field key -> printed value ("" if blank)
    items: list[dict[str, str]] = field(default_factory=list)
    issuer: str = ""
    footer: str = ""


def sample_values(t: Template, index: int) -> DocValues:
    seed = stable_seed("doc", t.template_id, index)
    rng = random.Random(seed)
    fake = _faker(t.lang)
    fake.seed_instance(seed)
    vals = {}
    for key in t.printed_fields:
        blank = key in t.schema_fields and rng.random() < t.blank_prob
        vals[key] = "" if blank else gen_value(t.field(key), t, rng, fake)
    items = []
    if t.items:
        for _ in range(rng.randint(1, t.max_items)):
            items.append({c: gen_value(t.item_field(c), t, rng, fake) for c in t.item_cols})
    issuer = fake.company() if t.issuer else ""
    footer = ""
    if t.footer:
        footer = fake.sentence(nb_words=6 if t.lang == "en" else 10)
    return DocValues(vals, items, issuer, footer)


def label_for(t: Template, v: DocValues) -> dict:
    out = {t.json_key(t.field(k)): v.fields[k] for k in t.schema_fields}
    if t.items:
        out[t.items_json_key()] = [{t.json_key(t.item_field(c)): row[c] for c in t.item_cols} for row in v.items]
    return out


# --------------------------------------------------------------------------- rendering

class _Canvas:
    def __init__(self, t: Template):
        from PIL import Image, ImageDraw

        self.t = t
        self.img = Image.new("RGB", (t.page_w, 6000), "white")
        self.d = ImageDraw.Draw(self.img)
        self.drawn: list[str] = []
        self.lf = _font(t.label_role, t.label_size)
        self.vf = _font(t.value_role, t.value_size)
        self.small = _font(TEXT_ROLES[t.lang][0], max(13, t.label_size - 4))

    def w(self, s: str, font) -> float:
        return self.d.textlength(s, font=font)

    @staticmethod
    def lh(font) -> int:
        return int(font.size * 1.45)

    def wrap(self, s: str, font, max_w: float) -> list[str]:
        if not s:
            return [""]
        max_w = max(max_w, font.size * 2)
        if " " in s and self.t.lang == "en":
            lines, cur = [], ""
            for word in s.split(" "):
                cand = f"{cur} {word}" if cur else word
                if self.w(cand, font) <= max_w:
                    cur = cand
                    continue
                if cur:
                    lines.append(cur)
                cur = word
                while self.w(cur, font) > max_w and len(cur) > 1:
                    cut = len(cur)
                    while cut > 1 and self.w(cur[:cut], font) > max_w:
                        cut -= 1
                    lines.append(cur[:cut])
                    cur = cur[cut:]
            lines.append(cur)
            return lines
        lines, cur = [], ""
        for ch in s:
            if self.w(cur + ch, font) > max_w and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
        return lines

    def text(self, xy, s: str, font, fill=(0, 0, 0), record: bool = False) -> None:
        if s:
            self.d.text(xy, s, font=font, fill=fill)
        if record and s:
            self.drawn.append(s)

    def block(self, x: float, y: float, s: str, font, max_w: float, fill=(0, 0, 0), record: bool = True,
              align: str = "left") -> int:
        """Wrapped text; returns the new y. Records the full string once."""
        lines = self.wrap(s, font, max_w)
        for ln in lines:
            xx = x + max_w - self.w(ln, font) if align == "right" else x
            self.text((xx, y), ln, font, fill)
            y += self.lh(font)
        if record and s:
            self.drawn.append(s)
        return int(y)


def _label(t: Template, f: FieldDef) -> str:
    return t.label_text(f) + t.colon


def _kv_rows(c: _Canvas, x0: float, x1: float, y: int, pairs: list[tuple[str, str]], aligned: bool, gap: int) -> int:
    lw = max((c.w(lbl, c.lf) for lbl, _ in pairs), default=0) + 16
    for lbl, val in pairs:
        c.text((x0, y), lbl, c.lf)
        vx = x0 + lw if aligned else x0 + c.w(lbl, c.lf) + 10
        y2 = c.block(vx, y, val, c.vf, x1 - vx)
        y = max(y2, y + c.lh(c.lf)) + gap
    return y


def _render_fields(c: _Canvas, t: Template, v: DocValues, y: int, rng: random.Random) -> int:
    m, W = t.margin, t.page_w
    pairs = [(_label(t, t.field(k)), v.fields[k]) for k in t.printed_fields]
    gap = rng.randint(6, 16)
    fam = t.family
    if fam in ("kv_lines", "card"):
        return _kv_rows(c, m + (20 if fam == "card" else 0), W - m, y, pairs, aligned=t.borders, gap=gap)
    if fam in ("kv_two_col", "letterhead"):
        half = (len(pairs) + 1) // 2
        mid = W // 2
        y1 = _kv_rows(c, m, mid - 20, y, pairs[:half], aligned=True, gap=gap)
        y2 = _kv_rows(c, mid + 10, W - m, y, pairs[half:], aligned=True, gap=gap)
        return max(y1, y2)
    if fam == "boxed_grid":
        ncol = 2 if W < 1100 else 3
        cw = (W - 2 * m) / ncol
        for r in range(0, len(pairs), ncol):
            row = pairs[r:r + ncol]
            hs = []
            for j, (lbl, val) in enumerate(row):
                x = m + j * cw
                c.text((x + 8, y + 6), lbl, c.small, fill=t.accent)
                hs.append(c.block(x + 8, y + 8 + c.lh(c.small), val, c.vf, cw - 16) - y + 8)
            h = max(hs)
            for j in range(len(row)):
                x = m + j * cw
                c.d.rectangle([x, y, x + cw, y + h], outline=(60, 60, 60), width=1)
            y += h
        return y + gap
    if fam == "underline":
        for lbl, val in pairs:
            c.text((m, y + 4), lbl, c.lf)
            vx = m + c.w(lbl, c.lf) + 14
            y2 = c.block(vx + rng.randint(0, 20), y, val, c.vf, W - m - vx - 20)
            line_y = max(y2, y + c.lh(c.vf)) + 2
            c.d.line([vx, line_y, W - m, line_y], fill=(90, 90, 90), width=1)
            y = line_y + gap + 6
        return y
    if fam == "table_2col":
        lw = max(c.w(lbl, c.lf) for lbl, _ in pairs) + 28
        for lbl, val in pairs:
            y2 = c.block(m + lw + 12, y + 8, val, c.vf, W - 2 * m - lw - 24)
            h = max(y2 - y + 8, c.lh(c.lf) + 16)
            c.d.rectangle([m, y, m + lw, y + h], fill=(236, 236, 236), outline=(80, 80, 80))
            c.d.rectangle([m + lw, y, W - m, y + h], outline=(80, 80, 80))
            c.text((m + 10, y + 8), lbl, c.lf)
            y += h
        return y + gap
    if fam == "receipt":
        for lbl, val in pairs:
            c.text((m, y), lbl, c.lf)
            lx = m + c.w(lbl, c.lf) + 12
            y2 = c.block(lx, y, val, c.vf, W - m - lx, align="right")
            y = max(y2, y + c.lh(c.lf)) + 2
        return y + gap
    raise KeyError(fam)


def _render_items(c: _Canvas, t: Template, v: DocValues, y: int) -> int:
    m, W = t.margin, t.page_w
    cols = [t.item_field(k) for k in t.item_cols]
    title = t.domain_def.items_zh if t.lang == "zh" else t.domain_def.items_en
    c.text((m, y), title, c.lf, fill=t.accent)
    y += c.lh(c.lf) + 4
    total = W - 2 * m
    widths = [total * 0.4] + [total * 0.6 / (len(cols) - 1)] * (len(cols) - 1) if len(cols) > 1 else [total]
    xs = [m + sum(widths[:i]) for i in range(len(cols))]
    hf = _font(t.label_role, max(13, t.label_size - 2))
    h = c.lh(hf) + 8
    if t.borders:
        c.d.rectangle([m, y, W - m, y + h], fill=(232, 232, 232), outline=(80, 80, 80))
    for x, f in zip(xs, cols):
        c.text((x + 6, y + 4), t.label_text(f), hf)
    y += h
    if not t.borders:
        c.d.line([m, y, W - m, y], fill=(80, 80, 80))
    for row in v.items:
        ys = [c.block(x + 6, y + 4, row[f.key], c.vf, wd - 12) for x, wd, f in zip(xs, widths, cols)]
        y2 = max(ys) + 4
        if t.borders:
            for x, wd in zip(xs, widths):
                c.d.rectangle([x, y, x + wd, y2], outline=(80, 80, 80))
        y = y2
    return y + 16


def render_doc(t: Template, index: int):
    """Render one document. Returns (PIL image, label dict, list of printed strings)."""
    import numpy as np
    from PIL import Image, ImageFilter

    v = sample_values(t, index)
    rng = random.Random(stable_seed("layout", t.template_id, index))
    c = _Canvas(t)
    m, W = t.margin, t.page_w
    y = m
    if t.family == "card":
        band = int(t.title_size * 1.8)
        c.d.rectangle([0, 0, W, band], fill=t.accent)
        tf = _font(t.title_role, t.title_size)
        c.text((m, (band - t.title_size) // 2), t.title, tf, fill=(255, 255, 255))
        y = band + 24
        if v.issuer:
            y = c.block(m, y, v.issuer, c.small, W - 2 * m, record=False) + 8
    else:
        if v.issuer:
            align_right = t.family == "letterhead"
            y = c.block(m, y, v.issuer, _font(t.title_role, max(18, t.title_size - 12)), W - 2 * m,
                        fill=t.accent, record=False, align="right" if align_right else "left") + 6
        tf = _font(t.title_role, t.title_size)
        tw = c.w(t.title, tf)
        tx = (W - tw) / 2 if t.title_align == "center" or t.family == "receipt" else m
        c.text((tx, y), t.title, tf)
        y += c.lh(tf) + 10
        if t.family in ("receipt", "letterhead"):
            c.d.line([m, y, W - m, y], fill=(60, 60, 60), width=2)
            y += 14
    y = _render_fields(c, t, v, y + 6, rng)
    if t.items:
        if t.family == "receipt":
            c.d.line([m, y, W - m, y], fill=(60, 60, 60))
            y += 8
        y = _render_items(c, t, v, y + 8)
    if v.footer:
        y = c.block(m, y + 10, v.footer, c.small, W - 2 * m, fill=(70, 70, 70), record=False)
    if t.footer:
        sig = "签名：" if t.lang == "zh" else "Signature:"
        c.text((W - m - 320, y + 24), sig, c.lf)
        c.d.line([W - m - 320 + c.w(sig, c.lf) + 8, y + 24 + c.lh(c.lf) - 4, W - m, y + 24 + c.lh(c.lf) - 4], fill=(0, 0, 0))
        y += 24 + c.lh(c.lf) + 10
    if t.family == "card":
        y = max(y + m, int(W * 0.6))
        c.d.rectangle([2, 2, W - 3, y - 3], outline=t.accent, width=3)
    h = max(int(y + m), int(W * 0.5))
    img = c.img.crop((0, 0, W, h))
    img = _augment(img, t, rng, np)
    return img, label_for(t, v), c.drawn


def _augment(img, t: Template, rng: random.Random, np):
    from PIL import Image, ImageFilter

    if t.scan == "clean":
        return img
    rough = t.scan == "rough"
    tint = tuple(255 - rng.randint(0, 14 if rough else 8) for _ in range(3))
    arr = np.asarray(img).astype(np.float32) * (np.array(tint, dtype=np.float32) / 255.0)
    nrng = np.random.default_rng(rng.getrandbits(32))
    arr += nrng.normal(0, 6.0 if rough else 3.0, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    ang = rng.uniform(-1.5, 1.5) if rough else rng.uniform(-0.6, 0.6)
    img = img.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=tint)
    r = rng.uniform(0.3, 0.9) if rough else rng.uniform(0.0, 0.4)
    if r > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(r))
    return img


def jpeg_quality(t: Template, index: int) -> int:
    rng = random.Random(stable_seed("jpeg", t.template_id, index))
    return {"clean": 95, "scan": rng.randint(82, 92), "rough": rng.randint(68, 82)}[t.scan]


def doc_text(label: dict) -> str:
    """All label strings of a doc, space-joined (used by the near-duplicate check)."""
    out: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif x not in (None, ""):
            out.append(str(x))

    walk(label)
    return " ".join(out)


def template_card(t: Template) -> dict:
    """task.json payload for a template task."""
    return {"task_id": t.template_id, "source": "synth", "family": "synthetic_form", "lang": t.lang,
            "instruction": INSTRUCTION, "schema": t.schema(), "template": t.to_json(),
            "fonts": {r: Path(font_path(r)).name for r in {t.label_role, t.value_role, t.title_role, TEXT_ROLES[t.lang][0]}}}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render a few synthetic docs for a look")
    ap.add_argument("--out", default="work/synth_preview")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--templates", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = make_templates()
    step = max(1, len(ts) // a.templates)
    for t in ts[::step][: a.templates]:
        for i in range(a.n):
            img, label, _ = render_doc(t, i)
            img.save(out / f"{t.template_id}_{i}.jpg", quality=jpeg_quality(t, i))
            (out / f"{t.template_id}_{i}.json").write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ts)} templates; held-out domains: {heldout_domains()}; wrote previews to {out}")
