"""Spend breakdown (analysis 1), report assembly and the three renderers."""

import html
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .analyses import Finding, run_all
from .prices import PriceBook, call_cost
from .records import Call, LoadResult

DAYS_PER_MONTH = 30  # projections use a flat 30-day month; stated in the report


@dataclass
class SpendSummary:
    total_calls: int
    priced_calls: int
    total_cost_usd: float
    by_model: Dict[str, dict]
    by_feature: Dict[str, dict]
    by_day: Dict[str, dict]
    unpriced_models: Dict[str, dict]
    span_days: Optional[float]  # None if fewer than two dated calls
    monthly_factor: float  # multiply log-period amounts by this for a 30-day month
    projected_monthly_usd: float
    warnings: List[str] = field(default_factory=list)


def _bucket(table: Dict[str, dict], key: str, cost: float, call: Call) -> None:
    b = table.setdefault(key, {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
    b["calls"] += 1
    b["cost_usd"] += cost
    b["input_tokens"] += call.total_input_tokens
    b["output_tokens"] += call.output_tokens


def spend_breakdown(calls: List[Call], prices: PriceBook) -> SpendSummary:
    """Where the money went, and a projection to 30 days.

    Projection = total priced cost / span_days * 30, where span_days is the
    time between the earliest and latest timestamp. A log shorter than one
    day is treated as one day (a few hours of traffic would otherwise be
    extrapolated wildly), and the report says so. This is a straight-line
    projection: it ignores growth, weekday/weekend swings and price changes.
    """
    by_model: Dict[str, dict] = {}
    by_feature: Dict[str, dict] = {}
    by_day: Dict[str, dict] = {}
    unpriced: Dict[str, dict] = {}
    total = 0.0
    priced_calls = 0
    for c in calls:
        price = prices.get(c.model)
        if price is None:
            _bucket(unpriced, c.model, 0.0, c)
            continue
        cost = call_cost(c, price)
        priced_calls += 1
        total += cost
        _bucket(by_model, c.model, cost, c)
        _bucket(by_feature, c.feature, cost, c)
        _bucket(by_day, c.timestamp.date().isoformat() if c.timestamp else "(no timestamp)", cost, c)

    warnings: List[str] = []
    stamps = [c.timestamp for c in calls if c.timestamp]
    span_days: Optional[float] = None
    if len(stamps) >= 2:
        span_days = (max(stamps) - min(stamps)).total_seconds() / 86400.0
    effective = span_days if span_days is not None else 0.0
    if effective < 1.0:
        warnings.append(
            "Log spans %s; projections treat it as a full day of traffic. Treat monthly figures as rough."
            % ("under one day" if span_days is not None else "an unknown period (fewer than two timestamps)"))
        effective = 1.0
    factor = DAYS_PER_MONTH / effective
    if unpriced:
        warnings.append("%d calls use models missing from the price table (unpriced, excluded from all dollar figures): %s."
                        % (sum(u["calls"] for u in unpriced.values()), ", ".join(sorted(unpriced))))
    return SpendSummary(len(calls), priced_calls, total, by_model, by_feature, by_day, unpriced,
                        span_days, factor, total * factor, warnings)


@dataclass
class Report:
    source: str
    lines_read: int
    malformed: int
    malformed_examples: List[list]
    summary: SpendSummary
    findings: List[Finding]
    price_sources: List[dict]


def build_report(source: str, loaded: LoadResult, prices: PriceBook) -> Report:
    summary = spend_breakdown(loaded.calls, prices)
    findings = run_all(loaded.calls, prices, summary.monthly_factor)
    if loaded.undated:
        summary.warnings.append("%d calls had no usable timestamp (excluded from the day breakdown and span)." % loaded.undated)
    used = sorted(set(summary.by_model))
    sources = []
    for m in used:
        p = prices.get(m)
        sources.append({"model": m, "priced_as": p.model, "input": p.input, "output": p.output,
                        "cache_read": p.cache_read, "cache_write": p.cache_write,
                        "source_url": p.source_url, "as_of": p.as_of})
    return Report(source, loaded.lines_read, loaded.malformed, [list(e) for e in loaded.malformed_examples],
                  summary, findings, sources)


# ------------------------------------------------------------------ renderers


def _usd(x: float) -> str:
    return "$%.2f" % x if abs(x) >= 0.01 or x == 0 else "$%.4f" % x


def _md(text: str) -> str:
    """Escape text for Markdown/HTML-capable viewers. Log-derived strings
    (feature tags, prompt snippets) are untrusted: neutralise HTML and the
    characters that make links, code spans and table cells."""
    text = html.escape(str(text), quote=True)
    for ch in "\\`*_[]|":
        text = text.replace(ch, "\\" + ch)
    return text


def _table_rows(table: Dict[str, dict], limit: Optional[int] = None):
    rows = sorted(table.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)
    return rows[:limit] if limit else rows


def render_text(r: Report) -> str:
    s = r.summary
    out: List[str] = []
    add = out.append
    add("LLM COST AUDIT")
    add("=" * 60)
    add("Source: %s" % r.source)
    add("Lines read: %d   Calls parsed: %d   Malformed (skipped): %d" % (r.lines_read, s.total_calls, r.malformed))
    for line_no, why in r.malformed_examples:
        add("  line %d: %s" % (line_no, why))
    span = "unknown" if s.span_days is None else "%.1f days" % s.span_days
    add("Log span: %s   Priced calls: %d of %d" % (span, s.priced_calls, s.total_calls))
    add("Spend in log: %s   Projected 30-day spend: %s (straight-line, x%.2f)"
        % (_usd(s.total_cost_usd), _usd(s.projected_monthly_usd), s.monthly_factor))
    for w in s.warnings:
        add("WARNING: " + w)

    for title, table in (("BY MODEL", s.by_model), ("BY FEATURE", s.by_feature), ("BY DAY", s.by_day)):
        add("")
        add(title)
        for key, b in (sorted(table.items()) if title == "BY DAY" else _table_rows(table)):
            share = 100.0 * b["cost_usd"] / s.total_cost_usd if s.total_cost_usd else 0.0
            add("  %-28s %10s %5.1f%%  %6d calls" % (key[:28], _usd(b["cost_usd"]), share, b["calls"]))

    add("")
    add("FINDINGS (ranked by estimated monthly saving; estimates, not guarantees; they overlap, do not add them up)")
    add("-" * 60)
    if not r.findings:
        add("No findings.")
    for i, f in enumerate(r.findings, 1):
        add("%d. %s" % (i, f.title))
        add("   Estimated saving: %s/month   Confidence: %s" % (_usd(f.monthly_saving_usd), f.confidence))
        for e in f.evidence:
            add("   Evidence: " + e)
        for a in f.assumptions:
            add("   Assumes: " + a)
        add("")
    add("PRICES USED (USD per million tokens)")
    for p in r.price_sources:
        add("  %s: in %s, out %s, cache read %s, cache write %s (checked %s, %s)"
            % (p["model"], p["input"], p["output"], p["cache_read"], p["cache_write"], p["as_of"], p["source_url"]))
    return "\n".join(out) + "\n"


def render_markdown(r: Report) -> str:
    s = r.summary
    out: List[str] = []
    add = out.append
    add("# LLM cost audit")
    add("")
    add("- Source: `%s`" % r.source.replace("`", "'"))
    add("- Lines read: %d; calls parsed: %d; malformed (skipped): %d" % (r.lines_read, s.total_calls, r.malformed))
    span = "unknown" if s.span_days is None else "%.1f days" % s.span_days
    add("- Log span: %s; priced calls: %d of %d" % (span, s.priced_calls, s.total_calls))
    add("- Spend in log: **%s**; projected 30-day spend: **%s** (straight-line, x%.2f)"
        % (_usd(s.total_cost_usd), _usd(s.projected_monthly_usd), s.monthly_factor))
    for w in s.warnings:
        add("- **Warning:** " + _md(w))
    for title, table in (("By model", s.by_model), ("By feature", s.by_feature), ("By day", s.by_day)):
        add("")
        add("## " + title)
        add("")
        add("| Name | Cost | Share | Calls |")
        add("|---|---:|---:|---:|")
        for key, b in (sorted(table.items()) if title == "By day" else _table_rows(table)):
            share = 100.0 * b["cost_usd"] / s.total_cost_usd if s.total_cost_usd else 0.0
            add("| %s | %s | %.1f%% | %d |" % (_md(key), _usd(b["cost_usd"]), share, b["calls"]))
    add("")
    add("## Findings")
    add("")
    add("Ranked by estimated monthly saving. These are estimates, not guarantees, and they overlap: do not add them up.")
    if not r.findings:
        add("")
        add("No findings.")
    for i, f in enumerate(r.findings, 1):
        add("")
        add("### %d. %s" % (i, _md(f.title)))
        add("")
        add("Estimated saving: **%s/month**. Confidence: **%s**." % (_usd(f.monthly_saving_usd), f.confidence))
        add("")
        add("Evidence:")
        for e in f.evidence:
            add("- " + _md(e))
        add("")
        add("Assumptions:")
        for a in f.assumptions:
            add("- " + _md(a))
    add("")
    add("## Prices used (USD per million tokens)")
    add("")
    add("| Model | Input | Output | Cache read | Cache write | Checked | Source |")
    add("|---|---:|---:|---:|---:|---|---|")
    for p in r.price_sources:
        add("| %s | %s | %s | %s | %s | %s | %s |" % (_md(p["model"]), p["input"], p["output"], p["cache_read"],
                                                      p["cache_write"], _md(p["as_of"]), _md(p["source_url"])))
    return "\n".join(out) + "\n"


def render_json(r: Report) -> str:
    return json.dumps(asdict(r), indent=2, sort_keys=False) + "\n"


RENDERERS = {"text": render_text, "markdown": render_markdown, "json": render_json}
