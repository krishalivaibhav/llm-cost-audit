"""Spend breakdown (analysis 1), report assembly and the three renderers."""

import html
import json
import math
from datetime import timedelta
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .analyses import Finding, run_all
from .prices import PriceBook, call_cost
from .records import Call, LoadResult, strip_invisible

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
    span_days: Optional[int]  # calendar days covered (robust, see _robust_span); None if < 2 dated calls
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
    number of CALENDAR days (UTC) the log covers, counting both end days, and
    measured after trimming timestamps isolated by a long gap (see
    _robust_span) so that one corrupt timestamp (e.g. 0, meaning 1970) cannot
    stretch the span.

    Why calendar days: first-to-last-call time understates the period (14 days
    of traffic shows as ~13.4 days because traffic starts mid-morning on day 1
    and stops before midnight on day 14), which inflates the projection.
    Counting whole days removes that bias for logs that cover many days. The
    opposite bias exists for SHORT logs: 26 hours spanning two dates count as
    2 days, understating the projection. Trimming 1% at each end also drops
    about 2% of calls from the span but not from the cost total, a small
    upward bias. This is a straight-line projection: it ignores growth,
    weekday/weekend swings and price changes.
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
    stamps = sorted(c.timestamp for c in calls if c.timestamp)
    span_days: Optional[int] = None
    if len(stamps) >= 2:
        span_days, ignored, full_days = _robust_span(stamps)
        if full_days - span_days >= max(1, 0.1 * span_days):
            warnings.append(
                "%d outlier timestamps isolated from the rest of the log by a long gap were ignored when measuring the span "
                "(they would have stretched it from %d to %d calendar days). Check the log for bad timestamps."
                % (ignored, span_days, full_days))
        if span_days == 1:
            warnings.append("Log covers a single calendar day; projections treat it as a full day of traffic. "
                            "Treat monthly figures as rough.")
    else:
        warnings.append("Log has fewer than two timestamps; projections treat it as one day of traffic. "
                        "Treat monthly figures as rough.")
    effective = float(span_days or 1)
    factor = DAYS_PER_MONTH / effective
    if unpriced:
        warnings.append("%d calls use models missing from the price table (unpriced, excluded from all dollar figures): %s."
                        % (sum(u["calls"] for u in unpriced.values()), ", ".join(sorted(unpriced))))
    return SpendSummary(len(calls), priced_calls, total, by_model, by_feature, by_day, unpriced,
                        span_days, factor, total * factor, warnings)


def _robust_span(stamps):
    """(calendar days of the trimmed range, timestamps trimmed, calendar days
    of the full range). `stamps` must be sorted and have at least two entries.

    WHY: one wrong clock (a record stamped 1970, or next year) would stretch
    the span to decades and shrink every monthly estimate to zero. A
    percentile cut cannot drop a single bad stamp from a small log, so we
    trim by gap instead: walking in from each end, a timestamp is dropped
    while it sits further from its neighbour than a week or five times the
    spread of the middle half of the data, whichever is larger. Real traffic
    with ordinary quiet periods is left alone."""
    n = len(stamps)
    q1, q3 = stamps[n // 4], stamps[(3 * n) // 4]
    gap = max(timedelta(days=7), 5 * (q3 - q1))
    lo, hi = 0, n - 1
    while lo < hi and stamps[lo + 1] - stamps[lo] > gap:
        lo += 1
    while hi > lo and stamps[hi] - stamps[hi - 1] > gap:
        hi -= 1
    return ((stamps[hi].date() - stamps[lo].date()).days + 1, lo + (n - 1 - hi),
            (stamps[-1].date() - stamps[0].date()).days + 1)


@dataclass
class Report:
    source: str
    lines_read: int
    malformed: int
    malformed_examples: List[list]
    summary: SpendSummary
    findings: List[Finding]
    price_sources: List[dict]


def build_report(source: str, loaded: LoadResult, prices: PriceBook, show_snippets: bool = False) -> Report:
    summary = spend_breakdown(loaded.calls, prices)
    findings = run_all(loaded.calls, prices, summary.monthly_factor, show_snippets)
    if loaded.undated:
        summary.warnings.append("%d calls had no usable timestamp (excluded from the day breakdown and span)." % loaded.undated)
    unpriced_1h = sorted({c.model for c in loaded.calls if c.cache_write_1h_tokens and prices.get(c.model)
                          and prices.get(c.model).cache_write_1h is None})
    if unpriced_1h:
        summary.warnings.append("1-hour cache writes on models with no 1-hour write price are priced at the "
                                "5-minute rate, which may understate cost: %s." % ", ".join(unpriced_1h))
    used = sorted(set(summary.by_model))
    sources = []
    for m in used:
        p = prices.get(m)
        sources.append({"model": m, "priced_as": p.model, "input": p.input, "output": p.output,
                        "cache_read": p.cache_read, "cache_write": p.cache_write, "cache_write_1h": p.cache_write_1h,
                        "source_url": p.source_url, "as_of": p.as_of})
    # Only the file name goes in the report: an absolute path leaks usernames and directory layout.
    return Report(strip_invisible(os.path.basename(source)), loaded.lines_read, loaded.malformed, [list(e) for e in loaded.malformed_examples],
                  summary, findings, sources)


# ------------------------------------------------------------------ renderers


def _usd(x: float) -> str:
    return "$%.2f" % x if abs(x) >= 0.01 or x == 0 else "$%.4f" % x


def _or_dash(price: Optional[float]) -> str:
    return "-" if price is None else str(price)


def _md(text: str) -> str:
    """Escape text for Markdown/HTML-capable viewers. Log-derived strings
    (feature tags, prompt snippets) are untrusted: neutralise HTML and the
    characters that make links, code spans and table cells."""
    text = html.escape(strip_invisible(text), quote=True)
    for ch in "\\`*_[]|":
        text = text.replace(ch, "\\" + ch)
    # Break bare-URL auto-linking: "://" and "www." become HTML entities, which
    # viewers render as the same text but do not turn into clickable links.
    return text.replace("://", "&#58;//").replace("www.", "www&#46;")


def _table_rows(table: Dict[str, dict], limit: Optional[int] = None):
    rows = sorted(table.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)
    return rows[:limit] if limit else rows


# Every finding's saving is scaled the same way, so the report says it once
# here instead of repeating it in each finding's assumptions.
_SCALE_NOTE = ("Savings measured over the log period are scaled by %.2f to a 30-day month "
               "(assumes the log's traffic is representative).")


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
    span = "unknown" if s.span_days is None else "%d calendar days" % s.span_days
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
    add(_SCALE_NOTE % s.monthly_factor)
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
        add("  %s: in %s, out %s, cache read %s, cache write %s, 1-hour cache write %s (checked %s, %s)"
            % (p["model"], p["input"], p["output"], p["cache_read"], p["cache_write"], _or_dash(p["cache_write_1h"]),
               p["as_of"], p["source_url"]))
    return "\n".join(out) + "\n"


def render_markdown(r: Report) -> str:
    s = r.summary
    out: List[str] = []
    add = out.append
    add("# LLM cost audit")
    add("")
    add("- Source: `%s`" % r.source.replace("`", "'"))
    add("- Lines read: %d; calls parsed: %d; malformed (skipped): %d" % (r.lines_read, s.total_calls, r.malformed))
    span = "unknown" if s.span_days is None else "%d calendar days" % s.span_days
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
    add("")
    add(_SCALE_NOTE % s.monthly_factor)
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
    add("| Model | Input | Output | Cache read | Cache write | 1-hour cache write | Checked | Source |")
    add("|---|---:|---:|---:|---:|---:|---|---|")
    for p in r.price_sources:
        add("| %s | %s | %s | %s | %s | %s | %s | %s |" % (_md(p["model"]), p["input"], p["output"], p["cache_read"],
                                                           p["cache_write"], _or_dash(p["cache_write_1h"]),
                                                           _md(p["as_of"]), _md(p["source_url"])))
    return "\n".join(out) + "\n"


def render_json(r: Report) -> str:
    # allow_nan=False makes json.dumps raise instead of ever writing Infinity/NaN.
    return json.dumps(asdict(r), indent=2, sort_keys=False, allow_nan=False) + "\n"


RENDERERS = {"text": render_text, "markdown": render_markdown, "json": render_json}
