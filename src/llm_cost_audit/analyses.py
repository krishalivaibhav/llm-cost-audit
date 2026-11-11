"""The analyses. Each is one small function: calls in, findings out.

Shared conventions
  * `factor` scales savings measured over the log's time span to 30 days:
    factor = 30 / span_days. Every finding states that assumption.
  * Calls whose model is not in the price table are skipped (we do not guess).
  * Confidence labels are about HOW MUCH THE ESTIMATE DEPENDS ON THINGS THE
    LOG CANNOT TELL US. None of them is a guarantee; "low" means "a lead
    worth checking", not "a number to promise a client".
  * Savings from different findings overlap (a duplicated request is also
    part of a repeated prefix). Do not add them up.
"""

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

from .prices import PriceBook, call_cost, input_cost
from .records import Call

# ---- tunable assumptions, all named so the report can quote them -------------
CACHE_TTL_SECONDS = 300  # Anthropic's default cache lifetime is 5 minutes.
CHARS_PER_TOKEN = 4  # rough English average; used only to size a text prefix.
MIN_CACHEABLE_TOKENS = 1024  # prefixes shorter than this are ignored; providers
# have minimum cacheable sizes (about 1024 tokens is common, but it varies by
# model: check the provider's docs for yours).
MIN_GROUP_CALLS = 5  # a "repeated" prefix needs at least this many calls
MIN_FEATURE_CALLS = 20  # per-feature statistics need a sample this big
OUTLIER_MULTIPLE = 4  # flag calls with input > 4x their feature's median...
TRIM_TARGET_MULTIPLE = 2  # ...and assume they could be trimmed to 2x the median
EXTREME_RATIO = 100  # median input:output ratio treated as extreme
EXTREME_RATIO_MIN_INPUT = 2000  # ...only when prompts are also big
ASSUMED_RATIO_TRIM = 0.25  # illustrative: 25% of input is trimmable
REUSED_CONTEXT_SHARE = 0.8  # a feature whose input is >= 80% cache reads is
# re-sending one growing context (agents, long chats), not a bloated prompt
TINY_OUTPUT_P95 = 100  # p95 output tokens at or below this = "tiny outputs"
RETRY_SHARE = 0.5  # assumed share of truncated calls that get retried


@dataclass
class Finding:
    """One ranked result. `monthly_saving_usd` is an estimate, never a promise."""

    kind: str
    title: str
    evidence: List[str]
    monthly_saving_usd: float
    assumptions: List[str]
    confidence: str  # "low" | "medium" | "high"
    # True when the saving is a placeholder, not computed from the log.
    # Illustrative findings rank after every measured one.
    illustrative: bool = False


def _money(x: float) -> str:
    return "$%.4f" % x if x < 1 else "$%.2f" % x


def _cap(saving: float, actual_cost: float) -> float:
    """A saving can never exceed what the calls involved actually cost, and
    never be negative. Applied to every analysis as a last sanity check on
    the arithmetic."""
    return max(0.0, min(saving, actual_cost))


def _percentile(values: List[int], q: float) -> float:
    """Nearest-rank percentile; simple enough to verify by hand."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def _by_feature(calls: List[Call]) -> Dict[str, List[Call]]:
    groups: Dict[str, List[Call]] = defaultdict(list)
    for c in calls:
        groups[c.feature].append(c)
    return groups


# ----------------------------------------------------------------- 2. prefixes


def uncached_prefixes(calls: List[Call], prices: PriceBook, factor: float) -> List[Finding]:
    """Long identical leading text, sent again and again, with no cache hits.

    WHY: providers bill cached prefix tokens at a small fraction of the normal
    input rate, but only if you opt in (Anthropic) or structure the prompt so
    the shared part comes first (OpenAI). A big system prompt resent on every
    call with cached_input_tokens == 0 is the classic missed saving.

    HOW the saving is computed: we replay the calls in time order. The first
    call (and any call arriving more than CACHE_TTL_SECONDS after the previous
    one in the group, because the cache has expired) pays the cache-WRITE rate
    on the prefix; every other call pays the cache-READ rate. Saving = what
    the prefix cost now, minus that. Prefix size is estimated from text length
    (chars / 4) because the log has no per-part token count.
    """
    groups: Dict[tuple, List[Call]] = defaultdict(list)
    for c in calls:
        if c.prefix_key:
            groups[(c.model, c.prefix_key)].append(c)

    findings = []
    for (model, _), group in groups.items():
        price = prices.get(model)
        if price is None or len(group) < MIN_GROUP_CALLS:
            continue
        if any(c.cached_input_tokens or c.cache_write_tokens for c in group):
            continue  # caching is already in use for this prefix
        # Never claim more prefix than the calls actually contained.
        prefix_tokens = min(group[0].prefix_chars // CHARS_PER_TOKEN,
                            min(c.total_input_tokens for c in group))
        if prefix_tokens < MIN_CACHEABLE_TOKENS:
            continue

        dated = sorted((c for c in group if c.timestamp), key=lambda c: c.timestamp)
        undated = len(group) - len(dated)
        writes = 1  # the very first call always has to write the cache
        reads = 0
        for prev, cur in zip(dated, dated[1:]):
            if cur.timestamp - prev.timestamp <= timedelta(seconds=CACHE_TTL_SECONDS):
                reads += 1
            else:
                writes += 1
        reads += undated  # no timestamp: optimistic, and flagged below

        now_cost = len(group) * prefix_tokens * price.input / 1e6
        cached_cost = (writes * prefix_tokens * price.cache_write + reads * prefix_tokens * price.cache_read) / 1e6
        saving = _cap(now_cost - cached_cost, sum(call_cost(c, price) for c in group))
        if saving <= 0:
            continue

        feature = Counter(c.feature for c in group).most_common(1)[0][0]
        assumptions = [
            "Prefix size is estimated at %d tokens (text length / %d, capped at the smallest call's input); the log has no exact figure."
            % (prefix_tokens, CHARS_PER_TOKEN),
            "A cache entry lives %d s and is refreshed by each hit; calls further apart than that pay a cache write again."
            % CACHE_TTL_SECONDS,
            "Prices used: input %s, cache write %s, cache read %s per million tokens (%s)."
            % (price.input, price.cache_write, price.cache_read, model),
            "The provider caches this prefix (minimum cacheable size varies by model) and that you can put the shared text first.",
        ]
        if undated:
            assumptions.append("%d calls had no timestamp and were assumed to be cache hits (optimistic)." % undated)
        findings.append(Finding(
            kind="uncached_prefix",
            title="Repeated long prefix with no caching: feature '%s' on %s" % (feature, model),
            evidence=[
                "%d calls share an identical ~%d-token prefix and report 0 cached tokens." % (len(group), prefix_tokens),
                "Replay: %d cache writes and %d cache reads. Prefix cost now %s, with caching %s over the log period."
                % (writes, reads, _money(now_cost), _money(cached_cost)),
            ],
            monthly_saving_usd=saving * factor,
            assumptions=assumptions,
            confidence="medium" if not undated else "low",
        ))
    return findings


# --------------------------------------------------------------- 3. duplicates


def exact_duplicates(calls: List[Call], prices: PriceBook, factor: float,
                     show_snippets: bool = False) -> List[Finding]:
    """Same model, same system prompt, same prompt, sent more than once.

    WHY: if the same question is asked repeatedly (FAQ bots, retries without
    idempotency keys, cron jobs re-running on unchanged data) a response cache
    answers repeats for free. Saving = the full cost of every repeat after the
    first call of each group.

    Evidence shows the prompt's hash, not its text, unless show_snippets is
    set: prompts often hold customer data, and an audit report gets forwarded.

    Limit: only valid when an identical prompt deserves an identical answer.
    Sampling-heavy features (brainstorming, temperature > 0 on purpose) do not
    qualify, and the log cannot tell us which is which.
    """
    groups: Dict[tuple, List[Call]] = defaultdict(list)
    for c in calls:
        if c.prompt_key:
            groups[(c.model, c.prompt_key)].append(c)

    per_feature: Dict[str, dict] = {}
    for (model, _), group in groups.items():
        price = prices.get(model)
        if price is None or len(group) < 2:
            continue
        group = sorted(group, key=lambda c: (c.timestamp is None, c.timestamp, c.line_no))
        repeat_cost = _cap(sum(call_cost(c, price) for c in group[1:]), sum(call_cost(c, price) for c in group))
        f = per_feature.setdefault(group[0].feature, {"cost": 0.0, "repeats": 0, "groups": 0, "example": ""})
        f["cost"] += repeat_cost
        f["repeats"] += len(group) - 1
        f["groups"] += 1
        if not f["example"]:
            if show_snippets and group[0].snippet:
                f["example"] = "Example prompt start: %r" % group[0].snippet
            else:
                f["example"] = "Example prompt hash: %s" % group[0].prompt_key

    findings = []
    for feature, f in per_feature.items():
        if f["cost"] <= 0:
            continue
        evidence = ["%d repeat calls across %d distinct repeated requests; repeats cost %s over the log period."
                    % (f["repeats"], f["groups"], _money(f["cost"]))]
        if f["example"]:
            evidence.append(f["example"])
        findings.append(Finding(
            kind="exact_duplicate",
            title="Exact duplicate requests: feature '%s'" % feature,
            evidence=evidence,
            monthly_saving_usd=f["cost"] * factor,
            assumptions=[
                "A response cache would serve every repeat after the first from storage at zero model cost.",
                "Identical prompt (and system prompt) on the same model should get a reusable answer; "
                "this fails for features that want varied outputs.",
                "Ignores cache expiry/invalidation and the cost of running a cache.",
            ],
            confidence="medium",
        ))
    return findings


# ---------------------------------------------------------------- 4. oversized


def oversized_prompts(calls: List[Call], prices: PriceBook, factor: float) -> List[Finding]:
    """Two separate signals that a feature sends more context than it needs.

    (a) Outliers: a call whose input is > OUTLIER_MULTIPLE x its OWN feature's
        median. We compare a feature with itself, not with other features,
        because legitimately different jobs have different sizes. Typical
        cause: a bug that appends a whole history or file by accident.
        Saving assumes the call could have been trimmed to
        TRIM_TARGET_MULTIPLE x the median (an assumption, not a measurement).
    (b) Extreme input:output ratio: a feature whose median call reads
        >= EXTREME_RATIO tokens per output token with large prompts. That is
        sometimes right (classification of long docs) and sometimes a sign of
        bloated context; the saving shown is an illustrative
        ASSUMED_RATIO_TRIM share of that feature's input spend.
    """
    findings = []
    for feature, group in _by_feature(calls).items():
        priced = [(c, prices.get(c.model)) for c in group]
        priced = [(c, p) for c, p in priced if p is not None]
        if len(priced) < MIN_FEATURE_CALLS:
            continue
        sizes = [c.total_input_tokens for c, _ in priced]
        median = statistics.median(sizes)
        if median <= 0:
            continue

        # (a) outliers
        limit = OUTLIER_MULTIPLE * median
        target = TRIM_TARGET_MULTIPLE * median
        outliers = [(c, p) for c, p in priced if c.total_input_tokens > limit]
        # Trimmed tokens are assumed to come off the call's input pro rata across its
        # token types, so each type is priced at its own rate (cached tokens are cheap).
        excess_cost = sum(_cap(input_cost(c, p) * (c.total_input_tokens - target) / c.total_input_tokens,
                               call_cost(c, p)) for c, p in outliers)
        if outliers and excess_cost > 0:
            biggest = max(c.total_input_tokens for c, _ in outliers)
            findings.append(Finding(
                kind="oversized_prompt",
                title="Oversized prompts (outliers): feature '%s'" % feature,
                evidence=[
                    "%d of %d calls have input > %dx the feature's median (%d tokens); largest is %d tokens."
                    % (len(outliers), len(priced), OUTLIER_MULTIPLE, median, biggest),
                    "Tokens above %dx median cost %s over the log period at their own rates."
                    % (TRIM_TARGET_MULTIPLE, _money(excess_cost)),
                ],
                monthly_saving_usd=excess_cost * factor,
                assumptions=[
                    "Flagged calls could be trimmed to %dx the feature's median input without hurting results (unverified)." % TRIM_TARGET_MULTIPLE,
                    "Removed tokens are assumed to come off pro rata across full-rate, cached and cache-write input, each at its own price.",
                ],
                confidence="low",
            ))

        # (b) extreme ratio
        ratios = [c.total_input_tokens / max(c.output_tokens, 1) for c, _ in priced]
        median_ratio = statistics.median(ratios)
        if median_ratio >= EXTREME_RATIO and median >= EXTREME_RATIO_MIN_INPUT:
            feature_input_cost = sum(input_cost(c, p) for c, p in priced)
            # WHY split: an agent or long chat re-sends its whole history on
            # every turn, mostly from cache. That looks like "huge prompt,
            # tiny answer" but the fix is compacting the history, not
            # trimming a prompt template, so the advice differs.
            total_in = sum(c.total_input_tokens for c, _ in priced)
            read_share = sum(c.cached_input_tokens for c, _ in priced) / total_in if total_in else 0.0
            evidence = [
                "Median call reads %d input tokens per output token (median input %d tokens)." % (median_ratio, median),
                "Input spend for this feature: %s over the log period." % _money(feature_input_cost),
            ]
            if read_share >= REUSED_CONTEXT_SHARE:
                kind, title = "reused_context", "Long reused context (agent or chat history): feature '%s'" % feature
                evidence.append("%d%% of input tokens are cache reads: the same growing context is re-sent each call."
                                % int(read_share * 100))
                lever = ("ILLUSTRATIVE: assumes %d%% of the input could be removed by compacting or summarising "
                         "older history. This is a placeholder, not a measurement." % int(ASSUMED_RATIO_TRIM * 100))
                caveat = "Agents and long chats need much of their history; check what the model actually uses before cutting."
            else:
                kind, title = "extreme_ratio", "Extreme input:output ratio: feature '%s'" % feature
                lever = ("ILLUSTRATIVE: assumes %d%% of the input could be removed (retrieval top-k, shorter history, "
                         "pre-filtering). This is a placeholder, not a measurement." % int(ASSUMED_RATIO_TRIM * 100))
                caveat = "Long input with short output is sometimes correct (e.g. classifying documents); this flags it for a human look."
            findings.append(Finding(
                kind=kind,
                title=title,
                evidence=evidence,
                monthly_saving_usd=_cap(ASSUMED_RATIO_TRIM * feature_input_cost, feature_input_cost) * factor,
                assumptions=[lever, caveat],
                confidence="low",
                illustrative=True,
            ))
    return findings


# ------------------------------------------------------------- 5. downgrades


def downgrade_candidates(calls: List[Call], prices: PriceBook, factor: float) -> List[Finding]:
    """Expensive model, tiny outputs, same feature every time.

    WHY: outputs of a few tokens (labels, yes/no, routing decisions) are the
    classic job a smaller model can do. Saving = recompute the same tokens at
    the cheapest listed model of the SAME provider, minus today's cost.

    THIS IS A LEAD, NOT A RECOMMENDATION. Short outputs say nothing about
    whether the smaller model gets the answers right. It needs a quality
    eval on your own data before any switch.
    """
    findings = []
    groups: Dict[tuple, List[Call]] = defaultdict(list)
    for c in calls:
        groups[(c.feature, c.model)].append(c)
    for (feature, model), group in groups.items():
        price = prices.get(model)
        if price is None or len(group) < MIN_FEATURE_CALLS:
            continue
        cheap = prices.cheapest(price.provider)
        if cheap is None or cheap.model == price.model or cheap.input + cheap.output >= price.input + price.output:
            continue
        p95 = _percentile([c.output_tokens for c in group], 0.95)
        if p95 > TINY_OUTPUT_P95:
            continue
        now = sum(call_cost(c, price) for c in group)
        then = sum(call_cost(c, cheap) for c in group)
        saving = _cap(now - then, now)
        if saving <= 0:
            continue
        findings.append(Finding(
            kind="downgrade_candidate",
            title="Model downgrade candidate (needs a quality eval before switching): feature '%s'" % feature,
            evidence=[
                "%d calls on %s; 95%% of outputs are <= %d tokens." % (len(group), model, p95),
                "Same tokens on %s would cost %s instead of %s over the log period." % (cheap.model, _money(then), _money(now)),
            ],
            monthly_saving_usd=saving * factor,
            assumptions=[
                "The cheaper model must do the job as well; the log says nothing about answer quality. Run an eval first.",
                "Cheapest model = lowest (input price + output price) among %s models in the price table." % price.provider,
                "Token counts stay the same on the new model (tokenizers differ, so real counts will move a little).",
            ],
            confidence="low",
        ))
    return findings


# ----------------------------------------------------------------- 6. truncation


def truncated_outputs(calls: List[Call], prices: PriceBook, factor: float) -> List[Finding]:
    """Calls that stopped because they hit max_tokens.

    WHY: a cut-off answer is usually unusable (broken JSON, half a summary),
    so it is either retried (you pay twice) or shipped broken. The log shows
    the truncation, not what the application did next, so the saving assumes
    RETRY_SHARE of truncated calls are retried at the same cost.
    """
    findings = []
    for feature, group in _by_feature(calls).items():
        hit = [c for c in group if c.truncated and prices.get(c.model)]
        if len(hit) < 3:
            continue
        cost = sum(call_cost(c, prices.get(c.model)) for c in hit)
        limits = Counter(c.max_tokens for c in hit if c.max_tokens)
        evidence = ["%d of %d calls (%.1f%%) hit the output limit; they cost %s over the log period."
                    % (len(hit), len(group), 100.0 * len(hit) / len(group), _money(cost))]
        if limits:
            evidence.append("Most common max_tokens on truncated calls: %s." % limits.most_common(1)[0][0])
        findings.append(Finding(
            kind="truncated_output",
            title="Truncated outputs (hit max_tokens): feature '%s'" % feature,
            evidence=evidence,
            monthly_saving_usd=_cap(RETRY_SHARE * cost, cost) * factor,
            assumptions=[
                "%d%% of truncated calls are retried at about the same cost; the log does not show retries, so this is a guess." % int(RETRY_SHARE * 100),
                "Saving is only realised if the limit is raised or the prompt asks for shorter answers; a higher limit also raises cost per call.",
            ],
            confidence="low",
        ))
    return findings


def run_all(calls: List[Call], prices: PriceBook, factor: float, show_snippets: bool = False) -> List[Finding]:
    """Run every analysis and sort by estimated monthly saving, largest first,
    with illustrative (placeholder) savings after all measured ones: a guess
    must never push a figure computed from the log down the list."""
    findings: List[Finding] = []
    findings.extend(uncached_prefixes(calls, prices, factor))
    findings.extend(exact_duplicates(calls, prices, factor, show_snippets))
    findings.extend(oversized_prompts(calls, prices, factor))
    findings.extend(downgrade_candidates(calls, prices, factor))
    findings.extend(truncated_outputs(calls, prices, factor))
    findings.sort(key=lambda f: (f.illustrative, -f.monthly_saving_usd))
    return findings
