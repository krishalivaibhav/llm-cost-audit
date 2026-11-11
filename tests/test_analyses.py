from llm_cost_audit.analyses import (downgrade_candidates, exact_duplicates, oversized_prompts,
                                     run_all, truncated_outputs, uncached_prefixes)
from llm_cost_audit.prices import call_cost
from llm_cost_audit.report import spend_breakdown
from tests.conftest import mk


def test_spend_breakdown_and_projection(book):
    calls = [mk(0, inp=1000, out=100, feature="a", secs=0),
             mk(1, model="small", inp=2000, out=0, feature="b", secs=86400 * 2),
             mk(2, model="mystery", feature="b")]
    s = spend_breakdown(calls, book)
    # big: (1000*3 + 100*15)/1e6 = 0.0045 ; small: 2000*0.5/1e6 = 0.001
    assert s.total_cost_usd == __import__("pytest").approx(0.0055)
    assert s.by_model["big"]["cost_usd"] == __import__("pytest").approx(0.0045)
    assert set(s.by_day) == {"2026-01-01", "2026-01-03"}
    assert s.span_days == 3 and s.monthly_factor == 10.0  # calendar days: Jan 1, 2 and 3
    assert s.projected_monthly_usd == __import__("pytest").approx(0.055)
    assert "mystery" in s.unpriced_models and s.priced_calls == 2
    assert any("unpriced" in w for w in s.warnings)


def test_short_span_warns(book):
    s = spend_breakdown([mk(0), mk(1, secs=60)], book)
    assert s.monthly_factor == 30.0 and any("single calendar day" in w for w in s.warnings)


def _prefix_calls(n=6, gap=10, **kw):
    return [mk(i, inp=2500, out=50, secs=i * gap, prefix_key="k", prefix_chars=8000, feature="chat", **kw)
            for i in range(n)]


def test_uncached_prefix_saving_by_hand(book):
    # prefix = 8000/4 = 2000 tokens. now: 6*2000*3/1e6 = 0.036
    # caching: 1 write 2000*3.75 + 5 reads 5*2000*0.3 = 7500+3000 -> 0.0105 ; saving 0.0255
    (f,) = uncached_prefixes(_prefix_calls(), book, 1.0)
    assert abs(f.monthly_saving_usd - 0.0255) < 1e-9
    assert f.confidence == "medium" and f.assumptions


def test_prefix_cache_expiry_means_more_writes(book):
    # calls 10 minutes apart: every call re-writes, caching costs MORE -> no finding
    assert uncached_prefixes(_prefix_calls(gap=600), book, 1.0) == []


def test_prefix_skipped_when_already_cached_or_short(book):
    assert uncached_prefixes(_prefix_calls(cached_input_tokens=100), book, 1.0) == []
    short = [mk(i, inp=2500, prefix_key="k", prefix_chars=2000) for i in range(6)]  # 500 tokens
    assert uncached_prefixes(short, book, 1.0) == []
    assert uncached_prefixes(_prefix_calls(n=3), book, 1.0) == []


def test_exact_duplicates(book):
    calls = [mk(i, prompt_key="p1", feature="faq", snippet="hi") for i in range(3)]
    calls.append(mk(9, prompt_key="p2", feature="faq"))  # unique: not counted
    calls.append(mk(10, model="small", prompt_key="p1", feature="faq"))  # other model: not a duplicate
    (f,) = exact_duplicates(calls, book, 2.0)
    one = call_cost(calls[0], book.get("big"))
    assert abs(f.monthly_saving_usd - 2 * one * 2.0) < 1e-12
    assert "2 repeat calls" in f.evidence[0]


def test_oversized_outliers(book):
    calls = [mk(i, inp=1000, out=500, feature="f") for i in range(24)]
    calls += [mk(100, inp=10000, out=500, feature="f")]  # > 4x median 1000
    (f,) = oversized_prompts(calls, book, 1.0)
    # excess over 2x median = 8000 tokens * $3/M = 0.024
    assert abs(f.monthly_saving_usd - 0.024) < 1e-9 and f.confidence == "low"


def test_extreme_ratio_flagged(book):
    calls = [mk(i, inp=5000, out=10, feature="f") for i in range(20)]
    kinds = [f.kind for f in oversized_prompts(calls, book, 1.0)]
    assert kinds == ["extreme_ratio"]


def test_small_features_are_not_judged(book):
    assert oversized_prompts([mk(i, inp=5000, out=1) for i in range(5)], book, 1.0) == []


def test_downgrade_candidate(book):
    calls = [mk(i, inp=1000, out=5, feature="cls") for i in range(20)]
    (f,) = downgrade_candidates(calls, book, 1.0)
    # big: (1000*3+5*15)/1e6 = 0.003075 ; small: (1000*.5+5*2)/1e6 = 0.00051 ; diff*20 = 0.0513
    assert abs(f.monthly_saving_usd - 20 * (0.003075 - 0.00051)) < 1e-9
    assert f.confidence == "low" and "quality eval" in f.title


def test_no_downgrade_when_outputs_long_or_already_cheapest(book):
    assert downgrade_candidates([mk(i, out=500) for i in range(20)], book, 1.0) == []
    assert downgrade_candidates([mk(i, model="small", out=5) for i in range(20)], book, 1.0) == []
    assert downgrade_candidates([mk(i, model="other", out=5) for i in range(20)], book, 1.0) == []


def test_truncated_outputs(book):
    calls = [mk(i, inp=1000, out=1024, max_tokens=1024, feature="rep", stop_reason="max_tokens") for i in range(4)]
    calls += [mk(10 + i, out=200, feature="rep") for i in range(6)]
    (f,) = truncated_outputs(calls, book, 1.0)
    cost = 4 * (1000 * 3 + 1024 * 15) / 1e6
    assert abs(f.monthly_saving_usd - 0.5 * cost) < 1e-12
    assert "4 of 10" in f.evidence[0]


def test_findings_sorted_and_unpriced_ignored(book):
    calls = [mk(i, model="mystery", prompt_key="x") for i in range(30)]
    assert run_all(calls, book, 1.0) == []
    calls = _prefix_calls(n=20) + [mk(100 + i, inp=1000, out=1024, max_tokens=1024, feature="t") for i in range(5)]
    savings = [f.monthly_saving_usd for f in run_all(calls, book, 1.0)]
    assert savings == sorted(savings, reverse=True) and len(savings) >= 2
