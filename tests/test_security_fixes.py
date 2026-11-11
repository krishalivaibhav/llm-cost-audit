"""One regression test per item of the security review."""
import json
import os
from datetime import timedelta

import pytest

from llm_cost_audit.analyses import oversized_prompts, truncated_outputs, uncached_prefixes
from llm_cost_audit.cli import main
from llm_cost_audit.prices import load_prices, parse_prices
from llm_cost_audit.records import load_calls, normalise, safe_text
from llm_cost_audit.report import build_report, render_json, render_markdown, render_text, spend_breakdown
from tests.conftest import T0, mk


def _log(tmp_path, *records, name="log.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _good(**kw):
    d = {"model": "gpt-4o-mini", "input_tokens": 100, "output_tokens": 10}
    d.update(kw)
    return d


# 1 ---------------------------------------------------------------------------
def test_01_huge_integer_is_malformed_not_a_crash(tmp_path):
    p = _log(tmp_path, '{"model":"m","input_tokens":%s,"output_tokens":1}' % ("9" * 400), _good())
    r = load_calls(str(p))
    assert len(r.calls) == 1 and r.malformed == 1
    build_report("x", r, load_prices())  # must not raise


# 2 ---------------------------------------------------------------------------
@pytest.mark.parametrize("tok", ["1e308", "Infinity", "NaN", "1e400"])
def test_02_non_finite_or_giant_tokens_rejected(tmp_path, tok):
    p = _log(tmp_path, '{"model":"m","input_tokens":%s,"output_tokens":1}' % tok, _good())
    assert load_calls(str(p)).malformed == 1


def test_02_prices_must_be_finite():
    with pytest.raises(ValueError):
        parse_prices({"models": {"m": {"input": 1e308, "output": 1, "cache_read": 0, "cache_write": 0}}})
    with pytest.raises(ValueError):
        parse_prices({"models": {"m": {"input": float("nan"), "output": 1, "cache_read": 0, "cache_write": 0}}})


def test_02_max_size_tokens_still_produce_valid_json(tmp_path):
    p = _log(tmp_path, _good(input_tokens=10 ** 12, output_tokens=10 ** 12))
    out = render_json(build_report("x", load_calls(str(p)), load_prices()))
    assert "Infinity" not in out and "NaN" not in out
    json.loads(out, parse_constant=lambda c: pytest.fail("non-finite constant " + c))


# 3 ---------------------------------------------------------------------------
def test_03_one_bad_timestamp_does_not_stretch_the_span(book):
    calls = [mk(i, secs=i * 3600) for i in range(200)]  # about 8.3 days of traffic
    good_days = spend_breakdown(calls, book).span_days
    calls.append(mk(999, timestamp=T0.replace(year=1970)))
    s = spend_breakdown(calls, book)
    assert s.span_days == good_days and s.span_days < 15
    assert any("outlier timestamps" in w for w in s.warnings)


# 4 ---------------------------------------------------------------------------
def test_04_prompt_only_prefix_path_fires(book):
    shared = "Shared instructions. " * 1000  # about 21k chars
    calls = [normalise({"model": "big", "input_tokens": 6000, "output_tokens": 50, "feature": "f",
                        "timestamp": (T0 + timedelta(seconds=10 * i)).isoformat(),
                        "prompt": shared + " unique tail %d" % i}, i) for i in range(30)]
    (f,) = uncached_prefixes(calls, book, 1.0)
    assert "30 calls share" in f.evidence[0] and "~1024-token" in f.evidence[0]


# 5 ---------------------------------------------------------------------------
def test_05_oversized_saving_uses_each_tokens_own_rate_and_is_capped(book):
    # outlier is almost all cache-read tokens at 0.3/M; full-rate pricing would claim far more
    calls = [mk(i, inp=1000, out=10, feature="f") for i in range(24)]
    big = mk(99, inp=100, out=10, cached_input_tokens=19900, feature="f")  # 20000 total, median 1000
    cost = (100 * 3 + 19900 * 0.3 + 10 * 15) / 1e6
    (f,) = oversized_prompts(calls + [big], book, 1.0)
    old_claim = (20000 - 2000) * 3 / 1e6
    assert f.monthly_saving_usd < old_claim and f.monthly_saving_usd <= cost + 1e-12
    # by hand: input part = (300+5970)/1e6 = 0.00627 ; removes 18000/20000 of it
    assert f.monthly_saving_usd == pytest.approx(0.00627 * 0.9)


def test_05_truncation_saving_never_exceeds_cost(book):
    calls = [mk(i, out=1024, max_tokens=1024, stop_reason="max_tokens", feature="f") for i in range(4)]
    (f,) = truncated_outputs(calls, book, 1.0)
    assert f.monthly_saving_usd <= sum((1000 * 3 + 1024 * 15) / 1e6 for _ in calls)


# 6 ---------------------------------------------------------------------------
def test_06_only_basename_in_reports(tmp_path):
    d = tmp_path / "secret-user-dir"
    d.mkdir()
    p = _log(d, _good(), name="calls.jsonl")
    rep = build_report(str(p), load_calls(str(p)), load_prices())
    for text in (render_text(rep), render_markdown(rep), render_json(rep)):
        assert "secret-user-dir" not in text and "calls.jsonl" in text


# 7 ---------------------------------------------------------------------------
def test_07_format_chars_stripped_and_urls_not_autolinked(tmp_path):
    assert safe_text("a‮b‍c") == "abc"
    p = _log(tmp_path, _good(feature="see https://evil.example and www.evil.example‮"))
    md = render_markdown(build_report("x", load_calls(str(p)), load_prices()))
    assert "://" not in md.split("## By feature")[1].split("## By day")[0]
    assert "‮" not in md and "www." not in md.split("## By feature")[1].split("## By day")[0]


# 8 ---------------------------------------------------------------------------
def _dup_log(tmp_path):
    return _log(tmp_path, *[_good(model="gpt-4.1", feature="faq", prompt="SECRET-CUSTOMER-TEXT") for _ in range(3)])


def test_08_snippets_hidden_by_default_and_opt_in(tmp_path):
    p = _dup_log(tmp_path)
    assert "SECRET-CUSTOMER-TEXT" not in render_text(build_report("x", load_calls(str(p)), load_prices()))
    assert "prompt hash" in render_text(build_report("x", load_calls(str(p)), load_prices()))
    shown = render_text(build_report("x", load_calls(str(p)), load_prices(), show_snippets=True))
    assert "SECRET-CUSTOMER-TEXT" in shown


def test_08_cli_flag(tmp_path, capsys):
    p = _dup_log(tmp_path)
    assert main(["analyze", str(p)]) == 0
    assert "SECRET-CUSTOMER-TEXT" not in capsys.readouterr().out
    assert main(["analyze", str(p), "--show-snippets"]) == 0
    assert "SECRET-CUSTOMER-TEXT" in capsys.readouterr().out


# 9 ---------------------------------------------------------------------------
def test_09_out_must_not_be_the_input(tmp_path, capsys):
    p = _log(tmp_path, _good())
    before = p.read_text()
    assert main(["analyze", str(p), "--out", str(p)]) == 2
    link = tmp_path / "link.jsonl"
    os.symlink(str(p), str(link))
    assert main(["analyze", str(p), "--out", str(link)]) == 2
    assert p.read_text() == before


def test_09_unwritable_out_is_a_clean_error(tmp_path, capsys):
    p = _log(tmp_path, _good())
    assert main(["analyze", str(p), "--out", str(tmp_path / "no" / "such" / "dir" / "r.txt")]) == 2
    assert "cannot write" in capsys.readouterr().err
    assert main(["sample", "--out", str(tmp_path / "no" / "dir" / "s.jsonl")]) == 2


# 10 --------------------------------------------------------------------------
def test_10_bom_does_not_drop_first_line(tmp_path):
    p = tmp_path / "bom.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(_good()).encode() + b"\n" + json.dumps(_good()).encode() + b"\n")
    r = load_calls(str(p))
    assert len(r.calls) == 2 and r.malformed == 0


# 11 --------------------------------------------------------------------------
def test_11_span_counts_calendar_days(book):
    # 14 days of traffic from 09:00 on day 1 to 17:00 on day 14: elapsed time is 13.3 days
    calls = [mk(i, secs=-3 * 3600 + d * 86400 + h * 3600) for i, (d, h) in
             enumerate((d, h) for d in range(14) for h in (0, 8))]
    s = spend_breakdown(calls, book)
    assert s.span_days == 14 and s.monthly_factor == pytest.approx(30 / 14)


# 12 --------------------------------------------------------------------------
@pytest.mark.parametrize("limit", [0, None])
def test_12_zero_or_missing_max_tokens_is_not_truncation(limit):
    assert not mk(out=500, max_tokens=limit).truncated
    assert mk(out=500, max_tokens=500).truncated
