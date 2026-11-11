import json

from llm_cost_audit.cli import main
from llm_cost_audit.prices import load_prices
from llm_cost_audit.records import load_calls
from llm_cost_audit.report import build_report, render_json, render_markdown, render_text
from llm_cost_audit.sample import generate


def test_sample_is_deterministic():
    assert generate(300, seed=5) == generate(300, seed=5)
    assert generate(300, seed=5) != generate(300, seed=6)


def test_end_to_end_on_sample(tmp_path, capsys):
    log = tmp_path / "sample.jsonl"
    assert main(["sample", "--out", str(log), "--calls", "2000", "--seed", "1"]) == 0
    loaded = load_calls(str(log))
    assert len(loaded.calls) == 2000 and loaded.malformed == 2

    report = build_report(str(log), loaded, load_prices())
    kinds = {f.kind for f in report.findings}
    assert kinds == {"uncached_prefix", "exact_duplicate", "oversized_prompt", "downgrade_candidate", "truncated_output"}
    flagged = " ".join(f.title for f in report.findings)
    assert "code-review" not in flagged  # the correctly-cached control feature stays clean
    savings = [f.monthly_saving_usd for f in report.findings]
    assert savings == sorted(savings, reverse=True)
    assert report.summary.priced_calls == 2000 and not report.summary.unpriced_models

    assert "FINDINGS" in render_text(report)
    assert json.loads(render_json(report))["summary"]["total_calls"] == 2000

    out = tmp_path / "r.md"
    assert main(["analyze", str(log), "--format", "markdown", "--out", str(out)]) == 0
    assert out.read_text().startswith("# LLM cost audit")

    # The 30-day scaling applies to every finding, so it is stated once, not per finding.
    for rendered in (render_text(report), render_markdown(report)):
        assert rendered.count("to a 30-day month") == 1


def test_markdown_escapes_untrusted_text(tmp_path):
    log = tmp_path / "evil.jsonl"
    rec = {"model": "gpt-4o-mini", "input_tokens": 1, "output_tokens": 1, "feature": "<script>alert(1)</script>|`x`"}
    log.write_text(json.dumps(rec) + "\n")
    md = render_markdown(build_report("x", load_calls(str(log)), load_prices()))
    assert "<script>" not in md and "&lt;script&gt;" in md


def test_cli_errors(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing.jsonl")]) == 2
    empty = tmp_path / "e.jsonl"
    empty.write_text("garbage\n")
    assert main(["analyze", str(empty)]) == 2
    bad_prices = tmp_path / "p.json"
    bad_prices.write_text("{not json")
    assert main(["analyze", str(empty), "--prices", str(bad_prices)]) == 2


def test_warns_when_one_hour_writes_have_no_price(tmp_path):
    log = tmp_path / "w.jsonl"
    rec = {"model": "gpt-4o", "input_tokens": 1, "output_tokens": 1,
           "cache_write_tokens": 100, "cache_write_1h_tokens": 100}
    log.write_text(json.dumps(rec) + "\n")
    report = build_report("x", load_calls(str(log)), load_prices())
    assert any("1-hour" in w and "gpt-4o" in w for w in report.summary.warnings)
