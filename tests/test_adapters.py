import json

import pytest

from llm_cost_audit.records import load_calls, normalise, safe_text


def test_canonical():
    c = normalise({"timestamp": "2026-01-02T03:04:05Z", "model": "m", "input_tokens": 10, "output_tokens": 5,
                   "cached_input_tokens": 3, "feature": "f", "stop_reason": "max_tokens"})
    assert (c.input_tokens, c.output_tokens, c.cached_input_tokens, c.feature) == (10, 5, 3, "f")
    assert c.truncated
    assert c.timestamp.year == 2026 and c.timestamp.tzinfo is not None
    assert c.total_input_tokens == 13


def test_anthropic_shape():
    c = normalise({"id": "msg_1", "type": "message", "model": "claude-x", "stop_reason": "end_turn",
                   "usage": {"input_tokens": 100, "output_tokens": 20,
                             "cache_read_input_tokens": 500, "cache_creation_input_tokens": 50}})
    assert (c.input_tokens, c.cached_input_tokens, c.cache_write_tokens, c.output_tokens) == (100, 500, 50, 20)
    assert c.request_id == "msg_1"
    assert c.total_input_tokens == 650


def test_openai_shape_subtracts_cached_from_prompt_tokens():
    c = normalise({"id": "chatcmpl-1", "model": "gpt-x", "created": 1767268800,
                   "choices": [{"finish_reason": "length"}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 30,
                             "prompt_tokens_details": {"cached_tokens": 800}}})
    assert (c.input_tokens, c.cached_input_tokens, c.output_tokens) == (200, 800, 30)
    assert c.truncated
    assert c.timestamp is not None


def test_truncated_by_max_tokens_field():
    c = normalise({"model": "m", "input_tokens": 1, "output_tokens": 256, "max_tokens": 256})
    assert c.truncated


@pytest.mark.parametrize("bad", [
    [1, 2], {"model": "m"}, {"model": "m", "input_tokens": "10", "output_tokens": 1},
    {"model": "m", "input_tokens": -1, "output_tokens": 1}, {"input_tokens": 1, "output_tokens": 1},
    {"model": "m", "input_tokens": True, "output_tokens": 1},
    {"model": "g", "usage": {"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 9}}},
])
def test_bad_records_raise_valueerror(bad):
    with pytest.raises(ValueError):
        normalise(bad)


def test_malformed_lines_are_skipped_and_counted(tmp_path):
    good = json.dumps({"model": "m", "input_tokens": 1, "output_tokens": 1})
    p = tmp_path / "log.jsonl"
    p.write_bytes(("\n".join([good, "not json", "[1]", "", '{"model":"m","input_tokens":"x"}', good]) + "\n").encode()
                  + b"\xff\xfe{bad utf8\n")
    r = load_calls(str(p))
    assert len(r.calls) == 2
    assert r.malformed == 4
    assert r.lines_read == 6  # the blank line is not counted
    assert r.malformed_examples[0][0] == 2


def test_overlong_line_is_rejected_without_crashing(tmp_path):
    good = json.dumps({"model": "m", "input_tokens": 1, "output_tokens": 1})
    p = tmp_path / "log.jsonl"
    p.write_text(good + "\n" + "x" * 1_500_000 + "\n" + good + "\n")
    r = load_calls(str(p))
    assert len(r.calls) == 2 and r.malformed == 1


def test_untrusted_text_is_sanitised():
    c = normalise({"model": "m", "input_tokens": 1, "output_tokens": 1, "feature": "a\x1b[31mred\x00" + "z" * 500})
    assert "\x1b" not in c.feature and "\x00" not in c.feature and len(c.feature) <= 80
    assert safe_text("a\nb") == "a b"


def test_missing_timestamp_is_kept_but_counted(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"model": "m", "input_tokens": 1, "output_tokens": 1, "timestamp": "garbage"}) + "\n")
    r = load_calls(str(p))
    assert len(r.calls) == 1 and r.undated == 1


def test_anthropic_one_hour_writes_are_kept():
    c = normalise({"model": "claude-x",
                   "usage": {"input_tokens": 2, "output_tokens": 10, "cache_read_input_tokens": 0,
                             "cache_creation_input_tokens": 500,
                             "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 300}}})
    assert (c.cache_write_tokens, c.cache_write_1h_tokens) == (500, 300)
    assert c.total_input_tokens == 502  # the 1-hour part is inside the 500, not added on top


def test_one_hour_writes_cannot_exceed_total_writes():
    with pytest.raises(ValueError):
        normalise({"model": "m", "input_tokens": 1, "output_tokens": 1,
                   "cache_write_tokens": 100, "cache_write_1h_tokens": 101})
