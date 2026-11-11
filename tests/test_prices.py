import json

import pytest

from llm_cost_audit.prices import call_cost, load_prices, parse_prices
from tests.conftest import mk


def test_cost_formula_by_hand(book):
    # (1000*3 + 500*15 + 2000*0.3 + 400*3.75) / 1e6 = (3000+7500+600+1500)/1e6
    c = mk(inp=1000, out=500, cached_input_tokens=2000, cache_write_tokens=400)
    assert call_cost(c, book.get("big")) == pytest.approx(0.0126)


def test_date_suffix_is_stripped_but_no_fuzzy_guess(book):
    assert book.get("big-2026-01-01").model == "big"
    assert book.get("big-20260101").model == "big"
    assert book.get("bigger") is None
    assert book.get("unknown-model") is None


def test_cheapest_is_per_provider(book):
    assert book.cheapest("p").model == "small"
    assert book.cheapest("q").model == "other"
    assert book.cheapest("nobody") is None


def test_invalid_prices_rejected():
    with pytest.raises(ValueError):
        parse_prices({"models": {"m": {"input": -1, "output": 1, "cache_read": 0, "cache_write": 0}}})
    with pytest.raises(ValueError):
        parse_prices({"nope": 1})


def test_shipped_prices_are_complete():
    book = load_prices()
    assert book.entries
    for p in book.entries.values():
        assert p.source_url.startswith("https://") and p.as_of


def test_prices_override_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"models": {"x": {"input": 1, "output": 2, "cache_read": 0.1, "cache_write": 1}}}))
    assert load_prices(str(f)).get("x").output == 2


def test_one_hour_writes_use_their_own_price():
    # 'big' has cache_write 3.75 and no 1-hour price; add one at 6.0 (2x input).
    entry = {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75, "cache_write_1h": 6}
    price = parse_prices({"models": {"m": entry}}).get("m")
    # 400 written: 100 at the 5-minute rate, 300 at the 1-hour rate.
    # (100*3.75 + 300*6) / 1e6 = (375 + 1800) / 1e6
    c = mk(inp=0, out=0, cache_write_tokens=400, cache_write_1h_tokens=300)
    assert call_cost(c, price) == pytest.approx(0.002175)


def test_missing_one_hour_price_falls_back_to_write_price(book):
    c = mk(inp=0, out=0, cache_write_tokens=400, cache_write_1h_tokens=300)
    assert call_cost(c, book.get("big")) == pytest.approx(400 * 3.75 / 1e6)


def test_invalid_one_hour_price_rejected():
    with pytest.raises(ValueError):
        parse_prices({"models": {"m": {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0,
                                       "cache_write_1h": "2"}}})


def test_shipped_anthropic_one_hour_price_is_twice_input():
    # The pricing page lists every Claude model's 1-hour write at exactly 2x input.
    for p in load_prices().entries.values():
        if p.provider == "anthropic":
            assert p.cache_write_1h == 2 * p.input, p.model
