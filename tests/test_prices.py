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
