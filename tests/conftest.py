"""Shared fixtures. The test price book uses round numbers so expected costs
can be checked by hand."""
from datetime import datetime, timedelta, timezone

import pytest

from llm_cost_audit.prices import parse_prices
from llm_cost_audit.records import Call

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def book():
    def entry(provider, i, o, r, w):
        return {"provider": provider, "input": i, "output": o, "cache_read": r, "cache_write": w,
                "source_url": "https://example.test", "as_of": "2026-01-01"}
    return parse_prices({"models": {
        "big": entry("p", 3, 15, 0.3, 3.75),
        "small": entry("p", 0.5, 2, 0.05, 0.6),
        "other": entry("q", 1, 1, 0.1, 1),
    }})


def mk(i=0, model="big", inp=1000, out=100, secs=0, **kw):
    """Build a Call; `secs` is the offset from T0 in seconds."""
    kw.setdefault("timestamp", T0 + timedelta(seconds=secs))
    return Call(line_no=i, model=model, input_tokens=inp, output_tokens=out, **kw)
