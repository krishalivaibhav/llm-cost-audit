"""Price table and the cost formula.

Prices are USD per MILLION tokens, stored in prices.json with the source URL
and the date they were checked. We never guess: a model that is not in the
table is "unpriced" and is excluded from dollar figures.
"""

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Dict, Optional

from .records import Call

_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
# Providers return dated names like "gpt-4o-2024-08-06"; the table lists the
# base name. We retry once with a trailing date removed.
_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


@dataclass
class Price:
    model: str
    provider: str
    input: float
    output: float
    cache_read: float
    cache_write: float
    source_url: str
    as_of: str
    note: str = ""


class PriceBook:
    def __init__(self, entries: Dict[str, Price]):
        self.entries = entries

    def get(self, model: str) -> Optional[Price]:
        """Exact match first, then with a trailing date suffix removed.
        Anything else is None (unpriced), never a fuzzy guess."""
        if model in self.entries:
            return self.entries[model]
        stripped = _DATE_SUFFIX.sub("", model)
        return self.entries.get(stripped)

    def cheapest(self, provider: str) -> Optional[Price]:
        """Cheapest listed model of a provider, ranked by input price plus
        output price. A crude but explainable proxy: it ignores your own
        input/output mix."""
        same = [p for p in self.entries.values() if p.provider == provider]
        return min(same, key=lambda p: p.input + p.output) if same else None


def parse_prices(data: dict) -> PriceBook:
    """Validate a decoded prices document. Raises ValueError on bad shape."""
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        raise ValueError('prices file must contain a "models" object')
    entries: Dict[str, Price] = {}
    for name, e in models.items():
        if not isinstance(e, dict):
            raise ValueError("price entry for %r is not an object" % name)
        for f in _PRICE_FIELDS:
            v = e.get(f)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError("%s.%s must be a non-negative number" % (name, f))
        entries[name] = Price(
            model=name,
            provider=str(e.get("provider", "unknown")),
            input=float(e["input"]),
            output=float(e["output"]),
            cache_read=float(e["cache_read"]),
            cache_write=float(e["cache_write"]),
            source_url=str(e.get("source_url", "")),
            as_of=str(e.get("as_of", "")),
            note=str(e.get("note", "")),
        )
    return PriceBook(entries)


def load_prices(path: Optional[str] = None) -> PriceBook:
    """Load the shipped table, or the file given with --prices."""
    if path is None:
        text = resources.files("llm_cost_audit").joinpath("prices.json").read_text("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    return parse_prices(json.loads(text))


def call_cost(call: Call, price: Price) -> float:
    """Exact cost of one call in USD.

        cost = ( input_tokens        * input
               + cached_input_tokens * cache_read
               + cache_write_tokens  * cache_write
               + output_tokens       * output ) / 1,000,000

    Reasoning/thinking tokens are assumed to be already inside output_tokens,
    which is how both providers report them.
    """
    return (
        call.input_tokens * price.input
        + call.cached_input_tokens * price.cache_read
        + call.cache_write_tokens * price.cache_write
        + call.output_tokens * price.output
    ) / 1_000_000
