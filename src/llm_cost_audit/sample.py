"""Generate a SYNTHETIC log that contains each problem the analyses look for.

Everything here is invented: feature names, prompts and numbers are generic.
The generator is deterministic for a given seed so demos and tests are stable.
Each feature below exists to trigger (or, for `code-review`, to NOT trigger)
one analysis:

  support-chat       big system prompt resent, no caching      -> uncached prefix
  faq-bot            small pool of repeated questions          -> exact duplicates
  doc-summarizer     a few huge prompts among normal ones      -> oversized prompts
  intent-classifier  expensive model, 3-8 token outputs        -> downgrade candidate
  report-writer      15% of calls cut off at max_tokens        -> truncated outputs
  code-review        caching used correctly (control)          -> should NOT be flagged
"""

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

START = datetime(2026, 10, 1, tzinfo=timezone.utc)  # fixed so output is reproducible
DAYS = 14

SUPPORT_SYSTEM = ("You are a friendly support assistant for a fictional software product. "
                  "Follow the policy below. " + "Be concise, polite and accurate. Never invent features. " * 100)
REVIEW_SYSTEM = ("You review small code changes for a fictional project. " + "Point out bugs and unclear names. " * 140)
FAQ_POOL = ["How do I reset my password?", "Where can I download my invoice?", "What are your opening hours?",
            "How do I change my email address?", "Can I export my data?", "How do I delete my account?",
            "Which plans include API access?", "How do I add a teammate?", "Is there a mobile app?",
            "How do I enable two-factor sign-in?", "Where is the status page?", "How do I cancel a subscription?"]

# feature -> (weight, model, bursty)
FEATURES = {
    "support-chat": (0.30, "claude-sonnet-5-5", True),
    "faq-bot": (0.15, "gpt-4.1", False),
    "doc-summarizer": (0.12, "claude-sonnet-5-5", False),
    "intent-classifier": (0.20, "claude-opus-5-5", False),
    "report-writer": (0.08, "claude-sonnet-5-5", True),
    "code-review": (0.15, "claude-sonnet-5-5", True),
}


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class _Clock:
    """Hands out timestamps. Bursty features emit sessions of several calls a
    few seconds to a couple of minutes apart (like a chat), which is what
    makes a 5-minute prompt cache useful; others are spread over working hours."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.sessions: Dict[str, list] = {}

    def _random_work_time(self) -> datetime:
        day = self.rng.randrange(DAYS)
        return START + timedelta(days=day, hours=self.rng.randrange(8, 18), seconds=self.rng.randrange(3600))

    def next(self, feature: str, bursty: bool) -> datetime:
        if not bursty:
            return self._random_work_time()
        state = self.sessions.get(feature)
        if state is None or state[1] <= 0:
            state = [self._random_work_time(), self.rng.randrange(4, 9)]  # [next time, calls left]
        t, left = state
        t_next = t + timedelta(seconds=self.rng.randrange(20, 120))
        self.sessions[feature] = [t_next, left - 1]
        return t


def _make_call(rng: random.Random, feature: str, model: str, ts: datetime) -> dict:
    """One canonical-shaped call (a dict) for the given feature."""
    c = {"timestamp": _iso(ts), "model": model, "feature": feature,
         "request_id": "req_%08x" % rng.getrandbits(32), "stop_reason": "end_turn"}
    if feature == "support-chat":
        user = "Customer question #%d about their account." % rng.randrange(10**6)
        c.update(system_prompt=SUPPORT_SYSTEM, prompt=user,
                 input_tokens=len(SUPPORT_SYSTEM) // 4 + rng.randrange(40, 300),
                 output_tokens=rng.randrange(80, 260), cached_input_tokens=0)
    elif feature == "faq-bot":
        q = rng.choice(FAQ_POOL) if rng.random() < 0.7 else "Custom question #%d" % rng.randrange(10**6)
        c.update(prompt=q, input_tokens=rng.randrange(60, 90), output_tokens=rng.randrange(90, 160))
    elif feature == "doc-summarizer":
        big = rng.random() < 0.06
        c.update(prompt="Summarise document #%d" % rng.randrange(10**6),
                 input_tokens=rng.randrange(25000, 40000) if big else rng.randrange(2500, 3500),
                 output_tokens=rng.randrange(300, 500))
    elif feature == "intent-classifier":
        c.update(prompt="Classify message #%d" % rng.randrange(10**6),
                 input_tokens=rng.randrange(150, 260), output_tokens=rng.randrange(3, 9))
    elif feature == "report-writer":
        cut = rng.random() < 0.15
        c.update(prompt="Write weekly report #%d" % rng.randrange(10**6), max_tokens=1024,
                 input_tokens=rng.randrange(1200, 1800),
                 output_tokens=1024 if cut else rng.randrange(400, 900))
        c["stop_reason"] = "max_tokens" if cut else "end_turn"
    else:  # code-review: caching used properly
        first = rng.random() < 0.15  # occasional cache write when the entry expired
        prefix = len(REVIEW_SYSTEM) // 4
        c.update(system_prompt=REVIEW_SYSTEM, prompt="Review diff #%d" % rng.randrange(10**6),
                 input_tokens=rng.randrange(200, 600), output_tokens=rng.randrange(150, 400),
                 cached_input_tokens=0 if first else prefix, cache_write_tokens=prefix if first else 0)
    return c


def _to_raw(rng: random.Random, c: dict) -> dict:
    """Re-express a canonical call as a raw Anthropic or OpenAI response record
    (plus the extra tag fields our adapters accept next to it)."""
    extras = {k: c[k] for k in ("timestamp", "feature", "prompt", "system_prompt", "max_tokens") if k in c}
    if c["model"].startswith("claude"):
        usage = {"input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"],
                 "cache_read_input_tokens": c.get("cached_input_tokens", 0),
                 "cache_creation_input_tokens": c.get("cache_write_tokens", 0)}
        return dict(id="msg_%08x" % rng.getrandbits(32), type="message", model=c["model"],
                    stop_reason=c["stop_reason"], usage=usage, **extras)
    cached = c.get("cached_input_tokens", 0)
    usage = {"prompt_tokens": c["input_tokens"] + cached, "completion_tokens": c["output_tokens"],
             "prompt_tokens_details": {"cached_tokens": cached}}
    finish = "length" if c["stop_reason"] == "max_tokens" else "stop"
    return dict(id="chatcmpl-%08x" % rng.getrandbits(32), object="chat.completion", model=c["model"],
                choices=[{"index": 0, "finish_reason": finish}], usage=usage, **extras)


def generate(calls: int = 2000, seed: int = 1) -> List[str]:
    """Return JSONL lines: ~half canonical, ~half raw provider shapes, plus two
    deliberately malformed lines so the skip-and-count path is visible."""
    rng = random.Random(seed)
    clock = _Clock(rng)
    names = list(FEATURES)
    weights = [FEATURES[n][0] for n in names]
    records = []
    for _ in range(calls):
        feature = rng.choices(names, weights)[0]
        _, model, bursty = FEATURES[feature]
        c = _make_call(rng, feature, model, clock.next(feature, bursty))
        records.append(_to_raw(rng, c) if rng.random() < 0.5 else c)
    records.sort(key=lambda r: r["timestamp"])
    lines = [json.dumps(r, separators=(",", ":")) for r in records]
    if calls >= 10:
        lines.insert(calls // 3, '{"model": "claude-sonnet-5-5", "usage": ')  # truncated JSON
        lines.insert(calls // 2, '{"hello": "not an llm call"}')
    return lines
