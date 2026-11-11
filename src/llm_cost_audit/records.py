"""Reading a JSONL log and turning each line into one canonical `Call`.

Log content is untrusted. Rules this module follows:
  * lines are parsed with json.loads only (no eval, no shell, no file paths
    taken from the log);
  * each line is capped at MAX_LINE_BYTES so one huge line cannot eat memory;
  * a bad line is skipped and counted, never fatal;
  * every free-text field we keep is stripped of control characters and
    length-capped (`safe_text`), so it cannot smuggle terminal escape codes
    into the report.

Token convention (the single most common source of wrong cost numbers):
  input_tokens        = prompt tokens billed at the FULL input rate.
  cached_input_tokens = prompt tokens read from a prompt cache (cheap rate).
  cache_write_tokens  = prompt tokens written to a cache (write rate).
These three are disjoint, so total prompt size is their sum.
  cache_write_1h_tokens = the part of cache_write_tokens written with a
                        1-hour lifetime. Anthropic bills those at 2x input
                        instead of 1.25x, so they need their own price.
                        It is a SUBSET of cache_write_tokens, not a fourth
                        disjoint count. Anthropic reports
them disjoint already. OpenAI's `prompt_tokens` INCLUDES cached tokens, so the
OpenAI adapter subtracts them.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

MAX_LINE_BYTES = 1_000_000  # 1 MB. Real call records are far smaller.
MAX_TEXT_CHARS = 80  # cap for tags/names/snippets kept for display
PREFIX_CHARS = 4096  # how much of a bare prompt counts as its "prefix" (4096 chars ~ 1024 tokens,
# the smallest prefix the prefix analysis will consider)
MAX_TOKENS_VALUE = 10 ** 12  # sanity cap: no real call has a trillion tokens; bigger is corrupt
# data, and unbounded numbers would overflow float maths or print as Infinity.

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def safe_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Make a log-supplied string safe to print: no control chars, capped.

    WHY: a log line is attacker-controllable. An ESC byte in a `feature` tag
    could rewrite the auditor's terminal; a megabyte tag could wreck a table.
    """
    text = strip_invisible(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def strip_invisible(value: Any) -> str:
    """Replace control characters and drop Unicode format characters (category
    Cf: zero-width joiners, bidi overrides such as U+202E). Bidi overrides can
    make text display reversed, e.g. disguising one name as another."""
    text = _CONTROL_CHARS.sub(" ", str(value))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


@dataclass
class Call:
    """One LLM call in canonical form. Prompt text itself is NOT kept; we keep
    hashes and a short snippet, which is all the analyses need."""

    line_no: int
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: Optional[datetime] = None
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0  # subset of cache_write_tokens
    feature: str = "(untagged)"
    prompt_key: Optional[str] = None  # identity of system+prompt, for duplicates
    prefix_key: Optional[str] = None  # identity of the shared leading text
    prefix_chars: int = 0  # length of that leading text
    snippet: str = ""  # short, sanitised prompt start for evidence lines
    max_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    request_id: Optional[str] = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.cache_write_tokens

    @property
    def truncated(self) -> bool:
        """True if the call stopped because it hit its output limit."""
        if self.stop_reason in ("max_tokens", "length"):
            return True
        # A limit of 0 (or no limit recorded) means "unknown", not "everything truncated".
        return bool(self.max_tokens) and self.output_tokens >= self.max_tokens


@dataclass
class LoadResult:
    calls: List[Call]
    lines_read: int  # non-blank lines
    malformed: int
    malformed_examples: List[Tuple[int, str]]  # (line number, reason), first few
    undated: int  # calls with a missing/unparseable timestamp


# ---------------------------------------------------------------- parsing helpers


def _as_int(value: Any, name: str) -> int:
    """Non-negative integer or ValueError. Rejects bools and strings: a log that
    says "1000" (string) is probably not the shape we think it is."""
    if isinstance(value, bool):
        raise ValueError("%s must be a number" % name)
    if isinstance(value, float) and value.is_integer():  # False for inf/NaN
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    if value > MAX_TOKENS_VALUE:
        raise ValueError("%s is implausibly large (> %d)" % (name, MAX_TOKENS_VALUE))
    return value


def _opt_int(value: Any, name: str) -> Optional[int]:
    return None if value is None else _as_int(value, name)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """ISO 8601 string or unix seconds -> aware UTC datetime; None if unusable.
    Python 3.9's fromisoformat rejects a trailing 'Z', so we rewrite it."""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _text_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt-identity fields shared by all three input shapes.

    prompt_key  : hash of system prompt + prompt (exact-duplicate detection).
    prefix_key  : hash of the system prompt if there is one, else of the first
                  PREFIX_CHARS of a long prompt. Calls sharing it share a
                  leading chunk that prompt caching could reuse.
    """
    prompt = obj.get("prompt")
    system = obj.get("system_prompt")
    prompt = prompt if isinstance(prompt, str) else None
    system = system if isinstance(system, str) and system else None
    prompt_hash = obj.get("prompt_hash")
    prompt_hash = prompt_hash if isinstance(prompt_hash, str) and prompt_hash else None

    out: Dict[str, Any] = {}
    if prompt is not None:
        out["prompt_key"] = _hash(system or "", prompt)
        out["snippet"] = safe_text(prompt)
    elif prompt_hash is not None:
        out["prompt_key"] = _hash(system or "", "hash:" + prompt_hash)
    if system is not None:
        out["prefix_key"] = _hash("sys", system)
        out["prefix_chars"] = len(system)
    elif prompt is not None and len(prompt) >= PREFIX_CHARS:
        out["prefix_key"] = _hash("prompt", prompt[:PREFIX_CHARS])
        out["prefix_chars"] = PREFIX_CHARS
    return out


def _common(obj: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    """Fields every shape supports as optional extras next to the raw record."""
    model = obj.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("missing model")
    feature = obj.get("feature")
    common = {
        "line_no": line_no,
        "model": safe_text(model.strip(), 120),
        "feature": safe_text(feature) if isinstance(feature, str) and feature.strip() else "(untagged)",
        "max_tokens": _opt_int(obj.get("max_tokens"), "max_tokens"),
        "latency_ms": _opt_int(obj.get("latency_ms"), "latency_ms"),
    }
    rid = obj.get("request_id", obj.get("id"))
    common["request_id"] = safe_text(rid) if isinstance(rid, str) else None
    common.update(_text_fields(obj))
    return common


# ---------------------------------------------------------------- the three adapters


def _from_canonical(obj: Dict[str, Any], line_no: int) -> Call:
    stop = obj.get("stop_reason")
    return Call(
        timestamp=parse_timestamp(obj.get("timestamp")),
        input_tokens=_as_int(obj["input_tokens"], "input_tokens"),
        output_tokens=_as_int(obj.get("output_tokens"), "output_tokens"),
        cached_input_tokens=_as_int(obj.get("cached_input_tokens") or 0, "cached_input_tokens"),
        cache_write_tokens=_as_int(obj.get("cache_write_tokens") or 0, "cache_write_tokens"),
        cache_write_1h_tokens=_as_int(obj.get("cache_write_1h_tokens") or 0, "cache_write_1h_tokens"),
        stop_reason=safe_text(stop, 40) if isinstance(stop, str) else None,
        **_common(obj, line_no)
    )


def _from_anthropic(obj: Dict[str, Any], line_no: int) -> Call:
    """Anthropic Messages response: usage.input_tokens already EXCLUDES cache
    reads and cache writes, so the fields map across one-to-one.
    usage.cache_creation splits the writes by cache lifetime (5 minutes or
    1 hour); we keep the 1-hour part because it is billed at a higher rate."""
    usage = obj["usage"]
    by_ttl = usage.get("cache_creation")
    write_1h = by_ttl.get("ephemeral_1h_input_tokens") if isinstance(by_ttl, dict) else 0
    stop = obj.get("stop_reason")
    return Call(
        timestamp=parse_timestamp(obj.get("timestamp", obj.get("created_at"))),
        input_tokens=_as_int(usage["input_tokens"], "usage.input_tokens"),
        output_tokens=_as_int(usage.get("output_tokens"), "usage.output_tokens"),
        cached_input_tokens=_as_int(usage.get("cache_read_input_tokens") or 0, "cache_read_input_tokens"),
        cache_write_tokens=_as_int(usage.get("cache_creation_input_tokens") or 0, "cache_creation_input_tokens"),
        cache_write_1h_tokens=_as_int(write_1h or 0, "ephemeral_1h_input_tokens"),
        stop_reason=safe_text(stop, 40) if isinstance(stop, str) else None,
        **_common(obj, line_no)
    )


def _from_openai(obj: Dict[str, Any], line_no: int) -> Call:
    """OpenAI chat completion: prompt_tokens INCLUDES cached tokens, so we
    subtract them to get the full-rate part. finish_reason "length" is
    OpenAI's word for hitting the output limit."""
    usage = obj["usage"]
    prompt_tokens = _as_int(usage["prompt_tokens"], "usage.prompt_tokens")
    details = usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        cached = _as_int(details.get("cached_tokens") or 0, "cached_tokens")
    if cached > prompt_tokens:
        raise ValueError("cached_tokens exceeds prompt_tokens")
    stop = None
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        stop = choices[0].get("finish_reason")
    stop = obj.get("stop_reason", stop)
    return Call(
        timestamp=parse_timestamp(obj.get("timestamp", obj.get("created"))),
        input_tokens=prompt_tokens - cached,
        output_tokens=_as_int(usage.get("completion_tokens"), "usage.completion_tokens"),
        cached_input_tokens=cached,
        stop_reason=safe_text(stop, 40) if isinstance(stop, str) else None,
        **_common(obj, line_no)
    )


def normalise(obj: Any, line_no: int = 0) -> Call:
    """Auto-detect the shape of one parsed JSON object. Raises ValueError."""
    if not isinstance(obj, dict):
        raise ValueError("line is not a JSON object")
    usage = obj.get("usage")
    if isinstance(usage, dict) and "prompt_tokens" in usage:
        call = _from_openai(obj, line_no)
    elif isinstance(usage, dict) and "input_tokens" in usage and "input_tokens" not in obj:
        call = _from_anthropic(obj, line_no)
    elif "input_tokens" in obj:
        call = _from_canonical(obj, line_no)
    else:
        raise ValueError("unrecognised record shape")
    # The 1-hour writes are part of the total written; more than the total
    # would make the 5-minute remainder negative and the cost wrong.
    if call.cache_write_1h_tokens > call.cache_write_tokens:
        raise ValueError("1-hour cache-write tokens exceed total cache-write tokens")
    return call


def load_calls(path: str) -> LoadResult:
    """Read a JSONL file. Never raises on bad content, only on an unreadable
    file (the caller handles OSError)."""
    calls: List[Call] = []
    lines_read = malformed = undated = 0
    examples: List[Tuple[int, str]] = []

    def reject(line_no: int, reason: str) -> None:
        nonlocal malformed
        malformed += 1
        if len(examples) < 5:
            examples.append((line_no, reason))

    with open(path, "rb") as fh:
        line_no = 0
        while True:
            raw = fh.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_no += 1
            if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
                # Over-long line: discard the rest of it without holding it.
                while True:
                    rest = fh.readline(MAX_LINE_BYTES)
                    if not rest or rest.endswith(b"\n"):
                        break
                lines_read += 1
                reject(line_no, "line longer than %d bytes" % MAX_LINE_BYTES)
                continue
            if not raw.strip():
                continue
            lines_read += 1
            try:
                call = normalise(json.loads(raw.decode("utf-8-sig", "replace")), line_no)
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                # json.JSONDecodeError is a ValueError subclass.
                reject(line_no, safe_text(exc, 100))
                continue
            if call.timestamp is None:
                undated += 1
            calls.append(call)
    return LoadResult(calls, lines_read, malformed, examples, undated)
