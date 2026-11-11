# llm-cost-audit

A small command-line tool that reads a log of LLM API calls and reports where
the money goes: spend by model, feature and day, plus a ranked list of
possible savings. Each saving comes with the evidence behind it, the
assumptions it rests on, and a confidence label.

It reads the files you point it at and writes a report. It makes no network
calls and sends no telemetry. Runtime needs only the Python 3.9+ standard
library.

The numbers are estimates meant to be checked by a person. They are leads, not
guarantees.

## Quickstart (about a minute)

```bash
git clone <this repo> && cd llm-cost-audit
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

llm-cost-audit sample --out sample.jsonl          # synthetic demo log, 2000 calls
llm-cost-audit analyze sample.jsonl               # text report
llm-cost-audit analyze sample.jsonl --format markdown --out report.md
llm-cost-audit analyze sample.jsonl --format json --out report.json
```

Run the tests with `pip install pytest && pytest`.

Options: `analyze LOG [--format text|markdown|json] [--out FILE] [--prices FILE] [--show-snippets]` (`--out` may not be the log file itself);
`sample --out FILE [--calls N] [--seed S]`. The sample data is entirely
synthetic (generic feature names, invented prompts) and deterministic for a
given seed.

## Input format

JSONL: one call per line. Blank lines are ignored. A leading UTF-8 byte-order mark is accepted. Lines that are not valid
JSON, are not a recognised shape, have bad token fields (negative, non-numeric, or above 10^12), or are longer than
1 MB are skipped and counted in the report; they never stop the run.

### Canonical shape

| Field | Required | Meaning |
|---|---|---|
| `model` | yes | Model name as sent to the API |
| `input_tokens` | yes | Prompt tokens billed at the full input rate |
| `output_tokens` | yes | Output tokens (include reasoning tokens if your provider bills them as output) |
| `timestamp` | no | ISO 8601. Without it a call is excluded from the day breakdown and the time span |
| `cached_input_tokens` | no | Prompt tokens read from a cache |
| `cache_write_tokens` | no | Prompt tokens written to a cache |
| `feature` | no | Your own tag (which part of the product made the call). Default `(untagged)` |
| `prompt`, `prompt_hash` | no | Prompt text, or a hash of it, used to find exact duplicates |
| `system_prompt` | no | Used to find repeated prefixes |
| `max_tokens`, `stop_reason` | no | Used to find truncated outputs (`max_tokens` or `length`) |
| `latency_ms`, `request_id` | no | Kept, not currently analysed |

**Important:** `input_tokens`, `cached_input_tokens` and `cache_write_tokens`
must not overlap. Total prompt size is their sum.

### Raw provider shapes (auto-detected per line)

- Anthropic Messages response: `usage.input_tokens`, `usage.output_tokens`,
  `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens`,
  `stop_reason`. These map one-to-one onto the canonical fields.
- OpenAI chat completion: `usage.prompt_tokens`, `usage.completion_tokens`,
  `usage.prompt_tokens_details.cached_tokens`, `choices[0].finish_reason`.
  OpenAI's `prompt_tokens` includes cached tokens, so the adapter subtracts
  them to get `input_tokens`.

Raw responses do not contain your feature tag or prompt, so add `feature`,
`prompt`, `system_prompt`, `max_tokens` and `timestamp` as extra top-level
keys next to the raw response if you want those analyses. Timestamps are
taken from `timestamp`, or `created_at` (Anthropic) / `created` (OpenAI unix
seconds).

Log content is treated as untrusted: nothing is evaluated or executed, no
path is read from the log, free-text fields are stripped of control and
Unicode format characters (such as bidi overrides) and length-capped, and
Markdown output escapes HTML and Markdown syntax and defuses bare URLs
(`://` and `www.` are written as HTML entities so viewers do not auto-link
them). Reports name the log by file name only, never its full path, and show
prompt hashes rather than prompt text unless you pass `--show-snippets`.

## Prices

`src/llm_cost_audit/prices.json` holds USD per million tokens for input,
output, cache read and cache write, plus `source_url` and `as_of` for each
model. Cost of a call:

```
(input_tokens*input + cached_input_tokens*cache_read
 + cache_write_tokens*cache_write + output_tokens*output) / 1,000,000
```

- Models not in the table are reported as **unpriced** and left out of every
  dollar figure. Dated names like `gpt-4o-2024-08-06` are matched to the base
  name by removing the date suffix; nothing else is fuzzy-matched.
- OpenAI's GPT-6 and GPT-5.6 models list a 30-minute cache-write price
  (1.25x input), which the table uses. The cache replay below assumes a
  5-minute lifetime, so for these models it counts more writes than a
  30-minute cache would, and the caching saving comes out on the low side.
  Older OpenAI models list no separate cache-write fee, so `cache_write`
  equals `input` there.
- Claude Haiku 5.5 has a higher price tier for prompts over 100,000 tokens.
  The table holds the lower tier only.
- Only models whose prices and API IDs were checked on the provider pages are
  included (see `as_of`). Older or other models are absent on purpose. Add
  them with `--prices FILE`, a JSON file shaped like the shipped one, which
  replaces the shipped table.
- Prices change. Check the `source_url` before relying on a report.

## What each estimate means, and its limits

Savings are measured over the log's period and scaled to 30 days:
`monthly = saving_in_log * 30 / span_days`. `span_days` is the number of UTC
calendar days covered, counting both end days. Timestamps separated from the
rest of the log by a long gap (more than a week, or five times the spread of
the middle half of the data) are left out of the span, so one corrupt
timestamp (say 0, i.e. 1970) cannot stretch it; the report warns when
timestamps were ignored. Calendar
days are used because first-to-last-call time understates the period (14
days of traffic reads as about 13.4), which inflates projections. The
opposite bias exists for short logs: 26 hours spanning two dates count as 2
days, so the projection is understated. A single-day log is treated as one
full day and the report warns. All this assumes the log is representative of
a normal month (no growth, no weekday/weekend pattern). Every saving is
capped at the actual cost of the calls it involves. Savings from different findings overlap, so do not add them up.

**Spend breakdown.** Plain sums of the formula above by model, feature and
UTC day. Projection is straight-line.

**Uncached repeated prefix** (confidence: medium, low if timestamps are
missing). Groups calls with the same model and the same `system_prompt` (or,
without one, the same first 4096 characters of a prompt at least that long), at least 5
calls, none reporting cached tokens. Prefix size is estimated as characters
divided by 4 (capped at the smallest call's input; a bare-prompt prefix is
4096 chars, i.e. 1024 tokens), and must be at least 1024
tokens. The calls are replayed in time order: the first call, and any call more
than 300 seconds after the previous one, pays the cache-write price on the
prefix; the rest pay the cache-read price. Saving is current prefix cost minus
that. Limits: the token estimate is rough; the 300 s lifetime is Anthropic's
default and other providers or settings differ; minimum cacheable sizes vary
by model; it assumes the shared text can be moved to the front of the prompt.
If caching would cost more than it saves, no finding is produced.

**Exact duplicates** (medium). Same model, same system prompt, same prompt
(or `prompt_hash`) seen more than once. Evidence shows the prompt hash (`--show-snippets` adds
the first 80 characters). Saving is the full cost of every
repeat after the first. Limits: only valid if an identical prompt should get
the same answer; it ignores cache expiry and the cost of running a cache.

**Oversized prompts** (low). Needs 20+ calls in a feature. (a) Calls above 4x
the feature's own median input are flagged; saving assumes they could be
trimmed to 2x the median, with removed tokens spread pro rata over the call's full-rate, cached and cache-write input, each at its own price. (b) Features whose
median input:output ratio is at least 100 with median input of at least 2000
tokens are flagged; the saving is an illustrative 25% of that feature's input
spend, a placeholder and not a measurement. All thresholds are constants at
the top of `analyses.py`.

**Model downgrade candidates** (low; needs a quality eval before switching).
A feature and model with 20+ calls where 95% of outputs are 100 tokens or
fewer. Saving is the same tokens priced on the cheapest model of the same
provider (lowest input price plus output price in the table) minus the current
cost. Short outputs say nothing about whether a smaller model answers
correctly; treat this as a prompt to run an evaluation.

**Truncated outputs** (low). Calls with `stop_reason` of `max_tokens`/`length`,
or `output_tokens >= max_tokens`; features with at least 3. Saving assumes 50%
of truncated calls are retried at the same cost. The log does not show
retries, so that share is a guess.

## What this tool does not do

- It does not call any API, fetch prices, or send data anywhere.
- It does not measure answer quality, so it cannot say whether a cheaper model
  or shorter prompt is good enough.
- It does not see your application: retries, intentional re-sampling, and what
  was done with truncated output are not in the log.
- It does not model batch discounts, long-context price tiers, fine-tuned
  models, image/audio tokens, or tool-use surcharges.
- It does not read prompt text beyond hashing it and keeping an 80-character
  snippet in memory, which is only shown with `--show-snippets`.
- It does not guarantee any saving. Estimates can be wrong in either
  direction.

## Layout

```
src/llm_cost_audit/
  records.py    parse lines, adapters, untrusted-input handling
  prices.py     price table and cost formula
  analyses.py   the five analyses, one function each
  report.py     spend breakdown, report assembly, text/markdown/json output
  sample.py     synthetic log generator
  cli.py        command line
tests/
```

MIT licensed. Copyright (c) 2026 Vaibhav Krishali.
