"""Command line entry point: `llm-cost-audit analyze` and `llm-cost-audit sample`.

Reads the files you name and writes the files you name. No network, ever.
"""

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .prices import load_prices
from .records import load_calls
from .report import RENDERERS, build_report
from .sample import generate


def _analyze(args: argparse.Namespace) -> int:
    try:
        prices = load_prices(args.prices)
        loaded = load_calls(args.log)
    except (OSError, ValueError) as exc:  # ValueError includes bad JSON in prices
        print("error: %s" % exc, file=sys.stderr)
        return 2
    if not loaded.calls:
        print("error: no valid calls found (%d lines read, %d malformed)" % (loaded.lines_read, loaded.malformed),
              file=sys.stderr)
        return 2
    text = RENDERERS[args.format](build_report(args.log, loaded, prices))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s" % args.out, file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def _sample(args: argparse.Namespace) -> int:
    if args.calls < 1:
        print("error: --calls must be at least 1", file=sys.stderr)
        return 2
    lines = generate(args.calls, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %d lines of synthetic data to %s" % (len(lines), args.out), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-cost-audit", description="Audit LLM call logs for avoidable spend.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a JSONL call log")
    a.add_argument("log", help="path to the JSONL log")
    a.add_argument("--format", choices=sorted(RENDERERS), default="text")
    a.add_argument("--out", help="write the report to this file instead of stdout")
    a.add_argument("--prices", help="prices JSON file that replaces the shipped one")
    a.set_defaults(func=_analyze)

    s = sub.add_parser("sample", help="generate a synthetic demo log")
    s.add_argument("--out", required=True, help="where to write the JSONL file")
    s.add_argument("--calls", type=int, default=2000)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=_sample)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
