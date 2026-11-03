"""Command-line entry point.

uv run python -m agent_eval_gate run    --model claude-opus-5 --k 5
uv run python -m agent_eval_gate report runs/<dir>/results.json
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_eval_gate.report import render
from agent_eval_gate.runner import DEFAULT_CONCURRENCY, DEFAULT_K, load_suite, run_suite
from agent_eval_gate.scenarios.loader import load_scenarios


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()
    scenarios = load_scenarios(Path(args.scenarios) if args.scenarios else None)
    if args.only:
        wanted = set(args.only)
        scenarios = [s for s in scenarios if s.id in wanted]
    if not scenarios:
        print("no scenarios selected", file=sys.stderr)
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{stamp}-{args.model}"
    judge = None if args.no_judge else args.judge

    print(
        f"{len(scenarios)} scenarios × k={args.k} on {args.model}"
        f" (judge: {judge or 'off'}) → {out_dir}",
        file=sys.stderr,
    )

    def progress(outcome):
        mark = "." if outcome.passed else ("E" if outcome.error else "F")
        print(mark, end="", flush=True, file=sys.stderr)

    suite = asyncio.run(
        run_suite(
            scenarios,
            model=args.model,
            judge_model=judge,
            k=args.k,
            out_dir=out_dir,
            concurrency=args.concurrency,
            on_outcome=progress,
        )
    )
    print(file=sys.stderr)
    report = render(suite)
    (out_dir / "report.md").write_text(report + "\n")
    print(report)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print(render(load_suite(Path(args.results))))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_eval_gate")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the suite against a model")
    run.add_argument("--model", default="claude-opus-5")
    run.add_argument("--judge", default="claude-sonnet-5", help="judge model; must differ")
    run.add_argument("--no-judge", action="store_true", help="skip the faithfulness layer")
    run.add_argument("--k", type=int, default=DEFAULT_K)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--scenarios", help="directory of .yaml scenarios (default: examples)")
    run.add_argument("--only", nargs="*", help="scenario ids to run")
    run.add_argument("--out", help="output directory (default: runs/<stamp>-<model>)")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="re-render a saved results.json")
    rep.add_argument("results")
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
