"""Minded CLI: fully autonomous local data analysis."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set

from minded.analyze import SUPPORTED_SUFFIXES, run_analysis


def _print_result(result: dict) -> None:
    print(json.dumps(result, indent=2, default=str))


def _collect_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(
                sorted(
                    f
                    for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        else:
            files.append(p)
    return files


def cmd_analyze(args: argparse.Namespace) -> int:
    files = _collect_files(args.paths)
    if not files:
        print("No supported data files found.", file=sys.stderr)
        return 2
    result = run_analysis(
        files,
        question=args.question,
        project_id=args.project_id,
        out_dir=Path(args.out),
        analysis_mode="AI_AUGMENTED" if args.ai else "DETERMINISTIC",
    )
    _print_result(result)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    inbox = Path(args.inbox).expanduser().resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    processed_dir = inbox / "_processed"
    processed_dir.mkdir(exist_ok=True)
    seen: Set[str] = set()
    print(f"Watching {inbox} for {sorted(SUPPORTED_SUFFIXES)} (Ctrl+C to stop)", file=sys.stderr)
    while True:
        for path in sorted(inbox.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            key = f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
            if key in seen:
                continue
            try:
                result = run_analysis(
                    [path],
                    question=args.question,
                    project_id=args.project_id,
                    out_dir=Path(args.out),
                    analysis_mode="AI_AUGMENTED" if args.ai else "DETERMINISTIC",
                )
                _print_result(result)
                dest = processed_dir / path.name
                if dest.exists():
                    dest = processed_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
                path.replace(dest)
                seen.add(key)
            except Exception as exc:  # noqa: BLE001 — keep the watcher alive
                print(f"Failed on {path}: {exc}", file=sys.stderr)
        time.sleep(args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minded",
        description="Fully autonomous data analyzer: ingest, profile, investigate, report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Ingest files and run an unattended investigation")
    analyze.add_argument("paths", nargs="+", help="Data files or a folder of files")
    analyze.add_argument("-q", "--question", default=None, help="Optional investigation question")
    analyze.add_argument("--project-id", default="proj-minded")
    analyze.add_argument("--out", default="reports", help="Report output directory")
    analyze.add_argument("--ai", action="store_true", help="Optional AI narrative after deterministic analysis")
    analyze.set_defaults(func=cmd_analyze)

    watch = sub.add_parser("watch", help="Watch a folder and analyze new files automatically")
    watch.add_argument("inbox", help="Folder to watch")
    watch.add_argument("-q", "--question", default=None)
    watch.add_argument("--project-id", default="proj-minded")
    watch.add_argument("--out", default="reports")
    watch.add_argument("--ai", action="store_true")
    watch.add_argument("--interval", type=float, default=5.0, help="Poll seconds")
    watch.set_defaults(func=cmd_watch)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
