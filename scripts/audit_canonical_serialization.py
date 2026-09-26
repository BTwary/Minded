"""AST audit for unsafe canonical serialization.

Fails on json.dumps(..., default=str) inside an identity-bearing function unless
that individual call has a same-line exemption:
    # CANONICAL_SERIALIZATION_EXEMPT: <specific reason>
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, NamedTuple

IDENTITY_SUBSTRINGS = (
    "identity",
    "hash",
    "digest",
    "fingerprint",
    "manifest",
    "canonical",
    "provenance",
    "evidence_identity",
    "analytical_identity",
    "content_hash",
)
EXEMPTION_MARKER = "CANONICAL_SERIALIZATION_EXEMPT"


class Violation(NamedTuple):
    path: Path
    lineno: int
    function: str
    source_line: str
    reason: str


def _line_has_exemption(source_lines: list[str], lineno: int) -> bool:
    if lineno <= 0 or lineno > len(source_lines):
        return False
    line = source_lines[lineno - 1]
    idx = line.find(EXEMPTION_MARKER)
    if idx < 0:
        return False
    tail = line[idx + len(EXEMPTION_MARKER) :]
    return tail.startswith(":") and bool(tail[1:].strip())


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    names: dict[int, str] = {}

    def visit(node: ast.AST, current: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = node.name
        if current is not None:
            names[id(node)] = current
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(tree, None)
    return names


def _is_default_str(call: ast.Call) -> bool:
    func = call.func
    is_dumps = (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "json"
        and func.attr == "dumps"
    ) or (isinstance(func, ast.Name) and func.id == "dumps")
    if not is_dumps:
        return False
    for kw in call.keywords:
        if kw.arg != "default":
            continue
        return isinstance(kw.value, ast.Name) and kw.value.id == "str" or (
            isinstance(kw.value, ast.Attribute) and kw.value.attr == "str"
        )
    return False


def _walk_file(path: Path) -> Iterable[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return
    lines = source.splitlines()
    enclosing = _enclosing_functions(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_default_str(node):
            continue
        fn = enclosing.get(id(node), "")
        if not fn or not any(part in fn.lower() for part in IDENTITY_SUBSTRINGS):
            continue
        if _line_has_exemption(lines, node.lineno):
            continue
        yield Violation(
            path=path,
            lineno=node.lineno,
            function=fn,
            source_line=lines[node.lineno - 1].rstrip() if 0 < node.lineno <= len(lines) else "",
            reason="json.dumps(..., default=str) in an identity-bearing function without same-line exemption",
        )


def _iter_python_files(root: Path, exclusions: tuple[str, ...]) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        text = str(path).replace("\\", "/")
        if any(excl and excl.replace("\\", "/") in text for excl in exclusions):
            continue
        yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--exclusions",
        default="node_modules,.next,__pycache__,.venv,venv,migrations/versions",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    exclusions = tuple(x.strip() for x in args.exclusions.split(",") if x.strip())

    scan_dirs = (
        root / "packages" / "analytics_core" / "src",
        root / "apps" / "api" / "src",
    )
    violations: list[Violation] = []
    scanned = 0
    for directory in scan_dirs:
        if not directory.exists():
            continue
        for path in _iter_python_files(directory, exclusions):
            scanned += 1
            violations.extend(_walk_file(path))

    if violations:
        print(f"CANONICAL_SERIALIZATION_AUDIT: FAIL ({len(violations)} violation(s), {scanned} files scanned)")
        for violation in violations:
            try:
                relative = violation.path.relative_to(root)
            except ValueError:
                relative = violation.path
            print(f"  {relative}:{violation.lineno} in {violation.function}()")
            print(f"    {violation.source_line}")
        print("Use an exemption only when the particular serialization is provably not identity-bearing:")
        print("    # CANONICAL_SERIALIZATION_EXEMPT: <specific reason>")
        return 1

    print(f"CANONICAL_SERIALIZATION_AUDIT: PASS ({scanned} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
