"""Utilities for validating that libclang can parse C and C++ sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import clang.cindex


DEFAULT_CLANG_ARGS = ("-std=c11",)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_FILE = PROJECT_ROOT / "tests" / "sample.c"


class ClangParseError(RuntimeError):
    """Raised when libclang cannot produce a valid translation unit."""


def configure_libclang(library_file: str | Path | None = None) -> None:
    """Configure a non-default libclang shared library before parsing.

    The ``libclang`` wheel normally finds its own shared library. Pass an
    explicit path only when an LLVM installation must be used instead.
    """

    if library_file is not None:
        clang.cindex.Config.set_library_file(str(Path(library_file).resolve()))


def parse_c_file(
    file_path: str | Path,
    compiler_args: Sequence[str] | None = None,
    *,
    library_file: str | Path | None = None,
) -> clang.cindex.TranslationUnit:
    """Parse a C or C++ source file and return its translation unit.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ClangParseError: If libclang reports an error-level diagnostic.
    """

    source_path = Path(file_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    configure_libclang(library_file)
    try:
        index = clang.cindex.Index.create()
        translation_unit = index.parse(
            str(source_path),
            args=list(compiler_args or DEFAULT_CLANG_ARGS),
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
    except clang.cindex.LibclangError as error:
        raise ClangParseError(f"Unable to initialize libclang: {error}") from error

    errors = [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if diagnostic.severity >= clang.cindex.Diagnostic.Error
    ]
    if errors:
        details = "\n".join(str(diagnostic) for diagnostic in errors)
        raise ClangParseError(f"Clang reported parse errors for {source_path}:\n{details}")

    return translation_unit


def diagnose_sample(
    sample_file: str | Path = DEFAULT_SAMPLE_FILE,
    compiler_args: Sequence[str] | None = None,
    *,
    library_file: str | Path | None = None,
) -> clang.cindex.TranslationUnit:
    """Parse the bundled sample program and print a compact AST summary."""

    translation_unit = parse_c_file(
        sample_file,
        compiler_args,
        library_file=library_file,
    )
    top_level_nodes = list(translation_unit.cursor.get_children())

    print(f"Parsed: {translation_unit.spelling}")
    print(f"Diagnostics: {len(translation_unit.diagnostics)}")
    print("Top-level AST nodes:")
    for node in top_level_nodes[:7]:
        print(f"- {node.kind.name}: {node.spelling or '<anonymous>'}")
    if len(top_level_nodes) > 7:
        print("- ...")

    return translation_unit


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the libclang connectivity check."""

    parser = argparse.ArgumentParser(
        description="Parse a C/C++ source file with libclang and print an AST summary."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="C/C++ source to parse (defaults to tests/sample.c).",
    )
    parser.add_argument(
        "--clang-arg",
        action="append",
        dest="clang_args",
        help="Compiler argument to pass to Clang. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--library-file",
        type=Path,
        help="Optional absolute path to the libclang shared library.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line diagnostic and return a process exit code."""

    args = build_argument_parser().parse_args(argv)
    try:
        diagnose_sample(
            args.source,
            args.clang_args,
            library_file=args.library_file,
        )
    except (FileNotFoundError, ClangParseError) as error:
        print(f"libclang diagnostic failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
