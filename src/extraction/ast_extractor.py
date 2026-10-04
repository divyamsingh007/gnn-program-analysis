"""Extract a source-local Abstract Syntax Tree from C or C++ code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from clang.cindex import Cursor

from .clang_utils import parse_c_file


SCHEMA_VERSION = "1.0"


def _location(cursor: Cursor) -> dict[str, Any]:
    """Return a JSON-safe source location for a Clang cursor."""

    location = cursor.location
    return {
        "file": str(location.file) if location.file else None,
        "line": location.line,
        "column": location.column,
        "offset": location.offset,
    }


def _is_from_source(cursor: Cursor, source_path: Path) -> bool:
    location_file = cursor.location.file
    if location_file is None:
        return False
    return Path(str(location_file)).resolve() == source_path


def _node_record(node_id: int, cursor: Cursor) -> dict[str, Any]:
    """Convert the stable cursor fields needed by later graph stages."""

    return {
        "id": node_id,
        "kind": cursor.kind.name,
        "spelling": cursor.spelling,
        "display_name": cursor.displayname,
        "type": cursor.type.spelling,
        "location": _location(cursor),
        "tokens": [token.spelling for token in cursor.get_tokens()],
    }


def extract_ast(
    file_path: str | Path,
    compiler_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse ``file_path`` and return a JSON-serializable AST representation.

    Nodes are restricted to the requested source file, which excludes Clang's
    system-header declarations. Each ``ast_child`` edge is directed from a
    parent cursor to its child cursor.
    """

    source_path = Path(file_path).expanduser().resolve()
    translation_unit = parse_c_file(source_path, compiler_args)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    root_id = 0
    nodes.append(_node_record(root_id, translation_unit.cursor))
    next_node_id = 1

    def visit(cursor: Cursor, parent_id: int) -> None:
        nonlocal next_node_id

        for child in cursor.get_children():
            if not _is_from_source(child, source_path):
                continue
            child_id = next_node_id
            next_node_id += 1
            nodes.append(_node_record(child_id, child))
            edges.append(
                {"source": parent_id, "target": child_id, "type": "ast_child"}
            )
            visit(child, child_id)

    visit(translation_unit.cursor, root_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "representation": "ast",
        "source_file": str(source_path),
        "nodes": nodes,
        "edges": edges,
    }
