"""Extract basic local definition-use chains from Clang cursors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from clang.cindex import Cursor

from .clang_utils import parse_c_file


SCHEMA_VERSION = "1.0"
DECLARATION_KINDS = {"VAR_DECL", "PARM_DECL"}
ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%="}


def _location(cursor: Cursor) -> dict[str, Any]:
    location = cursor.location
    return {
        "file": str(location.file) if location.file else None,
        "line": location.line,
        "column": location.column,
    }


def _variable_key(cursor: Cursor) -> str:
    """Return a stable variable identity for a cursor/reference pair."""

    referenced = cursor.referenced or cursor
    usr = referenced.get_usr()
    if usr:
        return usr
    location = referenced.location
    return f"{referenced.spelling}@{location.line}:{location.column}"


def _is_assignment_target(cursor: Cursor, parent: Cursor | None) -> bool:
    if parent is None or parent.kind.name != "BINARY_OPERATOR":
        return False
    tokens = [token.spelling for token in parent.get_tokens()]
    if not any(token in ASSIGNMENT_OPERATORS for token in tokens):
        return False
    children = list(parent.get_children())
    return bool(children) and children[0].hash == cursor.hash


def extract_dataflow(
    file_path: str | Path,
    compiler_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return local definitions, uses, and directed definition-use edges.

    The output models lexical reaching definitions within the source file. It
    deliberately does not claim interprocedural, alias, pointer, or path-
    sensitive analysis; those require later compiler-analysis extensions.
    """

    source_path = Path(file_path).expanduser().resolve()
    translation_unit = parse_c_file(source_path, compiler_args)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    variables: dict[str, dict[str, Any]] = {}
    last_definitions: dict[str, int] = {}

    def from_source(cursor: Cursor) -> bool:
        location_file = cursor.location.file
        return location_file is not None and Path(str(location_file)).resolve() == source_path

    def add_node(role: str, cursor: Cursor, variable_key: str) -> int:
        node_id = len(nodes)
        node = {
            "id": node_id,
            "role": role,
            "name": cursor.spelling,
            "variable_id": variable_key,
            "ast_kind": cursor.kind.name,
            "location": _location(cursor),
        }
        nodes.append(node)
        variables.setdefault(
            variable_key,
            {"id": variable_key, "name": cursor.spelling, "type": cursor.type.spelling},
        )
        return node_id

    def visit(cursor: Cursor, parent: Cursor | None = None) -> None:
        if not from_source(cursor) and cursor.kind.name != "TRANSLATION_UNIT":
            return

        kind = cursor.kind.name
        if kind in DECLARATION_KINDS:
            variable_key = _variable_key(cursor)
            definition_id = add_node("definition", cursor, variable_key)
            last_definitions[variable_key] = definition_id
        elif kind == "DECL_REF_EXPR" and cursor.referenced is not None:
            variable_key = _variable_key(cursor)
            if _is_assignment_target(cursor, parent):
                definition_id = add_node("definition", cursor, variable_key)
                last_definitions[variable_key] = definition_id
            else:
                use_id = add_node("use", cursor, variable_key)
                definition_id = last_definitions.get(variable_key)
                if definition_id is not None:
                    edges.append(
                        {
                            "source": definition_id,
                            "target": use_id,
                            "type": "def_use",
                            "variable_id": variable_key,
                        }
                    )

        for child in cursor.get_children():
            visit(child, cursor)

    visit(translation_unit.cursor)
    return {
        "schema_version": SCHEMA_VERSION,
        "representation": "dataflow",
        "source_file": str(source_path),
        "variables": list(variables.values()),
        "nodes": nodes,
        "edges": edges,
    }
