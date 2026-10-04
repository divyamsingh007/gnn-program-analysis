"""Build a conservative, AST-derived intraprocedural control-flow graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from clang.cindex import Cursor

from .clang_utils import parse_c_file


SCHEMA_VERSION = "1.0"
CONTROL_STATEMENTS = {"IF_STMT", "FOR_STMT", "WHILE_STMT", "DO_STMT"}


def _source_location(cursor: Cursor) -> dict[str, Any]:
    location = cursor.location
    return {
        "file": str(location.file) if location.file else None,
        "line": location.line,
        "column": location.column,
    }


def _direct_children(cursor: Cursor) -> list[Cursor]:
    return list(cursor.get_children())


def _body_statements(cursor: Cursor) -> list[Cursor]:
    """Return a statement sequence, unwrapping a compound statement."""

    if cursor.kind.name == "COMPOUND_STMT":
        return _direct_children(cursor)
    return [cursor]


def _function_body(function: Cursor) -> Cursor | None:
    return next(
        (child for child in function.get_children() if child.kind.name == "COMPOUND_STMT"),
        None,
    )


def _function_nodes(cursor: Cursor, source_path: Path) -> Iterable[Cursor]:
    for child in cursor.get_children():
        location_file = child.location.file
        if (
            child.kind.name == "FUNCTION_DECL"
            and location_file is not None
            and Path(str(location_file)).resolve() == source_path
            and child.is_definition()
        ):
            yield child


def extract_cfg(
    file_path: str | Path,
    compiler_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return an AST-derived CFG with ``next``, branch, and return edges.

    libclang's Python API does not expose LLVM's compiler CFG. This extractor
    therefore models source-level statement flow for each defined function and
    keeps the representation intentionally conservative for Phase 2.
    """

    source_path = Path(file_path).expanduser().resolve()
    translation_unit = parse_c_file(source_path, compiler_args)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    next_node_id = 0

    def add_block(function_name: str, block_type: str, cursor: Cursor | None) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        nodes.append(
            {
                "id": node_id,
                "function": function_name,
                "block_type": block_type,
                "ast_kind": cursor.kind.name if cursor else None,
                "label": cursor.displayname if cursor else block_type.lower(),
                "location": _source_location(cursor) if cursor else None,
            }
        )
        return node_id

    def connect(sources: Iterable[int], target: int, edge_type: str) -> None:
        for source in sources:
            edges.append({"source": source, "target": target, "type": edge_type})

    for function in _function_nodes(translation_unit.cursor, source_path):
        body = _function_body(function)
        if body is None:
            continue

        function_name = function.spelling
        entry_id = add_block(function_name, "entry", function)
        exit_id = add_block(function_name, "exit", function)

        def build_sequence(
            statements: list[Cursor],
            incoming: list[int],
            initial_edge_type: str = "next",
        ) -> list[int]:
            exits = incoming
            edge_type = initial_edge_type
            for statement in statements:
                exits = build_statement(statement, exits, edge_type)
                edge_type = "next"
            return exits

        def build_statement(
            statement: Cursor,
            incoming: list[int],
            edge_type: str,
        ) -> list[int]:
            kind = statement.kind.name
            if kind == "COMPOUND_STMT":
                return build_sequence(_body_statements(statement), incoming, edge_type)

            if kind == "IF_STMT":
                condition_id = add_block(function_name, "condition", statement)
                connect(incoming, condition_id, edge_type)
                children = _direct_children(statement)
                branches = [child for child in children if child.kind.name == "COMPOUND_STMT"]
                then_branch = branches[0] if branches else None
                else_branch = branches[1] if len(branches) > 1 else None

                then_exits = (
                    build_sequence(_body_statements(then_branch), [condition_id], "true")
                    if then_branch
                    else [condition_id]
                )
                else_exits = (
                    build_sequence(_body_statements(else_branch), [condition_id], "false")
                    if else_branch
                    else [condition_id]
                )
                join_id = add_block(function_name, "join", statement)
                connect(then_exits, join_id, "next")
                connect(else_exits, join_id, "next")
                return [join_id]

            if kind in {"FOR_STMT", "WHILE_STMT", "DO_STMT"}:
                condition_id = add_block(function_name, "loop_condition", statement)
                connect(incoming, condition_id, edge_type)
                body_cursor = next(
                    (
                        child
                        for child in reversed(_direct_children(statement))
                        if child.kind.name == "COMPOUND_STMT"
                    ),
                    None,
                )
                body_exits = (
                    build_sequence(_body_statements(body_cursor), [condition_id], "true")
                    if body_cursor
                    else [condition_id]
                )
                connect(body_exits, condition_id, "loop_back")
                after_loop_id = add_block(function_name, "after_loop", statement)
                connect([condition_id], after_loop_id, "false")
                return [after_loop_id]

            block_type = "return" if kind == "RETURN_STMT" else "statement"
            statement_id = add_block(function_name, block_type, statement)
            connect(incoming, statement_id, edge_type)
            if kind == "RETURN_STMT":
                connect([statement_id], exit_id, "return")
                return []
            return [statement_id]

        terminal_blocks = build_sequence(_body_statements(body), [entry_id])
        connect(terminal_blocks, exit_id, "next")
        functions.append(
            {"name": function_name, "entry": entry_id, "exit": exit_id}
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "representation": "cfg",
        "source_file": str(source_path),
        "functions": functions,
        "nodes": nodes,
        "edges": edges,
    }
