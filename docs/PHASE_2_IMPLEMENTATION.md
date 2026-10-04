# Phase 2 Implementation Record

## Scope

Phase 2 converts a parsed C/C++ source file into three JSON-serializable
intermediate representations for graph construction in Phase 3:

- an Abstract Syntax Tree (AST);
- an intraprocedural Control-Flow Graph (CFG); and
- a local Definition-Use data-flow graph (DFG).

All extractors accept a source path and optional Clang compiler arguments.
They use `parse_c_file(...)` from `src/extraction/clang_utils.py`, retaining
the Phase 1 behavior for path validation and Clang diagnostics.

## Shared Contract

Every extractor returns a dictionary containing:

```json
{
  "schema_version": "1.0",
  "representation": "ast | cfg | dataflow",
  "source_file": "absolute source path",
  "nodes": [],
  "edges": []
}
```

This stable, JSON-safe shape is the handoff contract for the NetworkX builders
planned in Phase 3.

## AST Extraction

`src/extraction/ast_extractor.py` provides `extract_ast(...)`.

- Nodes include a numeric ID, Clang cursor kind, spelling, display name, type,
  source location, and token spellings.
- `ast_child` edges are directed from a parent cursor to a child cursor.
- Only cursors located in the requested source file are retained, excluding
  system-header declarations that would otherwise dominate the graph.
- The translation-unit cursor is the root node with ID `0`.

## CFG Extraction

`src/extraction/cfg_extractor.py` provides `extract_cfg(...)`.

Each defined source function has explicit `entry` and `exit` blocks. Statement
blocks capture the source-level execution order. The extractor creates labeled
edges for `next`, `true`, `false`, `loop_back`, and `return` flow.

The Python libclang API exposes AST cursors but not LLVM's full compiler CFG.
For that reason, this implementation is explicitly an AST-derived,
intraprocedural approximation. It supports `if`, `for`, `while`, `do`, nested
compound statements, and return statements. It does not yet model switch
dispatch, goto, exceptions, macro expansion flow, or compiler-generated basic
blocks.

## Data-Flow Extraction

`src/extraction/dataflow_extractor.py` provides `extract_dataflow(...)`.

- `VAR_DECL` and `PARM_DECL` cursors create definition nodes.
- Assignment targets create replacement definition nodes for simple and common
  compound assignment operators.
- `DECL_REF_EXPR` cursors create use nodes.
- A `def_use` edge connects each use to the latest lexical definition seen for
  the same Clang variable identity.

This is a local lexical reaching-definition model. It intentionally does not
claim pointer alias analysis, interprocedural propagation, path sensitivity,
or full SSA semantics. Those are future extensions, not silent assumptions.

## Automated Validation

The Phase 2 checks are located in:

- `tests/test_ast.py`
- `tests/test_cfg.py`
- `tests/test_dataflow.py`

They verify the common JSON contract, AST function extraction, CFG function and
branch/return edges, and non-empty declaration-to-use links for the bundled
sample program.

Validation run on 2026-10-04:

```text
python -m pytest tests/test_clang_utils.py tests/test_ast.py tests/test_cfg.py tests/test_dataflow.py -q
6 passed in 3.02s
```

`pytest>=7.4` was installed into the active Python 3.13 environment before
this run because it was declared in the project requirements but absent from
that interpreter.

## Phase 3 Handoff

Phase 3 can convert the three `nodes`/`edges` payloads to directed NetworkX
graphs without re-parsing source files. Relationship labels map naturally to
edge attributes: `ast_child`, CFG flow types, and `def_use`.
