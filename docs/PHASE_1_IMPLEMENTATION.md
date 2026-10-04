# Phase 1 Implementation Record

## Scope

Phase 1 establishes a reproducible Python environment, central configuration,
and a repeatable libclang connectivity check for the Graph Neural Network for
Program Analysis project.

## Delivered Components

### Dependencies

`requirements.txt` declares the scientific Python, graph learning, compiler,
configuration, visualization, notebook, and testing dependencies required by
the first project phase. Versions are expressed as minimum compatible versions
so installation can select platform-appropriate PyTorch and PyG builds.

### Global Configuration

`config/config.yaml` centralizes:

- project identity and model hyperparameters;
- raw, intermediate, processed, split, checkpoint, and report paths;
- C11 compiler arguments and an optional `libclang` library override; and
- automatic device selection plus a reproducibility seed.

The `clang.library_file` value is `null` by default because the `libclang`
Python package normally discovers its bundled shared library. Set it to an
absolute path only when using a separate LLVM installation.

### C Parser Diagnostic

`tests/sample.c` provides a compact C11 fixture with declarations, an `if`
branch, a function call, and a deliberately unsafe `sprintf` pattern for later
program-analysis stages.

`src/extraction/clang_utils.py` now provides:

- `parse_c_file(...)`, which validates the input path, parses with C11
  arguments, and raises `ClangParseError` for error-level Clang diagnostics;
- `configure_libclang(...)`, which accepts an optional explicit shared-library
  path before Clang initializes;
- `diagnose_sample(...)`, which prints the parsed file, diagnostic count, and a
  concise top-level AST summary; and
- a command-line interface that exits with status `0` after a successful parse
  and status `1` with a readable error otherwise.

Run the diagnostic from the repository root:

```powershell
python src/extraction/clang_utils.py
```

For an LLVM installation whose library is not discovered automatically:

```powershell
python src/extraction/clang_utils.py --library-file C:/LLVM/bin/libclang.dll
```

### Automated Checks

`tests/test_clang_utils.py` verifies that:

1. the bundled C fixture yields a translation unit containing `copy_buffer`;
2. a missing input produces `FileNotFoundError`; and
3. invalid C produces `ClangParseError`.

The module is skipped only if `clang.cindex` is unavailable, which makes a
missing optional local installation visible without turning unrelated test
runs into import failures.

Run the focused tests:

```powershell
pytest tests/test_clang_utils.py
```

## Validation Record

On 2026-10-04, the configured Python 3.13 interpreter successfully ran:

```powershell
python src/extraction/clang_utils.py
```

The diagnostic parsed `tests/sample.c` and reported zero Clang diagnostics,
confirming that the installed `clang.cindex` binding and its `libclang` shared
library can analyze the fixture.

The automated `pytest` command could not run in that interpreter because the
`pytest` package was not installed. A focused installation attempt did not
complete through the restricted package-resolution path. The test module is in
place and should be run after installing the project requirements.

## Phase 1 Completion Criteria

Phase 1 is complete when the requirements are installed and the focused test
suite passes in the target environment. The next phase can build AST, CFG, and
data-flow extractors on the `parse_c_file(...)` contract.
