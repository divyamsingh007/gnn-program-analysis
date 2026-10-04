import json

import pytest


pytest.importorskip("clang.cindex", reason="libclang is required for extraction tests")

from src.extraction.ast_extractor import extract_ast  # noqa: E402
from src.extraction.clang_utils import DEFAULT_SAMPLE_FILE  # noqa: E402


def test_extract_ast_returns_serializable_source_local_graph() -> None:
    ast = extract_ast(DEFAULT_SAMPLE_FILE)

    assert ast["representation"] == "ast"
    assert ast["nodes"][0]["kind"] == "TRANSLATION_UNIT"
    assert any(node["spelling"] == "copy_buffer" for node in ast["nodes"])
    assert all(edge["type"] == "ast_child" for edge in ast["edges"])
    assert json.loads(json.dumps(ast))["source_file"] == str(DEFAULT_SAMPLE_FILE)
