import json

import pytest


pytest.importorskip("clang.cindex", reason="libclang is required for extraction tests")

from src.extraction.clang_utils import DEFAULT_SAMPLE_FILE  # noqa: E402
from src.extraction.dataflow_extractor import extract_dataflow  # noqa: E402


def test_extract_dataflow_links_declarations_to_uses() -> None:
    dataflow = extract_dataflow(DEFAULT_SAMPLE_FILE)

    assert dataflow["representation"] == "dataflow"
    assert {variable["name"] for variable in dataflow["variables"]} >= {"x", "y"}
    assert dataflow["edges"]
    assert all(edge["type"] == "def_use" for edge in dataflow["edges"])
    assert json.loads(json.dumps(dataflow))["schema_version"] == "1.0"
