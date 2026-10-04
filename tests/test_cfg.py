import json

import pytest


pytest.importorskip("clang.cindex", reason="libclang is required for extraction tests")

from src.extraction.cfg_extractor import extract_cfg  # noqa: E402
from src.extraction.clang_utils import DEFAULT_SAMPLE_FILE  # noqa: E402


def test_extract_cfg_models_functions_branches_and_returns() -> None:
    cfg = extract_cfg(DEFAULT_SAMPLE_FILE)

    assert cfg["representation"] == "cfg"
    assert {function["name"] for function in cfg["functions"]} >= {
        "copy_buffer",
        "main",
    }
    edge_types = {edge["type"] for edge in cfg["edges"]}
    assert {"true", "false", "return"} <= edge_types
    assert json.loads(json.dumps(cfg))["schema_version"] == "1.0"
