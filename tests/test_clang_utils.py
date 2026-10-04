from pathlib import Path

import pytest


pytest.importorskip("clang.cindex", reason="libclang is required for parser tests")

from src.extraction.clang_utils import (  # noqa: E402
    ClangParseError,
    DEFAULT_SAMPLE_FILE,
    parse_c_file,
)


def test_parse_sample_c_returns_translation_unit() -> None:
    translation_unit = parse_c_file(DEFAULT_SAMPLE_FILE)

    assert Path(translation_unit.spelling).resolve() == DEFAULT_SAMPLE_FILE.resolve()
    assert any(
        cursor.spelling == "copy_buffer" for cursor in translation_unit.cursor.get_children()
    )


def test_parse_c_file_rejects_missing_source() -> None:
    with pytest.raises(FileNotFoundError):
        parse_c_file("tests/does_not_exist.c")


def test_parse_c_file_reports_syntax_errors(tmp_path: Path) -> None:
    invalid_source = tmp_path / "invalid.c"
    invalid_source.write_text("int main( { return 0; }", encoding="utf-8")

    with pytest.raises(ClangParseError, match="parse errors"):
        parse_c_file(invalid_source)
