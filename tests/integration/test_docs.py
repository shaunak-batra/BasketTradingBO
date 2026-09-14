"""The README must agree with the generated results and must not link to missing files."""

from __future__ import annotations

import re

import pytest

from src.utils.config import PROJECT_ROOT
from src.utils.io import load_json
from src.visualization.summary import extract_block, render_results_markdown

README = PROJECT_ROOT / "README.md"
SUMMARY = PROJECT_ROOT / "results" / "case_studies" / "summary.json"
SUMMARY_MD = SUMMARY.with_name("summary.md")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _missing_local_links(markdown_file) -> list[str]:
    """Local link targets that do not exist, resolved from the markdown file's own folder as GitHub does."""
    return [
        target
        for target in LINK.findall(markdown_file.read_text(encoding="utf-8"))
        if not target.startswith(("http://", "https://", "#", "mailto:"))
        and not (markdown_file.parent / target.split("#")[0]).exists()
    ]


@pytest.mark.skipif(not SUMMARY.exists(), reason="case studies have not been generated")
def test_readme_results_table_is_generated_from_the_committed_results():
    expected = render_results_markdown(load_json(SUMMARY)["rows"])
    assert extract_block(README.read_text(encoding="utf-8")).strip() == expected.strip()


def test_readme_local_links_point_to_existing_files():
    missing = _missing_local_links(README)
    assert not missing, f"README links to missing files: {missing}"


@pytest.mark.skipif(not SUMMARY.exists(), reason="case studies have not been generated")
def test_case_study_summary_is_generated_with_links_relative_to_its_folder():
    expected = render_results_markdown(load_json(SUMMARY)["rows"], link_root="results/case_studies")
    assert SUMMARY_MD.read_text(encoding="utf-8").strip() == expected.strip()
    assert LINK.findall(expected)
    assert not _missing_local_links(SUMMARY_MD)


@pytest.mark.parametrize("name", sorted(path.name for path in (PROJECT_ROOT / "docs").glob("*.md")))
def test_docs_local_links_point_to_existing_files(name):
    missing = _missing_local_links(PROJECT_ROOT / "docs" / name)
    assert not missing, f"docs/{name} links to missing files: {missing}"
