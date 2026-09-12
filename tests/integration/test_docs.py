"""The README must agree with the generated results and must not link to missing files."""

from __future__ import annotations

import re

import pytest

from src.utils.config import PROJECT_ROOT
from src.utils.io import load_json
from src.visualization.summary import extract_block, render_results_markdown

README = PROJECT_ROOT / "README.md"
SUMMARY = PROJECT_ROOT / "results" / "case_studies" / "summary.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="case studies have not been generated")
def test_readme_results_table_is_generated_from_the_committed_results():
    expected = render_results_markdown(load_json(SUMMARY)["rows"])
    assert extract_block(README.read_text(encoding="utf-8")).strip() == expected.strip()


def test_readme_local_links_point_to_existing_files():
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    missing = [
        target
        for target in targets
        if not target.startswith(("http://", "https://", "#", "mailto:"))
        and not (PROJECT_ROOT / target.split("#")[0]).exists()
    ]
    assert not missing, f"README links to missing files: {missing}"
