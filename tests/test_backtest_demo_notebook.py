"""Structural tests for ``backtest_demo.ipynb``.

We validate the shape of the notebook (cells, sections, code parses)
and exercise the inline synthetic-data generator so it's known good.
We do **not** execute the notebook end-to-end — that's slow, and the
underlying engine + reporter are already covered by their own tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from bot.backtest.data_loader import validate_ohlc


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "backtest_demo.ipynb"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def nb() -> nbformat.NotebookNode:
    if not NOTEBOOK_PATH.exists():
        pytest.fail(f"backtest_demo.ipynb not found at {NOTEBOOK_PATH}")
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


class TestNotebookStructure:
    def test_loads_as_valid_nbformat(self, nb: nbformat.NotebookNode) -> None:
        # Raises ValidationError on malformed cells; success = silence.
        nbformat.validate(nb)

    def test_has_python_kernelspec(self, nb: nbformat.NotebookNode) -> None:
        ks = nb.metadata.get("kernelspec", {})
        assert ks.get("language") == "python"

    def test_has_both_markdown_and_code_cells(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        md = [c for c in nb.cells if c.cell_type == "markdown"]
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert md, "no markdown cells"
        assert code, "no code cells"

    def test_first_cell_is_title(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        first = nb.cells[0]
        assert first.cell_type == "markdown"
        assert first.source.lstrip().startswith("# ")

    @pytest.mark.parametrize(
        "section_heading",
        [
            "Section 1 — Setup",
            "Section 2 — Load data",
            "Section 3 — Configure",
            "Section 4 — Run",
            "Section 5 — Visualise",
            "Section 6 — Generate HTML report",
            "Section 7 — Iteration",
        ],
    )
    def test_section_present(
        self, nb: nbformat.NotebookNode, section_heading: str,
    ) -> None:
        joined = "\n".join(
            c.source for c in nb.cells if c.cell_type == "markdown"
        )
        assert section_heading in joined, (
            f"expected section heading {section_heading!r} in notebook"
        )

    def test_dukascopy_instructions_present(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        joined = "\n".join(
            c.source for c in nb.cells if c.cell_type == "markdown"
        )
        assert "dukascopy" in joined.lower()

    def test_colab_gating_present(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        # Some cell must define IN_COLAB so Colab-specific calls are gated.
        joined = "\n".join(
            c.source for c in nb.cells if c.cell_type == "code"
        )
        assert "IN_COLAB" in joined
        # google.colab is referenced inside a try/except, never at top level
        # of every run. (We check the import sits in a try block.)
        assert "from google.colab" in joined or "import google.colab" in joined


# --------------------------------------------------------------------------- #
# Code-cell sanity
# --------------------------------------------------------------------------- #


class TestCodeCells:
    def test_all_code_cells_parse(self, nb: nbformat.NotebookNode) -> None:
        """Every code cell must be syntactically valid Python.

        We strip Jupyter magics (``%pip install …``) and shell escapes
        (``!something``) before parsing — those are valid in a notebook
        but aren't valid Python.
        """
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            stripped = "\n".join(
                line for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            try:
                ast.parse(stripped)
            except SyntaxError as e:
                pytest.fail(
                    f"cell {i} (source starts {cell.source[:60]!r}): {e}"
                )

    def test_imports_pull_from_bot_backtest(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        joined = "\n".join(
            c.source for c in nb.cells if c.cell_type == "code"
        )
        assert "from bot.backtest import" in joined
        # Headline names that the rest of the notebook depends on.
        for sym in (
            "BacktestConfig",
            "BacktestEngine",
            "load_dukascopy_csv",
            "generate_equity_curve",
            "generate_html_report",
        ):
            assert sym in joined, f"expected import of {sym}"


# --------------------------------------------------------------------------- #
# Synthetic data generator — actually run it
# --------------------------------------------------------------------------- #


def _extract_generator(nb: nbformat.NotebookNode) -> dict[str, object]:
    """Find the cell defining ``generate_synthetic_xauusd`` and exec it."""
    for cell in nb.cells:
        if cell.cell_type == "code" and "def generate_synthetic_xauusd" in cell.source:
            ns: dict[str, object] = {"np": np, "pd": pd}
            exec(cell.source, ns)  # noqa: S102 — known cell content
            return ns
    pytest.fail("synthetic data generator cell not found")
    return {}  # unreachable


class TestSyntheticDataGenerator:
    def test_default_args_returns_1000_bars(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        ns = _extract_generator(nb)
        df = ns["generate_synthetic_xauusd"]()  # type: ignore[operator]
        assert len(df) == 1000

    def test_custom_n_bars(self, nb: nbformat.NotebookNode) -> None:
        ns = _extract_generator(nb)
        df = ns["generate_synthetic_xauusd"](n_bars=250)  # type: ignore[operator]
        assert len(df) == 250

    def test_passes_validate_ohlc(self, nb: nbformat.NotebookNode) -> None:
        ns = _extract_generator(nb)
        df = ns["generate_synthetic_xauusd"]()  # type: ignore[operator]
        # Same validator the loader uses — any structural issue raises here.
        validate_ohlc(df)

    def test_realistic_xauusd_price_range(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        ns = _extract_generator(nb)
        df = ns["generate_synthetic_xauusd"]()  # type: ignore[operator]
        # Should be in the $2000-2500 ballpark (current Gold).
        assert 2000.0 <= df["close"].min() <= 2500.0
        assert 2000.0 <= df["close"].max() <= 2500.0

    def test_seeded_output_is_deterministic(
        self, nb: nbformat.NotebookNode,
    ) -> None:
        ns = _extract_generator(nb)
        df1 = ns["generate_synthetic_xauusd"](seed=42)  # type: ignore[operator]
        df2 = ns["generate_synthetic_xauusd"](seed=42)  # type: ignore[operator]
        pd.testing.assert_frame_equal(df1, df2)

    def test_utc_index(self, nb: nbformat.NotebookNode) -> None:
        ns = _extract_generator(nb)
        df = ns["generate_synthetic_xauusd"]()  # type: ignore[operator]
        assert str(df.index.tz) == "UTC"
