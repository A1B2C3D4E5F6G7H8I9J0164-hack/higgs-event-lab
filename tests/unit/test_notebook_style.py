import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))
MAX_CODE_LINES = 40
MIN_EXPLANATION_WORDS = 25
MAX_OUTPUT_CHARS = 20_000


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(cell):
    return "".join(cell.get("source", []))


def test_notebooks_are_valid_json():
    for path in NOTEBOOKS:
        assert load(path)["nbformat"] >= 4, path


def test_first_cell_is_markdown_title():
    for path in NOTEBOOKS:
        cells = load(path)["cells"]
        assert cells and cells[0]["cell_type"] == "markdown", f"{path.name}: first cell must be a Markdown title/overview"


def test_every_code_cell_is_preceded_by_a_markdown_explanation():
    for path in NOTEBOOKS:
        cells = load(path)["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                assert index > 0 and cells[index - 1]["cell_type"] == "markdown", f"{path.name}: code cell {index} has no Markdown cell before it"


def test_explanations_are_long_enough_to_be_useful():
    for path in NOTEBOOKS:
        cells = load(path)["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code" and index > 0:
                words = len(re.findall(r"\w+", source(cells[index - 1])))
                assert words >= MIN_EXPLANATION_WORDS, f"{path.name}: explanation before cell {index} has only {words} words"


def test_code_cells_have_no_tutorial_comments():
    for path in NOTEBOOKS:
        for index, cell in enumerate(load(path)["cells"]):
            if cell["cell_type"] == "code":
                comments = [l for l in source(cell).splitlines() if l.strip().startswith("#")]
                assert not comments, f"{path.name}: code cell {index} contains comments {comments[:2]}"


def test_code_cells_are_short_enough_to_read():
    for path in NOTEBOOKS:
        for index, cell in enumerate(load(path)["cells"]):
            if cell["cell_type"] == "code":
                assert len(source(cell).splitlines()) <= MAX_CODE_LINES, f"{path.name}: code cell {index} is too long; move logic into src/higgs_event_lab/"


def test_no_large_or_stale_outputs_are_committed():
    for path in NOTEBOOKS:
        for index, cell in enumerate(load(path)["cells"]):
            if cell["cell_type"] == "code":
                assert len(json.dumps(cell.get("outputs", []))) <= MAX_OUTPUT_CHARS, f"{path.name}: cell {index} output too large"
