import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AMC23_DATA = REPO / "eval/chat_benchmarks/AMC23/data/amc23.json"

EXPECTED_NEW_ROWS = {
    9: ("AMC 12A", 18, "\\frac{3}{28}"),
    24: ("AMC 12A", 9, "2-\\sqrt3"),
    31: ("AMC 12B", 15, "E"),
    34: ("AMC 12B", 18, "A"),
    35: ("AMC 12B", 19, "E"),
    37: ("AMC 12B", 20, "\\frac{2\\arcsin\\frac{1}{4}}{\\pi}"),
    38: ("AMC 12B", 21, "6\\sqrt3+\\pi"),
    39: ("AMC 12B", 22, -2),
    42: ("AMC 12B", 25, "\\sqrt5-1"),
}

EXPECTED_MISSING = {("AMC 12A", 15)}
SOLUTION_MARKERS = (
    "==Solution",
    "Solution 1",
    "import olympiad",
    "[[File:",
    "[[Image:",
)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_exam_problem(url):
    match = re.search(r"2023_AMC_12([AB])_Problems/Problem_(\d+)", url)
    assert match, url
    return f"AMC 12{match.group(1)}", int(match.group(2))


def test_amc23_data_has_expected_text_ready_rows():
    rows = read_jsonl(AMC23_DATA)
    assert len(rows) == 49
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    assert {row["id"] for row in rows} == set(range(50)) - {6}

    exam_problems = {parse_exam_problem(row["url"]) for row in rows}
    all_exam_problems = {
        ("AMC 12A", problem_number) for problem_number in range(1, 26)
    } | {("AMC 12B", problem_number) for problem_number in range(1, 26)}
    assert all_exam_problems - exam_problems == EXPECTED_MISSING

    by_id = {row["id"]: row for row in rows}
    for row_id, (exam, problem_number, answer) in EXPECTED_NEW_ROWS.items():
        row = by_id[row_id]
        assert parse_exam_problem(row["url"]) == (exam, problem_number)
        assert row["answer"] == answer
        assert row["question"].strip()
        assert not any(marker in row["question"] for marker in SOLUTION_MARKERS)

    for row in rows:
        assert {"id", "answer", "url", "question"} <= set(row)
        assert str(row["answer"]).strip()
        assert row["url"].startswith("https://artofproblemsolving.com/wiki/index.php/")
        if "(A)" in row["question"] and "(E)" in row["question"]:
            answer = str(row["answer"])
            assert (
                row["answer"] in {"A", "B", "C", "D", "E"}
                or f"${answer}$" in row["question"]
            )
