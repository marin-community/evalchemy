import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.chat_benchmarks.amc_utils import (  # noqa: E402
    amc_answer_is_correct,
    extract_boxed_answer,
    normalize_amc_answer,
)

TASKS = {
    "AMC24": {
        "data": REPO / "eval/chat_benchmarks/AMC24/data/amc24.json",
        "manifest": REPO / "eval/chat_benchmarks/AMC24/data/source_manifest.json",
        "expected_rows": 48,
        "expected_year": 2024,
        "expected_extraction_source": "rawsh/2024_AMC12",
        "expected_excluded": {
            "2024-amc12a-18",
            "2024-amc12a-22",
        },
        "expected_manual_included": {
            "2024-amc12a-14",
            "2024-amc12a-20",
            "2024-amc12b-07",
            "2024-amc12b-19",
        },
    },
    "AMC25": {
        "data": REPO / "eval/chat_benchmarks/AMC25/data/amc25.json",
        "manifest": REPO / "eval/chat_benchmarks/AMC25/data/source_manifest.json",
        "expected_rows": 49,
        "expected_year": 2025,
        "expected_extraction_source": "sonthenguyen/amc12-2025-non-figure",
        "expected_excluded": {
            "2025-amc12a-05",
        },
        "expected_manual_included": {
            "2025-amc12a-10",
            "2025-amc12a-14",
            "2025-amc12a-20",
            "2025-amc12a-24",
            "2025-amc12b-12",
            "2025-amc12b-13",
            "2025-amc12b-15",
        },
    },
}

REQUIRED_ROW_FIELDS = {
    "id",
    "question",
    "answer",
    "url",
    "year",
    "exam",
    "problem_number",
    "source",
    "source_url",
    "canonical_source",
    "canonical_url",
    "has_figure",
    "include_in_eval",
    "prompt_type",
    "extraction_source",
    "extraction_source_url",
    "review_status",
    "checked_against",
}

SOLUTION_MARKERS = ("Solution", "Answer:", "\\includegraphics", "<img")


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with path.open() as f:
        return json.load(f)


def sort_key(row):
    exam_order = 0 if row["exam"].endswith("12A") else 1
    return exam_order, int(row["problem_number"])


def test_amc_data_files_are_valid_and_self_contained():
    for task, paths in TASKS.items():
        rows = read_jsonl(paths["data"])
        assert len(rows) == paths["expected_rows"], task
        assert rows == sorted(rows, key=sort_key)

        ids = set()
        exam_problem_pairs = set()
        for row in rows:
            assert REQUIRED_ROW_FIELDS <= set(row), row.get("id")
            assert row["id"] not in ids
            ids.add(row["id"])

            pair = (row["exam"], row["problem_number"])
            assert pair not in exam_problem_pairs
            exam_problem_pairs.add(pair)

            assert row["year"] == paths["expected_year"]
            assert row["include_in_eval"] is True
            assert row["has_figure"] is False or row.get("figure_handling")
            assert row["source"] == "AoPS"
            assert row["canonical_source"] == "MAA AMC"
            assert row["canonical_url"] == "https://maa.org/student-programs/amc/"
            if row["review_status"] == "hf_mirror_imported":
                assert row["extraction_source"] == paths["expected_extraction_source"]
                assert row["checked_against"] == [paths["expected_extraction_source"]]
            elif row["review_status"] == "aops_self_contained_imported":
                assert row["id"] in paths["expected_manual_included"]
                assert row["extraction_source"] == "AoPS"
                assert row["checked_against"] == ["AoPS"]
            else:
                assert False, row["review_status"]

            if row["has_figure"]:
                assert (
                    row.get("figure_handling")
                    == "source_figure_omitted_text_self_contained"
                )
            assert row["url"] == row["source_url"]
            assert row["url"].startswith(
                "https://artofproblemsolving.com/wiki/index.php/"
            )
            assert row["question"].strip()
            assert str(row["answer"]).strip()
            assert not any(marker in row["question"] for marker in SOLUTION_MARKERS)
            if "which of the following" in row["question"].lower():
                assert row.get("choices") or row.get("review_notes") == "self-contained"

        assert not ids & paths["expected_excluded"]
        assert paths["expected_manual_included"] <= ids


def test_amc_source_manifests_match_data():
    for task, paths in TASKS.items():
        rows = read_jsonl(paths["data"])
        manifest = read_json(paths["manifest"])
        assert manifest["task"] == task
        assert manifest["year"] == paths["expected_year"]
        assert manifest["candidate_count"] == 50
        assert manifest["included_count"] == len(rows)
        assert manifest["included_count"] + manifest["excluded_count"] == 50
        assert manifest["source_policy"]["runtime_network_access"] is False
        assert manifest["source_policy"]["canonical_owner"] == "MAA AMC"
        assert (
            manifest["source_policy"]["canonical_url"]
            == "https://maa.org/student-programs/amc/"
        )
        assert (
            manifest["source_policy"]["extraction_source"]
            == paths["expected_extraction_source"]
        )
        assert [check["name"] for check in manifest["cross_checks"]] == [
            paths["expected_extraction_source"]
        ]
        assert manifest["source_policy"]["manual_review_policy"]

        excluded_ids = {excluded["id"] for excluded in manifest["excluded"]}
        assert excluded_ids == paths["expected_excluded"]
        manual_ids = {included["id"] for included in manifest["manual_inclusions"]}
        assert manual_ids == paths["expected_manual_included"]
        assert not manual_ids & excluded_ids
        for excluded in manifest["excluded"]:
            assert {
                "id",
                "exam",
                "problem_number",
                "url",
                "reason",
                "review_status",
            } <= set(excluded)
            assert excluded["review_status"] == "excluded"
            assert excluded["reason"]
            assert excluded["url"].startswith(
                "https://artofproblemsolving.com/wiki/index.php/"
            )
        for included in manifest["manual_inclusions"]:
            assert {
                "id",
                "exam",
                "problem_number",
                "url",
                "reason",
                "review_status",
            } <= set(included)
            assert included["review_status"] == "included"
            assert included["reason"]
            assert included["url"].startswith(
                "https://artofproblemsolving.com/wiki/index.php/"
            )


def test_amc_answer_normalization_and_scoring():
    cases = [
        ("27", "27"),
        ("27", "27.0"),
        ("\\frac{39}{7}", "\\frac{39}{7}"),
        ("4{:}30", "4:30"),
        ("D", "d"),
        ("(0, \\frac{1}{2})", "(0,\\frac{1}{2})"),
        ("[\\frac{3}{4}, \\frac{7}{8}]", "[\\frac{3}{4},\\frac{7}{8}]"),
    ]
    for expected, predicted in cases:
        assert amc_answer_is_correct(expected, predicted)

    assert normalize_amc_answer("$\\boxed{4{:}30}$") == "4:30"
    assert extract_boxed_answer("Therefore $\\boxed{27}$.") == "27"
    assert extract_boxed_answer("Use the last one: \\boxed{1}, then \\boxed{2}.") == "2"
    assert amc_answer_is_correct("4:30", "$\\boxed{4{:}30}$")
