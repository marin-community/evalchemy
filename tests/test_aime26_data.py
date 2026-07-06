import json
from pathlib import Path


DATA_PATH = Path("eval/chat_benchmarks/AIME26/data/aime26.json")
SOURCE_URL = "https://huggingface.co/datasets/MathArena/aime_2026"
EXPECTED_IDS = [f"I-{i}" for i in range(1, 16)] + [f"II-{i}" for i in range(1, 16)]
EXPECTED_ANSWERS = [
    "277",
    "62",
    "79",
    "70",
    "65",
    "441",
    "396",
    "244",
    "29",
    "156",
    "896",
    "161",
    "39",
    "681",
    "83",
    "178",
    "243",
    "503",
    "279",
    "190",
    "50",
    "754",
    "245",
    "669",
    "850",
    "132",
    "223",
    "107",
    "157",
    "393",
]


def load_rows():
    return [json.loads(line) for line in DATA_PATH.read_text().splitlines()]


def test_aime26_jsonl_is_stable_and_portable():
    raw = DATA_PATH.read_bytes()

    assert raw.endswith(b"\n")
    assert all(byte < 128 for byte in raw)


def test_aime26_jsonl_schema_ids_and_source():
    rows = load_rows()

    assert len(rows) == 30
    assert [row["id"] for row in rows] == EXPECTED_IDS
    assert len({row["id"] for row in rows}) == len(rows)

    for row in rows:
        assert set(row) == {"id", "problem", "solution", "answer", "url"}
        assert row["problem"].strip()
        assert row["solution"] == ""
        assert row["url"] == SOURCE_URL


def test_aime26_answers_match_matharena_snapshot():
    rows = load_rows()

    assert [row["answer"] for row in rows] == EXPECTED_ANSWERS

    for row in rows:
        assert row["answer"].isdigit()
        assert 0 <= int(row["answer"]) <= 999
