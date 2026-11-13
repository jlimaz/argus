import json
from pathlib import Path

import pytest

from pipeline.inputs import (
    MAX_OBSERVATIONS,
    InputError,
    combine_observations,
    merge_knowledge,
    parse_knowledge,
    parse_observations,
)

ROOT = Path(__file__).resolve().parents[1]
TESTE1 = (ROOT / "data" / "teste1.json").read_text()
TESTE2 = (ROOT / "data" / "teste2.json").read_text()
QUESTIONS_20 = json.loads((ROOT / "data" / "questions_20.json").read_text())

QUESTIONS = {
    "mode": {
        "type": "choice",
        "instructions": "Which signal type is this most likely?",
        "criteria": {"morse": "Morse code; OOK", "other": "any other signal"},
    }
}
ROW = {"center_frequency_hz": 7030000, "bandwidth_hz": 100, "modulation": "OOK"}


def knowledge_error(text):
    with pytest.raises(InputError) as raised:
        parse_knowledge(text)
    assert raised.value.field == "knowledge"
    return str(raised.value)


def data_error(text):
    with pytest.raises(InputError) as raised:
        parse_observations(text)
    assert raised.value.field == "data"
    return str(raised.value)


def test_knowledge_is_returned_as_parsed():
    assert parse_knowledge(json.dumps(QUESTIONS)) == QUESTIONS


def test_knowledge_syntax_error_names_line_and_column():
    message = knowledge_error('{\n  "mode": {,\n}')
    assert "line 2" in message
    assert "column" in message


def test_empty_knowledge_file():
    assert "empty" in knowledge_error("  \n")


def test_knowledge_given_a_list_points_at_the_data_file():
    assert "data file" in knowledge_error(json.dumps([ROW]))


def test_knowledge_given_jsonl_points_at_the_data_file():
    text = json.dumps(ROW) + "\n" + json.dumps(ROW) + "\n"
    assert "data file" in knowledge_error(text)


def test_knowledge_without_questions():
    assert "no questions" in knowledge_error("{}")


def test_knowledge_files_merge_criteria_in_selection_order():
    merged = merge_knowledge([("teste1.json", TESTE1), ("teste2.json", TESTE2)])
    assert merged == QUESTIONS_20
    assert list(merged["mode"]["criteria"]) == list(QUESTIONS_20["mode"]["criteria"])


def test_criteria_only_knowledge_needs_instructions():
    with pytest.raises(InputError) as raised:
        merge_knowledge([("teste2.json", TESTE2)])
    assert raised.value.field == "knowledge"
    message = str(raised.value)
    assert "teste2.json" in message
    assert "'mode'" in message
    assert "instructions" in message


def test_two_knowledge_files_cannot_both_give_instructions():
    with pytest.raises(InputError) as raised:
        merge_knowledge([("teste1.json", TESTE1), ("questions_20.json", json.dumps(QUESTIONS_20))])
    message = str(raised.value)
    assert "teste1.json" in message
    assert "questions_20.json" in message
    assert "'mode'" in message


def test_repeated_answer_names_the_files():
    overlap = json.loads(TESTE2)
    overlap["mode"]["criteria"] = {"morse": "again", **overlap["mode"]["criteria"]}
    with pytest.raises(InputError) as raised:
        merge_knowledge([("teste1.json", TESTE1), ("teste2.json", json.dumps(overlap))])
    message = str(raised.value)
    assert "teste2.json" in message
    assert "teste1.json" in message
    assert "'morse'" in message


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"type": "number"}, "type"),
        ({"instructions": ""}, "instructions"),
        ({"criteria": {}}, "criteria"),
        ({"criteria": {"morse": "only one"}}, "criteria"),
        ({"criteria": {"morse": 1, "other": "x"}}, "morse"),
    ],
)
def test_knowledge_question_problems_name_the_question(change, expected):
    message = knowledge_error(json.dumps({"mode": {**QUESTIONS["mode"], **change}}))
    assert "'mode'" in message
    assert expected in message


def test_observations_from_a_json_array_get_an_id():
    observations, total = parse_observations(json.dumps([ROW, {**ROW, "id": "b7"}]))
    assert total == 2
    assert [observation["id"] for observation in observations] == [0, "b7"]
    assert observations[0]["modulation"] == "OOK"


def test_observations_from_jsonl():
    text = "\n".join(json.dumps({**ROW, "id": index}) for index in range(3)) + "\n"
    observations, total = parse_observations(text)
    assert total == 3
    assert [observation["id"] for observation in observations] == [0, 1, 2]


def test_a_single_observation_object_is_one_row():
    observations, total = parse_observations(json.dumps(ROW))
    assert total == 1
    assert observations[0]["id"] == 0


def test_data_files_are_concatenated():
    first = [ROW, {**ROW, "id": "b7"}]
    second = [{**ROW, "modulation": "FM"}]
    observations, total = combine_observations(
        [("a.json", json.dumps(first)), ("b.json", json.dumps(second))]
    )
    assert total == 3
    assert [observation["id"] for observation in observations] == [0, "b7", 2]
    assert observations[2]["modulation"] == "FM"


def test_combined_data_row_errors_name_the_file():
    with pytest.raises(InputError) as raised:
        combine_observations(
            [("good.json", json.dumps([ROW])), ("bad.json", json.dumps([{"id": 1}]))]
        )
    assert raised.value.field == "data"
    message = str(raised.value)
    assert "bad.json, row 1" in message
    assert "center_frequency_hz" in message


def test_combined_observations_are_capped():
    first = [{**ROW, "id": f"a{index}"} for index in range(98)]
    second = [{**ROW, "id": f"b{index}"} for index in range(5)]
    observations, total = combine_observations(
        [("a.json", json.dumps(first)), ("b.json", json.dumps(second))]
    )
    assert total == 103
    assert len(observations) == MAX_OBSERVATIONS
    assert observations[98]["id"] == "b0"
    assert observations[-1]["id"] == "b1"


def test_a_later_slice_keeps_the_requested_rows():
    rows = [{**ROW, "id": index} for index in range(10)]
    observations, total = combine_observations([("a.json", json.dumps(rows))], limit=3, offset=4)
    assert total == 10
    assert [observation["id"] for observation in observations] == [4, 5, 6]


def test_a_later_slice_defaults_the_id_to_the_row_in_the_file():
    observations, total = combine_observations(
        [("", json.dumps([dict(ROW) for _ in range(5)]))], limit=2, offset=3
    )
    assert total == 5
    assert [observation["id"] for observation in observations] == [3, 4]


def test_rows_outside_the_window_are_not_checked():
    rows = [ROW, {"id": "bad"}]
    observations, total = combine_observations([("", json.dumps(rows))], limit=1, offset=0)
    assert total == 2
    assert len(observations) == 1
    assert observations[0]["id"] == 0


def test_observations_are_capped():
    rows = [{**ROW, "id": index} for index in range(MAX_OBSERVATIONS + 5)]
    observations, total = parse_observations(json.dumps(rows))
    assert len(observations) == MAX_OBSERVATIONS
    assert total == MAX_OBSERVATIONS + 5


def test_observations_syntax_error_names_line_and_column():
    message = data_error('[\n  {"center_frequency_hz": 1,}\n]')
    assert "line 2" in message


def test_broken_jsonl_names_the_line():
    message = data_error(json.dumps(ROW) + "\n" + '{"center_frequency_hz": \n')
    assert "line 2" in message


def test_empty_observations():
    assert "no observations" in data_error("[]")


def test_observations_given_a_knowledge_file_says_so():
    assert "knowledge file" in data_error(json.dumps(QUESTIONS))


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"bandwidth_hz": 100, "modulation": "OOK"}, "center_frequency_hz"),
        ({**ROW, "bandwidth_hz": "wide"}, "bandwidth_hz"),
        ({**ROW, "center_frequency_hz": True}, "center_frequency_hz"),
        ({**ROW, "center_frequency_hz": -5}, "center_frequency_hz"),
        ({"center_frequency_hz": 1, "bandwidth_hz": 1}, "modulation"),
        ({**ROW, "modulation": 3}, "modulation"),
        ("not an object", "object"),
    ],
)
def test_observation_problems_name_the_row_and_field(row, expected):
    message = data_error(json.dumps([ROW, row]))
    assert "Row 2" in message
    assert expected in message
