from api_server import (
    ROSTER_PATH,
    match_roster_names,
    normalize_spoken_name,
    read_roster_rows,
)


SAMPLE = [
    {
        "participant_id": "SB01",
        "name": "王小明",
        "name_aliases": "王曉明;小明",
        "age": 75,
        "height_cm": 165.0,
        "weight_kg": 60.0,
        "gender": "male",
    },
    {
        "participant_id": "SB02",
        "name": "李美玲",
        "name_aliases": "美玲",
        "age": 72,
        "height_cm": 158.0,
        "weight_kg": 52.0,
        "gender": "female",
    },
]


def test_normalize_spoken_name_only_removes_safe_noise():
    assert normalize_spoken_name(" 王・小 明。") == "王小明"


def test_exact_name_and_alias_match():
    direct = match_roster_names(SAMPLE, ["王小明"])
    alias = match_roster_names(SAMPLE, ["王曉明"])
    assert direct[0]["participant_id"] == "SB01"
    assert alias[0]["participant_id"] == "SB01"


def test_similar_but_not_registered_name_does_not_match():
    assert match_roster_names(SAMPLE, ["王大明"]) == []


def test_generated_roster_is_readable():
    assert ROSTER_PATH.exists()
    records = read_roster_rows()
    assert [record["participant_id"] for record in records] == ["SB01", "SB02"]
