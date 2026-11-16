import pandas as pd

from evaluate import score


def test_score_counts_right_wrong_ambiguous_and_non_adj():
    gold = pd.DataFrame(
        [
            ("s", "es muy bueno", "bueno", "muy", "y", "predicative"),
            ("s", "casa grande", "grande", "/", "y", "attributive"),
            ("s", "qué rico", "rico", "/", "y", "predicative"),
            ("s", "loza cubiertos", "cubiertos", "/", "n", "/"),
            ("s", "muy triste", "triste", "muy", "y", "unclear"),
            ("s", "no está", "falta", "/", "y", "predicative"),
        ],
        columns=["Speaker", "Phr", "Adj", "Int", "gold_adj", "gold_type"],
    )
    output = pd.DataFrame(
        [
            ("s", "es muy bueno", "bueno", "muy", "predicative"),
            ("s", "casa grande", "grande", "/", "predicative"),
            ("s", "qué rico", "rico", "/", "ambiguous"),
            ("s", "loza cubiertos", "cubiertos", "/", "attributive"),
            ("s", "muy triste", "triste", "muy", "ambiguous"),
        ],
        columns=["Speaker", "Phr", "Adj", "Int", "Adj_type"],
    )
    assert score(gold, output) == {
        "gold rows": 6,
        "found in output": 5,
        "not really ADJ": 1,
        "real ADJ, clear Adj_type": 3,
        "right": 1,
        "wrong": 1,
        "left ambiguous": 1,
    }


def test_in_split_keeps_only_that_split(tmp_path):
    import json

    from evaluate import in_split

    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"train": ["a"], "dev": [], "test": ["b"]}))
    gold = pd.DataFrame({"Speaker": ["a", "b", "b"], "Phr": ["x", "y", "z"]})
    assert in_split(gold, path)["Phr"].tolist() == ["y", "z"]
