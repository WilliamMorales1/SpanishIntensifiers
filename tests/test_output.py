import sqlite3

import pandas as pd

from output import count_syllables, drop_invariant_hosts, write_outputs


def _rows():
    return pd.DataFrame(
        {
            "Speaker": ["S1", "S1", "S2"],
            "Phr": ["muy bonito, sí", "lejos", "bonito"],
            "Int": ["muy", "/", "/"],
            "Adv": ["/", "lejos", "/"],
            "Adj": ["bonito", "/", "bonito"],
            "Lex_Adj": ["bonito", "/", "bonito"],
            "Token": [3, 7, 0],
        }
    )


def test_count_syllables():
    assert count_syllables("bonito") == 3 and count_syllables("queso") == 2 and count_syllables("/") == 0


def test_drop_invariant_hosts():
    assert drop_invariant_hosts(_rows())["Adj"].tolist() == ["bonito", "bonito"]


def test_write_outputs_numbers_rows_and_keeps_token(tmp_path):
    out = str(tmp_path / "x")
    write_outputs(_rows(), out)
    with sqlite3.connect(out + ".sqlite") as con:
        df = pd.read_sql("SELECT * FROM intensifier_data", con)
    assert df["Number"].tolist() == ["S1_01", "S1_02", "S2_01"]
    assert df["Token"].tolist() == [3, 7, 0]
    assert df["Phr"].iloc[0] == "muy bonito sí"
