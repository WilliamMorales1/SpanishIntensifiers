import json

import pandas as pd

from int_type import add_int_type, load_map, read_table, write_table


def test_add_int_type_places_and_maps(capsys):
    df = pd.DataFrame({"Speaker": ["S1"] * 4, "Int": ["muy", "/", "Casi", "pues"], "Adj": ["a", "b", "c", "d"]})
    out = add_int_type(df, {"muy": "booster", "casi": "approximator"})
    assert list(out.columns) == ["Speaker", "Int", "Int_type", "Adj"]
    assert out["Int_type"].tolist() == ["booster", "/", "approximator", "?"]
    assert "pues" in capsys.readouterr().out


def test_add_int_type_replaces_old_column():
    df = pd.DataFrame({"Int": ["bastante"], "Int_type": ["booster"], "Adj": ["bueno"]})
    out = add_int_type(df, {"bastante": "compromiser"})
    assert out["Int_type"].tolist() == ["compromiser"] and list(out.columns) == ["Int", "Int_type", "Adj"]


def test_default_map_covers_intensifier_words():
    mapping = load_map()
    assert mapping["muy"] == "booster" and mapping["más o menos"] == "compromiser"
    assert set(mapping.values()) <= {"maximizer", "booster", "approximator", "compromiser", "emphasizer"}


def test_relabel_sqlite_round_trip(tmp_path):
    from output import write_outputs

    rows = pd.DataFrame(
        {"Speaker": ["S1"], "Phr_tagged": ["x"], "Phr": ["muy bueno"], "Int": ["muy"], "Adj": ["bueno"]}
    )
    write_outputs(add_int_type(rows, load_map()), str(tmp_path / "t"))
    custom = tmp_path / "map.json"
    custom.write_text(json.dumps({"muy": "intensifier"}), encoding="utf-8")
    path = str(tmp_path / "t.sqlite")
    write_table(add_int_type(read_table(path), load_map(str(custom))), path)
    assert read_table(path)["Int_type"].tolist() == ["intensifier"]
