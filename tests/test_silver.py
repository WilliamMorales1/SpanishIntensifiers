import os
import sys

import pandas as pd
import pytest

from annotate import parse_relations
from extract import analyze, file_segments, process_segment
from silver import (
    apply_fixes,
    apply_rule,
    build_maps,
    clean_words,
    compare_rows,
    corpus,
    decode_rows,
    edit_rule,
    file_labels,
    read_tagged,
    row_cols,
    split_speakers,
    word_segments,
)

demo = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.mark.parametrize(
    ("form", "target"),
    [("buenas", "bueno"), ("es", "ser"), ("fue", "ir"), ("Casas", "casa"), ("muy", "muy"), ("lejísimos", "lejos")],
)
def test_edit_rule_round_trip(form, target):
    assert apply_rule(form, edit_rule(form, target)) == target


def test_edit_rule_generalizes():
    # the same rule lemmatizes other words with the same ending
    assert apply_rule("bonitas", edit_rule("buenas", "bueno")) == "bonito"
    assert apply_rule("a", edit_rule("buenas", "bueno")) == "a"


def test_process_segment_reports_host_index():
    hosts = []
    rows = process_segment("es_AUX_ser muy_ADV_muy bonito_ADJ_bonito ._PUNCT_.", "S01", hosts=hosts)
    assert len(rows) == 1 and hosts == [2]


def test_file_segments_maps_tokens():
    tokens = ["muy_ADV_muy", "bueno_ADJ_bueno", ",_PUNCT_,", "casa_NOUN_casa", "._PUNCT_."]
    _, _, segments = file_segments(tokens, join=False)
    assert [index for _, index in segments if index] == [[0, 1, 2], [3, 4]]


def _demo_labels():
    tagged = read_tagged(os.path.join(demo, "stanza_corrected", "DEMO_M45.txt"))
    return file_labels("DEMO_M45", tagged, parse_relations(os.path.join(demo, "stanza_parsed"))("DEMO_M45.txt"))


def test_demo_labels_decode_to_rule_rows():
    rec, df = _demo_labels()
    rules = analyze(os.path.join(demo, "stanza_corrected"), None, parse_relations(os.path.join(demo, "stanza_parsed")))
    pd.testing.assert_frame_equal(df[row_cols].reset_index(drop=True), rules[row_cols].reset_index(drop=True))
    back = decode_rows(rec, build_maps([(rec, df)]))
    pd.testing.assert_frame_equal(back, df[row_cols].reset_index(drop=True))
    assert (df["Int"] != "/").sum() > 0


def test_word_segments_match_tagged_segments():
    rec, _ = _demo_labels()
    assert word_segments(rec["words"]) == rec["seg"]


def test_compare_rows():
    rec, df = _demo_labels()
    gold = decode_rows(rec, build_maps([(rec, df)]), token=True)
    assert compare_rows(gold, gold)["rows"]["f1"] == 1.0
    pred = gold.iloc[1:].copy()
    pred.loc[pred.index[0], "Adj_type"] = "wrong"
    scores = compare_rows(gold, pred)
    assert scores["host"]["r"] == (len(gold) - 1) / len(gold)
    assert scores["rows"]["pred"] == len(gold) - 1
    assert scores["rows"]["f1"] < scores["host"]["f1"]


def test_compare_rows_chance_corrected():
    rec, df = _demo_labels()
    gold = decode_rows(rec, build_maps([(rec, df)]), token=True)
    same = compare_rows(gold, gold)
    assert same["kappa"]["Adj_type"] == 1.0
    assert same["adj_intensified_rate"]["gold"] == same["adj_intensified_rate"]["pred"]
    assert same["select"] == 1.0
    # predicting the majority Adj_type everywhere scores its share in accuracy but 0 kappa
    adj = gold[gold["Adj"] != "/"]
    pred = adj.assign(Adj_type=adj["Adj_type"].mode()[0])
    scores = compare_rows(adj, pred)
    assert abs(scores["kappa"]["Adj_type"]) < 1e-9
    assert scores["columns"]["Adj_type"] == scores["majority"]["Adj_type"]
    # dropping every intensifier shows up as a rate gap
    bare = gold.copy()
    bare["Int"] = "/"
    assert compare_rows(gold, bare)["adj_intensified_rate"]["pred"] == 0.0
    assert compare_rows(gold, bare)["intensifier"]["f1"] == 0.0


def test_corpus_groups_numbered_cartagena_files():
    assert corpus("22-Inés_CTG") == corpus("31-Pedro") == "CTG-num"
    assert corpus("TCali_H11_01_1996b_2019") == "TCali"


def test_split_speakers_by_file_and_corpus():
    speakers = [f"MEDE_H{i:02d}" for i in range(20)] + [f"TCali_M{i:02d}" for i in range(10)]
    splits = split_speakers(speakers, 0.1, 0.1, seed=1)
    assert sorted(s for names in splits.values() for s in names) == sorted(speakers)
    assert sum(s.startswith("TCali") for s in splits["test"]) == 1
    assert len(splits["dev"]) == 3 and len(splits["test"]) == 3


def _model():
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "finetune"))
    import model

    return model


def test_model_chunks_keep_segments_whole():
    model = _model()
    rec = {"words": list("abcdefgh"), "seg": [0, 0, 0, 1, 1, 2, 2, 2]}
    # max_len 7 leaves 5 subwords of core with no context
    assert model.chunks(rec, 7, 0) == [(0, 0, 5, 5), (5, 5, 8, 8)]
    # a segment longer than the window is cut
    assert model.chunks({"words": list("abcdefg"), "seg": [0] * 7}, 5, 0) == [(0, 0, 3, 3), (3, 3, 6, 6), (6, 6, 7, 7)]


def test_model_chunks_budget_subwords_and_add_context():
    model = _model()
    rec = {"words": list("abcdef"), "seg": [0] * 6, "nsub": [1, 3, 1, 1, 1, 1]}
    # 4 subwords of core, 1 of context each side
    # b (3 subwords) is too long to be context for the second window
    assert model.chunks(rec, 8, 1) == [(0, 0, 2, 3), (2, 2, 6, 6)]
    rec.update({p: [None] * 6 for p in model.pointers})
    rec["int"] = [None, None, 0, None, None, None]
    assert model.lost_pointers(rec, model.chunks(rec, 8, 1)) == 1
    assert model.lost_pointers(rec, model.chunks(rec, 40, 8)) == 0


def test_clean_words_joins_splits():
    assert clean_words(["muy", "bue", "no", "y", "distin", "to"]) == ["muy", "bueno", "y", "distinto"]
    # bue no without an intensifier is a restart
    assert clean_words(["bue", "no", "sé"]) == ["bue", "no", "sé"]


def test_demo_words_are_clean_words():
    rec, _ = _demo_labels()
    raw = [t.split("_", 1)[0] for t in read_tagged(os.path.join(demo, "stanza_corrected", "DEMO_M45.txt"))]
    assert clean_words(raw) == rec["words"]


def test_double_menos_keeps_its_word():
    rec = {"words": ["más", "o", "menos", "bien", "bueno"], "host": ["O", "O", "O", "O", "ADJ"], "int": [-1] * 5}
    rec["double"] = [-1] * 4 + [2]
    df = pd.DataFrame([{"Int": "bien", "Double": "menos", "Int_type": "booster"}])
    rec["int"][4] = 3
    maps = build_maps([(rec, df)])
    assert maps["double"] == {"menos": "menos"} and "menos" not in maps["int"]


def test_apply_fixes_overrides_hosts_and_types():
    rec, df = _demo_labels()
    adj = df[df["Adj"] != "/"]
    first, second = adj.iloc[0], adj.iloc[1]
    new = "predicative" if first["Adj_type"] != "predicative" else "attributive"
    fixes = pd.DataFrame(
        [
            {
                **{c: first[c].replace(",", "") for c in ["Speaker", "Phr", "Adj", "Int"]},
                "gold_adj": "y",
                "gold_type": new,
            },
            {
                **{c: second[c].replace(",", "") for c in ["Speaker", "Phr", "Adj", "Int"]},
                "gold_adj": "n",
                "gold_type": "/",
            },
        ]
    )
    records = [(rec, df)]
    counts = apply_fixes(records, fixes)
    assert counts["Adj_type set"] == 1 and counts["not ADJ, host dropped"] == 1
    rec2, df2 = records[0]
    assert len(df2) == len(df) - 1
    assert df2[(df2["Phr"] == first["Phr"]) & (df2["Adj"] == first["Adj"])]["Adj_type"].iloc[0] == new
    back = decode_rows(rec2, build_maps(records))
    pd.testing.assert_frame_equal(back, df2[row_cols].reset_index(drop=True))


def test_model_lexicon_keeps_majority_lemma_and_host_rules():
    model = _model()
    rec = {
        "words": ["Eres", "eres", "eres", "grandes"],
        "upos": ["AUX", "AUX", "AUX", "ADJ"],
        "lemma": [edit_rule("Eres", "ser"), edit_rule("eres", "ser"), edit_rule("eres", "erer"), "0||1|"],
        "host": ["O", "O", "O", "ADJ"],
        "form": [None, None, None, "0||0|"],
        "lex": [None, None, None, "0||1|"],
    }
    lex = model.build_lexicon([rec])
    assert lex["lemma"] == {"eres\tAUX": "ser", "grandes\tADJ": "grande"}
    assert lex["lex"] == {"grandes\tADJ": "0||1|"} and lex["form"] == {"grandes\tADJ": "0||0|"}
