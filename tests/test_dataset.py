import copy

import pandas as pd
import pytest

from dataset import (
    apply_rows,
    apply_rule,
    build_maps,
    clean_words,
    compare_rows,
    corpus,
    decode_rows,
    edit_rule,
    file_segments,
    host_labels,
    join_splits,
    merge,
    read_records,
    read_rows,
    row_cols,
    split_speakers,
    word_segments,
    write_dataset,
)
from output import write_outputs


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


def test_file_segments_maps_tokens():
    tokens = ["muy_ADV_muy", "bueno_ADJ_bueno", ",_PUNCT_,", "casa_NOUN_casa", "._PUNCT_."]
    assert [index for _, index in file_segments(tokens) if index] == [[0, 1, 2], [3, 4]]


def test_join_splits_keeps_tags_of_known_pairs():
    tokens = ["muy_ADV_muy", "bue_X_bue", "no_ADV_no", "cuida_VERB_cuidar", "do_NOUN_do"]
    assert join_splits(tokens) == ["muy_ADV_muy", "bueno_ADJ_bueno", "cuidado_ADJ_cuidado"]


words = "la casa es muy bonita , no está tan lejos . es riquísimo , más o menos bien bueno .".split()  # noqa: SIM905
upos = "DET NOUN AUX ADV ADJ PUNCT ADV VERB ADV ADV PUNCT AUX ADJ PUNCT ADV CCONJ ADV ADV ADJ PUNCT".split()  # noqa: SIM905
lemmas = "el casa ser muy bonito , no estar tan lejos . ser rico , más o menos bien bueno .".split()  # noqa: SIM905
# host index: (host, form, lex, adj_type, bound, int, double, neg, noun, verb)
hosts = {
    4: ("ADJ", "bonita", "bonito", "predicative", "/", 3, -1, -1, 1, 2),
    9: ("ADV", "lejos", None, None, "/", 8, -1, 6, -1, 7),
    12: ("ADJ", "rico", "rico", "predicative", "-ísimo", -1, -1, -1, -1, 11),
    18: ("ADJ", "bueno", "bueno", "ambiguous", "/", 17, 16, -1, -1, -1),
}


def _demo():
    n = len(words)
    rec = {"speaker": "DEMO_M45", "words": words, "upos": upos, "seg": word_segments(words)}
    rec["lemma"] = [edit_rule(w, lemma) for w, lemma in zip(words, lemmas)]
    rec["host"] = ["O"] * n
    rec.update({name: [None] * n for name in host_labels})
    for k, (host, form, lex, adj_type, bound, *ptrs) in hosts.items():
        rec["host"][k] = host
        rec["form"][k] = edit_rule(words[k], form)
        rec["lex"][k] = lex and edit_rule(words[k], lex)
        rec["adj_type"][k], rec["bound"][k] = adj_type, bound
        for p, v in zip(["int", "double", "neg", "noun", "verb"], ptrs):
            rec[p][k] = v
    maps = {
        "int": {"muy": "muy", "tan": "tan", "bien": "bien"},
        "double": {"menos": "menos"},
        "int_type": {"muy": "booster", "tan": "booster", "bien": "booster", "-ísimo": "booster"},
    }
    return rec, maps


def _cleared(rec):
    rec = copy.deepcopy(rec)
    rec["host"] = ["O"] * len(rec["words"])
    for name in host_labels:
        rec[name] = [None] * len(rec["words"])
    return rec


def test_decode_rows():
    rec, maps = _demo()
    df = decode_rows(rec, maps, token=True)
    assert df["Token"].tolist() == [4, 9, 12, 18]
    first = df.iloc[0]
    assert (first["Phr"], first["Int"], first["Adj"], first["Lex_Adj"], first["Noun"]) == (
        "la casa es muy bonita ,", "muy", "bonita", "bonito", "casa"
    )  # fmt: skip
    assert (first["Verb"], first["Inf_Verb"]) == ("es", "ser")
    assert df.iloc[1][["Adv", "Negation", "Int"]].tolist() == ["lejos", "no", "tan"]
    assert df.iloc[2][["Adj", "Int", "Int_type"]].tolist() == ["rico", "-ísimo", "booster"]
    assert df.iloc[3][["Double", "Int", "Adj_type"]].tolist() == ["menos", "bien", "ambiguous"]


@pytest.mark.parametrize("token", [True, False])
def test_apply_rows_round_trip(token):
    rec, maps = _demo()
    rows = decode_rows(rec, maps, token=True).astype(str)
    # predict.py writes no commas; hand-added rows have no Token
    rows = rows.apply(lambda c: c.str.replace(",", "", regex=False))
    if not token:
        rows["Token"] = ""
    back = _cleared(rec)
    assert apply_rows(back, rows, maps) == []
    assert back == rec


def test_apply_rows_corrections():
    rec, maps = _demo()
    rows = decode_rows(rec, maps, token=True).astype(str)
    rows = rows[rows["Adj"] != "rico"].copy()  # not a host
    rows.loc[rows["Adj"] == "bonita", ["Int", "Int_type", "Adj_type"]] = ["/", "/", "attributive"]
    rows.loc[rows["Adv"] == "lejos", "Negation"] = "/"
    added = rows.iloc[0].copy()
    added[["Adj", "Lex_Adj", "Token", "Int", "Noun", "Verb", "Inf_Verb"]] = ["casa", "caso", "", "/", "/", "/", "/"]
    rows = pd.concat([rows, added.to_frame().T])
    fixed = _cleared(rec)
    assert apply_rows(fixed, rows, maps) == []
    df = decode_rows(fixed, maps)
    assert df["Adj"].tolist() == ["casa", "bonita", "/", "bueno"]
    assert df.iloc[1][["Int", "Adj_type"]].tolist() == ["/", "attributive"]
    assert df.iloc[2]["Negation"] == "/" and df.iloc[2]["Int"] == "tan"
    assert fixed["host"][12] == "O"


def test_apply_rows_reports_problems():
    rec, maps = _demo()
    rows = decode_rows(rec, maps, token=True).astype(str).iloc[:1]
    bad = rows.assign(Token="", Phr="otra frase", Noun="perro")
    problems = apply_rows(_cleared(rec), pd.concat([rows.assign(Noun="perro"), bad]), maps)
    assert [why for _, why in problems] == ["Noun not found", "host not found"]


@pytest.mark.parametrize("ext", ["sqlite", "csv"])
def test_output_table_reads_back(tmp_path, ext):
    rec, maps = _demo()
    out = str(tmp_path / "out")
    write_outputs(decode_rows(rec, maps, token=True), out, csv=True)
    rows = read_rows(f"{out}.{ext}")
    assert list(rows) == row_cols + ["Token"]
    back = _cleared(rec)
    assert apply_rows(back, rows, maps) == []
    assert back == rec


def test_merge_keeps_splits_and_adds_new_files(tmp_path):
    rec, maps = _demo()
    old = {"train": [], "dev": [], "test": [dict(rec, host=["O"] * len(words))]}
    write_dataset(str(tmp_path), old, maps)
    new = dict(copy.deepcopy(rec), speaker="NEW_H01")
    rows = decode_rows(rec, maps, token=True).astype(str)
    rows = pd.concat([rows, rows.assign(Speaker="NEW_H01", Int="harto", Int_type="booster")])
    by_split, merged, problems = merge(str(tmp_path), [_cleared(rec), _cleared(new)], rows)
    assert problems == []
    assert [r["speaker"] for r in by_split["test"]] == ["DEMO_M45"] and by_split["test"][0] == rec
    assert [r["speaker"] for r in by_split["train"]] == ["NEW_H01"]
    # harto is not before any host there, so it is a bound Int
    assert merged["int_type"]["harto"] == "booster" and merged["int"]["muy"] == "muy"
    write_dataset(str(tmp_path), by_split, merged)
    assert read_records(str(tmp_path / "test.jsonl.gz")) == [rec]


def test_compare_rows():
    rec, maps = _demo()
    gold = decode_rows(rec, maps, token=True)
    assert compare_rows(gold, gold)["rows"]["f1"] == 1.0
    pred = gold.iloc[1:].copy()
    pred.loc[pred.index[0], "Adj_type"] = "wrong"
    scores = compare_rows(gold, pred)
    assert scores["host"]["r"] == (len(gold) - 1) / len(gold)
    assert scores["rows"]["pred"] == len(gold) - 1
    assert scores["rows"]["f1"] < scores["host"]["f1"]


def test_compare_rows_chance_corrected():
    rec, maps = _demo()
    gold = decode_rows(rec, maps, token=True)
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


def test_double_menos_keeps_its_word():
    rec = {"words": ["más", "o", "menos", "bien", "bueno"], "host": ["O", "O", "O", "O", "ADJ"], "int": [-1] * 5}
    rec["double"] = [-1] * 4 + [2]
    df = pd.DataFrame([{"Int": "bien", "Double": "menos", "Int_type": "booster"}])
    rec["int"][4] = 3
    maps = build_maps([(rec, df)])
    assert maps["double"] == {"menos": "menos"} and "menos" not in maps["int"]


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
