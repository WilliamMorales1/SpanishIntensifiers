# training data for the fine-tuned model (train/, pipeline/model.py): per-token label records, decode_rows (records -> rows)
# and corrections (hand-corrected rows -> records); python pipeline/dataset.py merges corrected files into data/finetune

import argparse
import bisect
import difflib
import gzip
import json
import os
import random
import re
from collections import Counter, defaultdict

import pandas as pd

row_cols = [
    "Speaker", "Phr_tagged", "Phr", "Double", "Int", "Adv", "Adj", "Lex_Adj", "Adj_type", "Noun",
    "Negation", "Verb", "Inf_Verb",
]  # fmt: skip
# per-host labels; pointers hold a token index in the file, -1 for none
host_labels = ["bound", "adj_type", "form", "lex", "int", "double", "neg", "noun", "verb"]
pointers = ["int", "double", "neg", "noun", "verb"]


def edit_rule(form, target):
    """edit_rule returns the prefix/suffix edit that turns form into target (UDPipe-style lemma rule)."""
    if form == target:
        return "0||0|"
    a, b, n = difflib.SequenceMatcher(None, form, target, autojunk=False).find_longest_match(
        0, len(form), 0, len(target)
    )
    if n == 0:
        return "=" + target
    return f"{a}|{target[:b]}|{len(form) - a - n}|{target[b + n :]}"


def apply_rule(form, rule):
    """apply_rule applies an edit_rule to form; a rule that cuts more than form has leaves form unchanged."""
    if rule.startswith("="):
        return rule[1:]
    cut_pre, add_pre, cut_suf, add_suf = rule.split("|")
    cut_pre, cut_suf = int(cut_pre), int(cut_suf)
    if cut_pre + cut_suf >= len(form) and form:
        return form
    return add_pre + form[cut_pre : len(form) - cut_suf] + add_suf


def _split(token):
    parts = token.split("_", 2)
    return parts if len(parts) == 3 else [token, "X", token]


def _nearest(texts, target, i, span, before=False):
    # the rows only say which word, so the occurrence nearest the host is taken
    found = [j for j in span if texts[j] == target and (not before or j < i)]
    return min(found, key=lambda j: abs(j - i)) if found else -1


punct = re.compile(r"[^\w\s]+")


# transcript splits of one word, rejoined before labeling (muy bue no -> bueno)
split_words = {
    ("distin", "to"): ("ADJ", "distinto"),
    ("distin", "ta"): ("ADJ", "distinto"),
    ("lejo", "s"): ("ADV", "lejos"),
    ("calma", "do"): ("ADJ", "calmado"),
    ("cuida", "do"): ("ADJ", "cuidado"),
    ("horri", "bles"): ("ADJ", "horrible"),
    ("bue", "no"): ("ADJ", "bueno"),
    ("entonce", "s"): ("ADV", "entonces"),
    ("automática", "mente"): ("ADV", "automáticamente"),
    ("actual", "mente"): ("ADV", "actualmente"),
}
intensifiers = {
    "absolutamente", "bastante", "casi", "completamente", "demasiado", "estrictamente", "especialmente",
    "extremadamente", "harto", "increíblemente", "mucho", "muy", "plenamente", "realmente", "sumamente", "súper",
    "totalmente", "verdaderamente", "bien", "super", "supel", "re", "requete", "recontra", "archi",
}  # fmt: skip


def join_splits(words):
    """join_splits rejoins split_words pairs of word_UPOS_lemma tokens into one token."""
    out, i = [], 0
    while i < len(words):
        first = words[i].split("_")
        second = words[i + 1].split("_") if i + 1 < len(words) else []
        key = (first[0].lower(), second[0].lower()) if len(first) == 3 and len(second) == 3 else None
        # bue no son buenos restarts; only muy bue no is one word
        if key in split_words and (key[0] != "bue" or (i > 0 and words[i - 1].split("_")[0] in intensifiers)):
            upos, lemma = split_words[key]
            # no tiene mucho cuida do: the noun cuidado
            if key == ("cuida", "do") and first[1] == "NOUN":
                upos = "NOUN"
            out.append(f"{first[0]}{second[0]}_{upos}_{lemma}")
            i += 2
            continue
        out.append(words[i])
        i += 1
    return out


def file_segments(tokens):
    """file_segments splits word_UPOS_lemma tokens into Phr segments at , . ! ? (segment text, token index per word)."""
    content = " ".join(tokens)
    starts = [m.start() for m in re.finditer(r"\S+", content)]
    out, offset = [], 0
    for segment in re.split(r"(?<=[,.!?]_PUNCT_[,.!?])", content):
        # the split can fall inside a token ("..._PUNCT_..."), so map by character offset
        index = [bisect.bisect_right(starts, offset + m.start()) - 1 for m in re.finditer(r"\S+", segment)]
        offset += len(segment)
        out.append((segment, index))
    return out


def clean_words(words):
    """clean_words rejoins transcript splits (distin to, lejo s, muy bue no) in a word list."""
    return [t.split("_", 1)[0] for t in join_splits([f"{w}_X_{w}" for w in words])]


def word_segments(words):
    """word_segments gives each word its Phr segment id from the words alone (punctuation is PUNCT), as at inference."""
    tokens = [f"{w}_{'PUNCT' if punct.fullmatch(w) else 'X'}_{w}" for w in words]
    seg = [-1] * len(words)
    for s_id, (_, index) in enumerate(file_segments(tokens)):
        for k in index:
            if seg[k] == -1:
                seg[k] = s_id
    return seg


def build_maps(records):
    """build_maps learns Int and Double from the intensifier word (mas -> más, menos -> más o menos)."""
    ints, doubles = defaultdict(Counter), defaultdict(Counter)
    for rec, df in records:
        hosts = [k for k, h in enumerate(rec["host"]) if h != "O"]
        for k, row in zip(hosts, df.to_dict("records")):
            if rec["int"][k] >= 0:
                ints[rec["words"][rec["int"][k]].lower()][row["Int"]] += 1
            if rec["double"][k] >= 0:
                doubles[rec["words"][rec["double"][k]].lower()][row["Double"]] += 1
    return {
        "int": {w: c.most_common(1)[0][0] for w, c in ints.items()},
        # a double menos is just menos (más o menos bien), unlike the Int menos
        "double": {w: c.most_common(1)[0][0] for w, c in doubles.items()},
    }


def decode_rows(rec, maps, token=False):
    """decode_rows rebuilds output rows from per-token labels (words, upos, lemma rules, host labels, pointers)."""
    words = rec["words"]
    lemmas = [apply_rule(w, r) for w, r in zip(words, rec["lemma"])]
    tokens = [f"{w}_{u}_{lemma}" for w, u, lemma in zip(words, rec["upos"], lemmas)]
    segments = file_segments(tokens)
    rows, seen = [], set()

    def word(j, table="int"):
        return maps.get(table, {}).get(words[j].lower(), words[j].lower()) if j >= 0 else "/"

    for segment, index in segments:
        seg_words = segment.split()
        phrase_tagged = " ".join(seg_words)
        phrase = " ".join(re.sub(r"_.*", "", w) for w in seg_words)
        for k in index:
            if k in seen or rec["host"][k] == "O":
                continue
            seen.add(k)
            adj = rec["host"][k] == "ADJ"
            low = words[k].lower()
            bound = rec["bound"][k] or "/"
            intensifier = bound if bound != "/" else word(rec["int"][k])
            noun, verb, neg = rec["noun"][k], rec["verb"][k], rec["neg"][k]
            form = apply_rule(low, rec["form"][k])
            rows.append(
                {
                    "Speaker": rec["speaker"],
                    "Phr_tagged": phrase_tagged,
                    "Phr": phrase,
                    "Double": word(rec["double"][k], "double"),
                    "Int": intensifier,
                    "Adv": "/" if adj else form,
                    "Adj": form if adj else "/",
                    "Lex_Adj": apply_rule(low, rec["lex"][k]) if adj else "/",
                    "Adj_type": rec["adj_type"][k] if adj else "/",
                    "Noun": words[noun].lower() if adj and noun >= 0 else "/",
                    "Negation": words[neg].lower() if neg >= 0 else "/",
                    "Verb": words[verb] if verb >= 0 else "/",
                    "Inf_Verb": lemmas[verb] if verb >= 0 else "/",
                    "Token": k,
                }
            )
    return pd.DataFrame(rows, columns=row_cols + ["Token"]) if token else pd.DataFrame(rows, columns=row_cols)


def _f1(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "pred": n_pred, "gold": n_gold}


def corpus(speaker):
    """corpus names a speaker's corpus (MEDE, TCali, ...); numbered Cartagena files (22-Inés_CTG_...) are CTG-num."""
    return "CTG-num" if speaker[:1].isdigit() else speaker.split("_")[0]


def _kappa(g, p):
    # Cohen's kappa: accuracy above what guessing from each side's label frequencies would get
    if not len(g):
        return 0.0
    acc = float((g.values == p.values).mean())
    gf, pf = g.value_counts(normalize=True), p.value_counts(normalize=True)
    chance = float(sum(gf[v] * pf.get(v, 0.0) for v in gf.index))
    return (acc - chance) / (1 - chance) if chance < 1 else 1.0


def _adj_rate(df):
    adj = df[df["Adj"] != "/"]
    return float((adj["Int"] != "/").mean()) if len(adj) else 0.0


def compare_rows(gold, pred):
    """compare_rows scores predicted rows against reference rows (both from decode_rows with token=True)."""
    key = ["Speaker", "Token"]
    both = gold.merge(pred, on=key, suffixes=("_g", "_p"))
    out = {"host": _f1(len(both), len(pred), len(gold))}
    out["host_class"] = _f1(int(((both["Adj_g"] != "/") == (both["Adj_p"] != "/")).sum()), len(pred), len(gold))
    g_int, p_int = gold[gold["Int"] != "/"], pred[pred["Int"] != "/"]
    hit = g_int.merge(p_int, on=key, suffixes=("_g", "_p"))
    out["intensified"] = _f1(len(hit), len(p_int), len(g_int))
    out["intensifier"] = _f1(int((hit["Int_g"] == hit["Int_p"]).sum()), len(p_int), len(g_int))
    cols = row_cols[1:]
    out["columns"] = {c: float((both[c + "_g"] == both[c + "_p"]).mean()) if len(both) else 0.0 for c in cols}
    # most columns are mostly "/", so accuracy alone looks high; majority and kappa show what is above chance
    out["majority"] = {
        c: float(both[c + "_g"].value_counts(normalize=True).iloc[0]) if len(both) else 0.0 for c in cols
    }
    out["kappa"] = {c: _kappa(both[c + "_g"], both[c + "_p"]) for c in cols}
    # rare columns: F1 of the rows that have a value, the value must match
    out["filled"] = {}
    for c in ("Double", "Negation", "Noun", "Verb"):
        g, p = both[c + "_g"] != "/", both[c + "_p"] != "/"
        tp = int((g & p & (both[c + "_g"] == both[c + "_p"])).sum())
        out["filled"][c] = _f1(tp, int(p.sum()), int(g.sum()))
    adj = both[(both["Adj_g"] != "/") & (both["Adj_p"] != "/")]
    f1s = []
    for t in ("attributive", "predicative", "ambiguous"):
        g, p = adj["Adj_type_g"] == t, adj["Adj_type_p"] == t
        f1s.append(_f1(int((g & p).sum()), int(p.sum()), int(g.sum()))["f1"])
    out["adj_type_macro_f1"] = sum(f1s) / len(f1s)
    strict = [c for c in row_cols[2:]]
    match = (both[[c + "_g" for c in strict]].values == both[[c + "_p" for c in strict]].values).all(axis=1)
    # a row is right only when every column but Phr_tagged matches
    out["rows"] = _f1(int(match.sum()), len(pred), len(gold))
    # the study's number: share of adjective rows intensified, which a biased model shifts
    out["adj_intensified_rate"] = {"gold": _adj_rate(gold), "pred": _adj_rate(pred)}
    out["by_corpus"] = {}
    for c in sorted({corpus(s) for s in gold["Speaker"]}):
        g = gold[gold["Speaker"].map(corpus) == c]
        p = pred[pred["Speaker"].map(corpus) == c]
        m = int(match[(both["Speaker"].map(corpus) == c).values].sum())
        out["by_corpus"][c] = {
            "rows_f1": _f1(m, len(p), len(g))["f1"],
            "rate_gold": _adj_rate(g),
            "rate_pred": _adj_rate(p),
        }
    # intensified rows are a tenth of all rows, so row F1 alone would let them slide
    out["select"] = (out["rows"]["f1"] + out["intensifier"]["f1"]) / 2
    return out


def split_speakers(speakers, dev=0.1, test=0.1, seed=13):
    """split_speakers assigns whole files to train/dev/test, stratified by corpus (MEDE, TCali, ...)."""
    groups = defaultdict(list)
    for s in sorted(speakers):
        groups[corpus(s)].append(s)
    out = {"train": [], "dev": [], "test": []}
    rng = random.Random(seed)
    for names in groups.values():
        rng.shuffle(names)
        n_dev, n_test = round(len(names) * dev), round(len(names) * test)
        out["test"] += names[:n_test]
        out["dev"] += names[n_test : n_test + n_dev]
        out["train"] += names[n_test + n_dev :]
    return out


def read_rows(path):
    """read_rows loads an output table (.sqlite, .xlsx or .csv) as strings, keeping only row_cols and Token."""
    if path.endswith(".sqlite"):
        import sqlite3

        with sqlite3.connect(path) as con:
            df = pd.read_sql("SELECT * FROM intensifier_data", con)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "Token" not in df:
        df["Token"] = ""
    df = df[row_cols + ["Token"]].fillna("/").astype(str)
    return df.replace("", "/").assign(Token=df["Token"].replace("/", ""))


def _plain(text):
    # write_outputs drops commas from every cell
    return text.replace(",", "")


def _find_host(rec, row, taken):
    # a row added by hand may have no Token: find its Phr segment, then the host word in it
    # punctuation is easy to get wrong by hand, so only the words must match
    phrase = [w for w in _plain(row["Phr"]).split() if not punct.fullmatch(w)]
    target = (row["Adj"] if row["Adj"] != "/" else row["Adv"]).lower()
    segs = defaultdict(list)
    for k, s in enumerate(rec["seg"]):
        segs[s].append(k)
    for ks in segs.values():
        if [rec["words"][k] for k in ks if not punct.fullmatch(rec["words"][k])] != phrase:
            continue
        # Adj can be a rebuilt base (riquísimo -> rico), so the word's lemma also counts
        hits = [k for k in ks if rec["words"][k].lower() == target] or [
            k for k in ks if apply_rule(rec["words"][k], rec["lemma"][k]).lower() == target
        ]
        if hits:
            # a host already taken comes back so apply_rows reports the duplicate
            return next((k for k in hits if k not in taken), hits[0])
    return None


def apply_rows(rec, rows, maps):
    """apply_rows sets a record's host labels to exactly the given rows (one file's corrected table); returns problems."""
    n = len(rec["words"])
    for name in ["host", *host_labels]:
        rec[name] = ["O" if name == "host" else None] * n
    lows = [w.lower() for w in rec["words"]]
    problems, taken = [], set()
    for row in rows.to_dict("records"):
        k = int(float(row["Token"])) if str(row["Token"]).strip() else _find_host(rec, row, taken)
        if k is None or not 0 <= k < n:
            problems.append((row, "host not found"))
            continue
        if k in taken:
            problems.append((row, "two rows on one host"))
            continue
        taken.add(k)
        span = [j for j in range(n) if rec["seg"][j] == rec["seg"][k]]
        _apply_tags(rec, row, span)
        adj = row["Adj"] != "/"
        rec["host"][k] = "ADJ" if adj else "ADV"
        rec["form"][k] = edit_rule(lows[k], row["Adj"] if adj else row["Adv"])
        rec["lex"][k] = edit_rule(lows[k], row["Lex_Adj"]) if adj else None
        rec["adj_type"][k] = row["Adj_type"] if adj else None
        rec["int"][k], rec["bound"][k], rec["double"][k] = -1, "/", -1
        if row["Int"] != "/":
            # a free intensifier is a word before the host in its segment; anything else (-ísimo) is bound
            free = [j for j in span if j < k and maps["int"].get(lows[j], lows[j]) == row["Int"]]
            if free:
                rec["int"][k] = max(free)
            else:
                rec["bound"][k] = row["Int"]
        if row["Double"] != "/":
            before = rec["int"][k] if rec["int"][k] >= 0 else k
            found = [j for j in span if j < before and maps["double"].get(lows[j], lows[j]) == row["Double"]]
            rec["double"][k] = max(found) if found else -1
            if not found:
                problems.append((row, "Double not found"))
        for p, col, before in (("neg", "Negation", True), ("noun", "Noun", False), ("verb", "Verb", False)):
            value = row[col].lower()
            rec[p][k] = -1 if value == "/" or (p == "noun" and not adj) else _nearest(lows, value, k, span, before)
            if value != "/" and rec[p][k] < 0 and not (p == "noun" and not adj):
                problems.append((row, f"{col} not found"))
        if rec["verb"][k] >= 0 and row["Inf_Verb"] != "/":
            j = rec["verb"][k]
            rec["lemma"][j] = edit_rule(rec["words"][j], row["Inf_Verb"])
    return problems


def _apply_tags(rec, row, span):
    # Phr_tagged edits (word_UPOS_lemma) fix the segment's UPOS and lemmas
    tagged = _plain(row["Phr_tagged"]).split()
    words = [j for j in span if _plain(rec["words"][j])]
    tagged = [t for t in tagged if not t.startswith("_")]
    if len(tagged) != len(words):
        return
    for j, t in zip(words, tagged):
        w, upos, lemma = _split(t)
        if w == _plain(rec["words"][j]):
            rec["upos"][j] = upos
            rec["lemma"][j] = edit_rule(rec["words"][j], lemma)


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_dataset(out, by_split, maps):
    os.makedirs(out, exist_ok=True)
    for split, recs in by_split.items():
        with gzip.open(os.path.join(out, f"{split}.jsonl.gz"), "wt", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "maps.json"), "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({s: [r["speaker"] for r in recs] for s, recs in by_split.items()}, f, indent=1)


def merge(dataset, records, rows, split="train"):
    """merge applies corrected rows to predicted records and adds them to the dataset folder's splits.

    Each record's file gets exactly its rows in the table (none if it has none). A file already in the dataset is
    replaced and keeps its split; a new file goes to split. Returns (records per split, maps, problems).
    """
    by_split = {s: [] for s in ("train", "dev", "test")}
    maps = {"int": {}, "double": {}}
    if os.path.exists(os.path.join(dataset, "maps.json")):
        for s in by_split:
            path = os.path.join(dataset, f"{s}.jsonl.gz")
            by_split[s] = read_records(path) if os.path.exists(path) else []
        with open(os.path.join(dataset, "maps.json"), encoding="utf-8") as f:
            maps = json.load(f)
        # Int_type moved to pipeline/int_type.py; older datasets still carry it
        maps.pop("int_type", None)
    old_split = {r["speaker"]: s for s, recs in by_split.items() for r in recs}
    rows = rows.copy()
    rows["Int"] = rows["Int"].str.lower()
    # an intensifier word the maps have never seen stands for itself
    for col, table in (("Int", "int"), ("Double", "double")):
        maps[table] = {w: w for w in rows[col] if w != "/"} | maps[table]
    problems = []
    for rec in records:
        problems += [
            (rec["speaker"], r, why) for r, why in apply_rows(rec, rows[rows["Speaker"] == rec["speaker"]], maps)
        ]
        s = old_split.get(rec["speaker"], split)
        by_split = {t: [r for r in recs if r["speaker"] != rec["speaker"]] for t, recs in by_split.items()}
        by_split[s].append(rec)
    missing = sorted(set(rows["Speaker"]) - {r["speaker"] for r in records})
    problems += [(s, None, "speaker not in the records file") for s in missing]
    return by_split, maps, problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="add hand-corrected predict.py output to the training data (see docs/finetune.md)"
    )
    parser.add_argument("records", help="<output>.records.jsonl.gz written by predict.py")
    parser.add_argument("rows", help="the corrected table (.xlsx, .csv or .sqlite) of the same files")
    parser.add_argument("--dataset", default="data/finetune", help="dataset folder to update (made if missing)")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"], help="split for new files")
    args = parser.parse_args()

    records = read_records(args.records)
    by_split, maps, problems = merge(args.dataset, records, read_rows(args.rows), args.split)
    for speaker, row, why in problems:
        print(f"{speaker}: {why}" + (f": {row['Phr']} | {row['Adj']}{row['Adv']} | Int {row['Int']}" if row else ""))
    write_dataset(args.dataset, by_split, maps)
    print(
        f"{len(records)} files merged ({len(problems)} problems); dataset "
        + ", ".join(f"{s} {len(v)} files" for s, v in by_split.items())
    )
