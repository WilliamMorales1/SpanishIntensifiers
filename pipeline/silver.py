# silver training data for the fine-tuned model (scripts/finetune): the rule pipeline's rows mapped back onto
# token positions, and decode_rows, which turns per-token labels (silver or predicted) back into output rows

import argparse
import difflib
import gzip
import json
import os
import random
import re
from collections import Counter, defaultdict

import pandas as pd

from extract import _modifier, degree_types, file_segments, fix_isimo_hosts, join_splits, process_segment, retag

row_cols = [
    "Speaker", "Phr_tagged", "Phr", "Double", "Int", "Int_type", "Adv", "Adj", "Lex_Adj", "Adj_type", "Noun",
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


def file_labels(speaker, tagged, parse=None):
    """file_labels returns a file's per-token silver labels (dict of lists) and the rule pipeline's rows for it."""
    tokens, parse, segments = file_segments(tagged, parse, speaker)
    degree = degree_types(retag(tokens))
    n = len(tokens)
    split = [_split(t) for t in tokens]
    rec = {"speaker": speaker, "words": [s[0] for s in split], "upos": [s[1] for s in split], "seg": [-1] * n}
    rec["lemma"] = [edit_rule(s[0], s[2]) for s in split]
    rec["host"] = ["O"] * n
    for name in host_labels:
        rec[name] = [None] * n

    rows, at = [], []
    for s_id, (segment, index) in enumerate(segments):
        words = retag(segment.split())
        for w, k in zip(words, index):
            if rec["seg"][k] == -1:
                rec["seg"][k] = s_id
                # a token cut inside ("..._PUNCT_.") keeps its own tags
                if _split(w)[0] == split[k][0]:
                    rec["upos"][k] = _split(w)[1]
        seg_parse = [parse[k] for k in index] if parse is not None else None
        seg_degree = [degree[k] for k in index]
        hosts = []
        seg_rows = process_segment(segment, speaker, seg_parse, seg_degree, hosts)
        lows = [_split(w)[0].lower() for w in words]
        span = range(len(words))
        for row, i in zip(seg_rows, hosts):
            modifier = _modifier(words, i, seg_degree, seg_parse[i] if seg_parse else None)
            free = modifier is not None and modifier[0] != "/" and modifier[0] == row["Int"]
            local = {
                "int": i - 1 if free else -1,
                "double": i - 2 if row["Double"] != "/" else -1,
                "neg": _nearest(lows, row["Negation"], i, span, before=True) if row["Negation"] != "/" else -1,
                "noun": _nearest(lows, row["Noun"], i, span) if row["Noun"] != "/" else -1,
                "verb": _nearest(lows, row["Verb"].lower(), i, span) if row["Verb"] != "/" else -1,
            }
            at.append(
                (index[i], {p: index[j] if j >= 0 else -1 for p, j in local.items()}, row["Int"] if not free else "/")
            )
            rows.append(row)

    df = fix_isimo_hosts(pd.DataFrame(rows, columns=row_cols)) if rows else pd.DataFrame(columns=row_cols)
    for (k, ptrs, bound), row in zip(at, df.to_dict("records")):
        if rec["host"][k] != "O":
            raise ValueError(f"{speaker}: token {k} is two hosts")
        adj = row["Adj"] != "/"
        low = rec["words"][k].lower()
        rec["host"][k] = "ADJ" if adj else "ADV"
        rec["bound"][k] = bound
        rec["adj_type"][k] = row["Adj_type"] if adj else None
        rec["form"][k] = edit_rule(low, row["Adj"] if adj else row["Adv"])
        rec["lex"][k] = edit_rule(low, row["Lex_Adj"]) if adj else None
        for p, j in ptrs.items():
            rec[p][k] = j
    return rec, df


punct = re.compile(r"[^\w\s]+")


def clean_words(words):
    """clean_words rejoins transcript splits (distin to, lejo s, muy bue no), the one cleaning step the rules do on words."""
    return [t.split("_", 1)[0] for t in join_splits([f"{w}_X_{w}" for w in words])[0]]


def word_segments(words):
    """word_segments gives each word its Phr segment id from the words alone (punctuation is PUNCT), as at inference."""
    tokens = [f"{w}_{'PUNCT' if punct.fullmatch(w) else 'X'}_{w}" for w in words]
    seg = [-1] * len(words)
    for s_id, (_, index) in enumerate(file_segments(tokens, join=False)[2]):
        for k in index:
            if seg[k] == -1:
                seg[k] = s_id
    return seg


def build_maps(records):
    """build_maps learns Int and Double from the intensifier word (mas -> más, menos -> más o menos) and Int_type from Int."""
    ints, doubles = defaultdict(Counter), defaultdict(Counter)
    for rec, df in records:
        hosts = [k for k, h in enumerate(rec["host"]) if h != "O"]
        for k, row in zip(hosts, df.to_dict("records")):
            if rec["int"][k] >= 0:
                ints[rec["words"][rec["int"][k]].lower()][row["Int"]] += 1
            if rec["double"][k] >= 0:
                doubles[rec["words"][rec["double"][k]].lower()][row["Double"]] += 1
    types = {}
    for _, df in records:
        types.update({i: t for i, t in zip(df["Int"], df["Int_type"]) if i != "/"})
    return {
        "int": {w: c.most_common(1)[0][0] for w, c in ints.items()},
        # a double menos is just menos (más o menos bien), unlike the Int menos
        "double": {w: c.most_common(1)[0][0] for w, c in doubles.items()},
        "int_type": types,
    }


def decode_rows(rec, maps, token=False):
    """decode_rows rebuilds output rows from per-token labels (words, upos, lemma rules, host labels, pointers)."""
    words = rec["words"]
    lemmas = [apply_rule(w, r) for w, r in zip(words, rec["lemma"])]
    tokens = [f"{w}_{u}_{lemma}" for w, u, lemma in zip(words, rec["upos"], lemmas)]
    _, _, segments = file_segments(tokens, join=False)
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
                    "Int_type": maps["int_type"].get(intensifier, "/"),
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


def apply_fixes(records, fixes):
    """apply_fixes overwrites silver labels with hand labels (evaluate.py tsv format) and returns the counts changed."""
    gold = {(r["Speaker"], r["Phr"], r["Adj"], r["Int"]): r for r in fixes.to_dict("records")}
    counts = Counter()
    for i, (rec, df) in enumerate(records):
        hosts = [k for k, h in enumerate(rec["host"]) if h != "O"]
        keep = []
        for k, row in zip(hosts, df.to_dict("records")):
            # written tables drop commas (write_outputs), and the hand labels were made on those
            g = gold.get(tuple(row[c].replace(",", "") for c in ("Speaker", "Phr", "Adj", "Int")))
            keep.append(g is None or g["gold_adj"] != "n")
            if g is None or row["Adj"] == "/":
                continue
            counts["matched"] += 1
            if g["gold_adj"] == "n":
                rec["host"][k] = "O"
                for name in host_labels:
                    rec[name][k] = None
                counts["not ADJ, host dropped"] += 1
            elif g["gold_type"] in ("attributive", "predicative") and g["gold_type"] != row["Adj_type"]:
                rec["adj_type"][k] = g["gold_type"]
                counts["Adj_type set"] += 1
        df = df[keep].copy()
        df.loc[:, "Adj_type"] = [
            rec["adj_type"][k] if rec["host"][k] == "ADJ" else a
            for k, a in zip([k for k, h in enumerate(rec["host"]) if h != "O"], df["Adj_type"])
        ]
        records[i] = (rec, df.reset_index(drop=True))
    return counts


def read_tagged(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split()


if __name__ == "__main__":
    from annotate import parse_relations

    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected", default="data/stanza_corrected")
    parser.add_argument("--parsed", default="data/stanza_parsed")
    parser.add_argument("--out", default="data/finetune")
    parser.add_argument("--dev", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--fixes",
        default="data/eval/retired/adj_gold_dev_2026-11-02.tsv",
        help="hand labels that override the rules (evaluate.py format); 'none' to skip",
    )
    args = parser.parse_args()
    if os.path.basename(args.fixes) == "adj_gold.tsv":
        raise SystemExit("adj_gold.tsv is the test set; never train on it")

    relations = parse_relations(args.parsed)
    names = sorted(f for f in os.listdir(args.corrected) if f.endswith(".txt"))
    records = []
    for name in names:
        tagged = read_tagged(os.path.join(args.corrected, name))
        if tagged:
            records.append(file_labels(name[:-4], tagged, relations(name)))
    if args.fixes != "none" and os.path.exists(args.fixes):
        fixes = pd.read_csv(args.fixes, sep="\t", keep_default_na=False)
        print(f"hand fixes from {args.fixes}:", dict(apply_fixes(records, fixes)))
    splits = split_speakers([r["speaker"] for r, _ in records], args.dev, args.test, args.seed)
    os.makedirs(args.out, exist_ok=True)
    by_speaker = {r["speaker"]: r for r, _ in records}
    for split, speakers in splits.items():
        with gzip.open(os.path.join(args.out, f"{split}.jsonl.gz"), "wt", encoding="utf-8") as f:
            for s in speakers:
                f.write(json.dumps(by_speaker[s], ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "maps.json"), "w", encoding="utf-8") as f:
        json.dump(build_maps(records), f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    rows = sum(len(df) for _, df in records)
    words = sum(len(r["words"]) for r, _ in records)
    # decoding the silver labels must give back the rule pipeline's rows, else the labels lost something
    maps = build_maps(records)
    diff = Counter()
    for rec, df in records:
        back = decode_rows(rec, maps)
        if len(back) != len(df):
            diff["row count"] += 1
            continue
        for c in row_cols:
            diff[c] += int((back[c].values != df[c].values).sum())
    print(f"{len(records)} files, {words} words, {rows} rows; splits {[(s, len(v)) for s, v in splits.items()]}")
    print("round-trip mismatches:", {c: n for c, n in diff.items() if n} or "none")
