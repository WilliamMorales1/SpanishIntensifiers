# runs a fine-tuned IntensifierTagger over cleaned transcripts; writes the rows table (with a Token column, the host's
# word index) and <output>.records.jsonl.gz, the predicted labels pipeline/dataset.py applies hand corrections to

import argparse
import gzip
import json
import os
from datetime import date

import pandas as pd
import torch

import model as m
from dataset import clean_words, decode_rows, row_cols, word_segments
from output import drop_invariant_hosts, write_outputs


def clean_field(text):
    return (text or "").replace(" ", "").replace("_", "-")


def join_paragraphs(paragraphs):
    """join_paragraphs joins turns into one text, ending unpunctuated turns with a temporary " ." (its offsets)."""
    # without the period Stanza splits turn-final words (caser + o)
    parts, added, pos = [], set(), 0
    for p in paragraphs:
        if p[-1] not in ".,;:!?¡¿…)]\"'":
            added.add(pos + len(p) + 1)
            p += " ."
        parts.append(p)
        pos += len(p) + 2
    return "\n\n".join(parts), added


def words_from_tagged(path):
    with open(path, encoding="utf-8") as f:
        return [t.split("_", 1)[0] for t in f.read().split()]


def words_from_text(paths):
    """words_from_text tokenizes raw transcripts with Stanza's tokenizer only (CPU is fine)."""
    import stanza

    nlp = stanza.Pipeline("es", processors={"tokenize": "combined_charlm", "mwt": "combined"}, verbose=False)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            paragraphs = [line.strip() for line in f if line.strip()]
        if not paragraphs:
            yield []
            continue
        text, added = join_paragraphs(paragraphs)
        doc = nlp(text)
        yield [
            clean_field(w.text) for s in doc.sentences for t in s.tokens if t.start_char not in added for w in t.words
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/intensifier")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text", default=None, help="folder of cleaned transcripts (default data/text); Stanza tokenizes"
    )
    source.add_argument("--words", default=None, help="folder of word_UPOS_lemma files; only the words are used")
    parser.add_argument("--only", nargs="*", help="speaker names to run (default: every file)")
    parser.add_argument("--output", default="data/output/intensifier_data_model_" + date.today().isoformat())
    parser.add_argument("--speakers", help="speaker metadata sheet (xlsx with a Speaker column)")
    parser.add_argument("--drop-invariant", action="store_true")
    parser.add_argument("--xlsx", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    folder = args.words or args.text or "data/text"

    names = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    if args.only:
        names = [n for n in names if n[:-4] in set(args.only)]
    paths = [os.path.join(folder, n) for n in names]
    texts = (words_from_tagged(p) for p in paths) if args.words else words_from_text(paths)
    records = []
    for name, words in zip(names, texts):
        words = clean_words(words)
        if words:
            records.append({"speaker": name[:-4], "words": words, "seg": word_segments(words)})
    print(f"{len(records)} files, {sum(len(r['words']) for r in records)} words", flush=True)

    model, tokenizer, vocab, maps, margs = m.load(args.model, args.device)
    preds = m.predict_records(
        model, tokenizer, vocab, records, margs["max_len"], margs.get("context", 0), args.batch, args.device
    )
    frames = [decode_rows(p, maps, token=True) for p in preds]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=row_cols + ["Token"])
    print(f"{len(df)} rows, {(df['Int'] != '/').sum()} intensified", flush=True)
    if args.drop_invariant:
        df = drop_invariant_hosts(df)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_outputs(df, args.output, args.speakers, xlsx=args.xlsx, csv=args.csv)
    with gzip.open(args.output + ".records.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}.records.jsonl.gz")


if __name__ == "__main__":
    main()
