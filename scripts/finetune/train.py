# fine-tunes IntensifierTagger on the silver data from pipeline/silver.py; GPU (Colab T4) for real runs, CPU for smoke tests

import argparse
import gzip
import json
import os
import random
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as m

from silver import compare_rows, decode_rows, row_cols


def read_split(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rows(records, maps):
    frames = [decode_rows(rec, maps, token=True) for rec in records]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=row_cols + ["Token"])


def evaluate(model, tokenizer, vocab, maps, records, gold, args, device):
    model.eval()
    preds = m.predict_records(model, tokenizer, vocab, records, args.max_len, args.context, args.batch * 2, device)
    model.train()
    return compare_rows(gold, rows(preds, maps))


def encoder_groups(encoder, lr, decay):
    """encoder_groups gives layer k of n the lr times decay ** (n - k); embeddings get the lowest."""
    n = encoder.config.num_hidden_layers
    groups = {}
    for name, p in encoder.named_parameters():
        parts = name.split(".")
        depth = int(parts[parts.index("layer") + 1]) + 1 if "layer" in parts else 0
        groups.setdefault(depth, []).append(p)
    return [{"params": ps, "lr": lr * decay ** (n - d)} for d, ps in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/finetune")
    parser.add_argument("--model", default="bertin-project/bertin-roberta-base-spanish")
    parser.add_argument("--out", default="models/intensifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512, help="subwords per window")
    parser.add_argument("--context", type=int, default=64, help="subwords of unlabelled context on each side")
    parser.add_argument("--ptr-dim", type=int, default=256)
    parser.add_argument("--min-rule", type=int, default=2)
    parser.add_argument("--ptr", choices=["dot", "biaffine"], default="dot", help="pointer scorer")
    parser.add_argument("--dist", type=int, default=0, help="learned pointer bias per word offset up to +-dist (0 = off)")
    parser.add_argument("--smoothing", type=float, default=0.0, help="label smoothing of the classification heads")
    parser.add_argument("--llrd", type=float, default=1.0, help="encoder lr multiplied by this per layer below the top")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=0, help="use only this many files per split (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = {s: read_split(os.path.join(args.data, f"{s}.jsonl.gz")) for s in ("train", "dev", "test")}
    if args.limit:
        splits = {s: recs[: args.limit] for s, recs in splits.items()}
    with open(os.path.join(args.data, "maps.json"), encoding="utf-8") as f:
        maps = json.load(f)
    vocab = m.build_vocab(splits["train"], args.min_rule)
    print({h: len(v) for h, v in vocab.items()}, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    encoder = AutoModel.from_pretrained(args.model, add_pooling_layer=False)
    model = m.IntensifierTagger(encoder, vocab, args.ptr_dim, ptr=args.ptr, dist=args.dist).to(args.device)
    # RoBERTa positions start after the padding index, so 514 positions hold 512 subwords
    args.max_len = min(args.max_len, encoder.config.max_position_embeddings - 2)

    items, lost = [], 0
    for rec in splits["train"]:
        m.subword_counts(rec, tokenizer)
        windows = m.chunks(rec, args.max_len, args.context)
        lost += m.lost_pointers(rec, windows)
        items += [m.encode(rec, w, vocab, tokenizer, args.max_len) for w in windows]
    print(f"{len(items)} training windows, {lost} pointer targets outside their window", flush=True)
    gold = {s: rows(splits[s], maps) for s in ("dev", "test")}

    groups = encoder_groups(model.encoder, args.lr, args.llrd)
    groups.append({"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": args.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = args.epochs * ((len(items) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, step = -1.0, 0
    log = []
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(items)
        start, total = time.time(), 0.0
        for b in range(0, len(items), args.batch):
            batch = {
                k: v.to(args.device) for k, v in m.collate(items[b : b + args.batch], tokenizer.pad_token_id).items()
            }
            inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "first", "word_mask", "seg")}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**inputs)
            loss = m.loss_fn(out, batch, args.smoothing)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            total += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {total / (b // args.batch + 1):.4f} "
                      f"{time.time() - start:.0f}s", flush=True)  # fmt: skip
        dev = evaluate(model, tokenizer, vocab, maps, splits["dev"], gold["dev"], args, args.device)
        score = dev["select"]
        log.append({"epoch": epoch + 1, "loss": total / max(1, len(items) // args.batch), "dev": dev})
        rate = dev["adj_intensified_rate"]
        print(f"epoch {epoch + 1} dev host F1 {dev['host']['f1']:.4f} intensifier F1 {dev['intensifier']['f1']:.4f} "
              f"row F1 {dev['rows']['f1']:.4f} Adj_type kappa {dev['kappa']['Adj_type']:.3f} "
              f"ADJ intensified {rate['pred']:.2%} (rules {rate['gold']:.2%}) select {score:.4f}", flush=True)  # fmt: skip
        if score > best:
            best = score
            m.save(model, tokenizer, vocab, maps, args.out, vars(args))
            print(f"saved {args.out}", flush=True)

    model, tokenizer, vocab, maps, _ = m.load(args.out, args.device)
    test = evaluate(model, tokenizer, vocab, maps, splits["test"], gold["test"], args, args.device)
    print(f"test host F1 {test['host']['f1']:.4f} intensified F1 {test['intensified']['f1']:.4f} "
          f"row F1 {test['rows']['f1']:.4f}", flush=True)  # fmt: skip
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"epochs": log, "test": test, "best_dev_select": best}, f, indent=1)


if __name__ == "__main__":
    main()
