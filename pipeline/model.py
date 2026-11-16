# multi-head token model for the fine-tuned intensifier tagger: per-word UPOS, lemma rule and host class, per-host
# labels, and pointers from each host to its intensifier, double, negator, noun and verb (same Phr segment only)

import json
import math
import os
from collections import Counter

import torch
from torch import nn

from dataset import apply_rule, edit_rule, pointers

HUB_MODEL = "Hydragon123/spanish-intensifier-tagger"

token_heads = ["upos", "lemma", "host"]
host_heads = ["bound", "adj_type", "form", "lex"]
unk = "<unk>"
identity = "0||0|"


def build_vocab(records, min_rule=2):
    """build_vocab lists the classes of every head; edit rules seen fewer than min_rule times become <unk>."""
    counts = {h: Counter() for h in token_heads + host_heads}
    for rec in records:
        for h, c in counts.items():
            c.update(v for v in rec[h] if v is not None)
    vocab = {}
    for h, c in counts.items():
        if h in ("lemma", "form", "lex"):
            vocab[h] = [unk] + sorted(v for v, n in c.items() if n >= min_rule)
        else:
            vocab[h] = sorted(c)
    vocab["lexicon"] = build_lexicon(records)
    return vocab


def build_lexicon(records):
    """build_lexicon maps "word<TAB>upos" to its most common lemma, and "word<TAB>host class" to its most common
    form and lex rule; predict_records uses them over the model's edit rules, which invent irregular lemmas."""
    counts = {h: {} for h in ("lemma", "form", "lex")}
    for rec in records:
        for k, w in enumerate(rec["words"]):
            low = w.lower()
            counts["lemma"].setdefault(f"{low}\t{rec['upos'][k]}", Counter())[apply_rule(w, rec["lemma"][k])] += 1
            if rec["host"][k] == "O":
                continue
            for h in ("form", "lex"):
                if rec[h][k] is not None:
                    counts[h].setdefault(f"{low}\t{rec['host'][k]}", Counter())[rec[h][k]] += 1
    return {h: {key: c.most_common(1)[0][0] for key, c in table.items()} for h, table in counts.items()}


def subword_counts(rec, tokenizer):
    """subword_counts stores in rec["nsub"] how many subwords the tokenizer gives each word."""
    if "nsub" not in rec:
        ids = tokenizer(rec["words"], is_split_into_words=True, add_special_tokens=False).word_ids()
        rec["nsub"] = [max(1, c) for c in _per_word(ids, len(rec["words"]))]
    return rec["nsub"]


def _per_word(word_ids, n):
    out = [0] * n
    for w in word_ids:
        if w is not None:
            out[w] += 1
    return out


def chunks(rec, max_len, context=64):
    """chunks cuts a file into windows (context start, start, end, context end) that fit max_len subwords.

    The core [start, end) holds whole Phr segments (a longer segment is cut) and gets labels; up to context subwords
    of the words on each side are seen too, so a cut segment's pointers can reach past the core.
    """
    n = len(rec["words"])
    nsub = rec.get("nsub") or [1] * n
    budget = max(1, max_len - 2 - 2 * context)
    cores, start, used, k = [], 0, 0, 0
    while k < n:
        end, size = k, 0
        while end < n and rec["seg"][end] == rec["seg"][k]:
            size += nsub[end]
            end += 1
        if used + size > budget and k > start:
            cores.append((start, k))
            start, used = k, 0
        if size > budget:
            # a segment over the budget is cut word by word
            for j in range(k, end):
                if used + nsub[j] > budget and j > start:
                    cores.append((start, j))
                    start, used = j, 0
                used += nsub[j]
        else:
            used += size
        k = end
    if start < n:
        cores.append((start, n))
    out = []
    for s, e in cores:
        cs, room = s, context
        while cs > 0 and nsub[cs - 1] <= room:
            cs -= 1
            room -= nsub[cs]
        ce, room = e, context
        while ce < n and nsub[ce] <= room:
            room -= nsub[ce]
            ce += 1
        out.append((cs, s, e, ce))
    return out


def lost_pointers(rec, windows):
    """lost_pointers counts pointer targets outside their host's window (the model cannot learn those)."""
    lost = 0
    for cs, s, e, ce in windows:
        for k in range(s, e):
            for p in pointers:
                j = rec[p][k]
                lost += j is not None and j >= 0 and not cs <= j < ce
    return lost


_index = {}


def encode(rec, window, vocab, tokenizer, max_len, labels=True):
    """encode turns a chunks window of a record into model inputs, plus label ids for the core words when labels is set."""
    cs, start, end, ce = window
    words = rec["words"][cs:ce]
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len)
    first = [-1] * len(words)
    for pos, w in enumerate(enc.word_ids()):
        if w is not None and first[w] == -1:
            first[w] = pos
    item = {
        "input_ids": enc["input_ids"],
        "first": [max(f, 0) for f in first],
        "word_mask": [f >= 0 for f in first],
        "seg": rec["seg"][cs:ce],
    }
    if not labels:
        return item
    if id(vocab) not in _index:
        _index[id(vocab)] = {h: {v: i for i, v in enumerate(vocab[h])} for h in token_heads + host_heads}
    index = _index[id(vocab)]
    for h in token_heads + host_heads:
        ids = []
        for k in range(cs, ce):
            v = rec[h][k]
            ids.append(-100 if v is None else index[h].get(v, 0))
        item[h] = ids
    for p in pointers:
        ids = []
        for k in range(cs, ce):
            j = rec[p][k]
            if j is None:
                ids.append(-100)
            elif j < 0:
                ids.append(0)
            else:
                ids.append(j - cs + 1 if cs <= j < ce else -100)
        item[p] = ids
    # context words are labelled in their own window; truncated words are not seen
    core = [start <= k < end for k in range(cs, ce)]
    for h in token_heads + host_heads + pointers:
        item[h] = [v if m and c else -100 for v, m, c in zip(item[h], item["word_mask"], core)]
    return item


def collate(items, pad_id):
    """collate pads a batch of encode outputs into tensors."""
    n_sub = max(len(it["input_ids"]) for it in items)
    n_words = max(len(it["first"]) for it in items)
    out = {
        "input_ids": torch.full((len(items), n_sub), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(items), n_sub), dtype=torch.long),
        "first": torch.zeros((len(items), n_words), dtype=torch.long),
        "word_mask": torch.zeros((len(items), n_words), dtype=torch.bool),
        "seg": torch.full((len(items), n_words), -1, dtype=torch.long),
    }
    label_names = [k for k in items[0] if k in token_heads + host_heads + pointers]
    for k in label_names:
        out[k] = torch.full((len(items), n_words), -100, dtype=torch.long)
    for b, it in enumerate(items):
        out["input_ids"][b, : len(it["input_ids"])] = torch.tensor(it["input_ids"])
        out["attention_mask"][b, : len(it["input_ids"])] = 1
        w = len(it["first"])
        out["first"][b, :w] = torch.tensor(it["first"])
        out["word_mask"][b, :w] = torch.tensor(it["word_mask"])
        out["seg"][b, :w] = torch.tensor(it["seg"])
        for k in label_names:
            out[k][b, :w] = torch.tensor(it[k])
    return out


class IntensifierTagger(nn.Module):
    """IntensifierTagger is a transformer encoder with classification heads on each word's first subword and pointer heads."""

    def __init__(self, encoder, vocab, ptr_dim=256, dropout=0.1, ptr="dot", dist=0):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, len(vocab[h])) for h in token_heads + host_heads})

        def proj():
            if ptr == "biaffine":
                return nn.Sequential(nn.Linear(hidden, ptr_dim), nn.GELU(), nn.Dropout(dropout))
            return nn.Linear(hidden, ptr_dim)

        # keys are prefixed: "double" is an nn.Module method name
        self.query = nn.ModuleDict({"p_" + p: proj() for p in pointers})
        self.key = nn.ModuleDict({"p_" + p: proj() for p in pointers})
        # learned key for "no intensifier / no noun ..."
        self.null = nn.ParameterDict({"p_" + p: nn.Parameter(torch.randn(ptr_dim) * 0.02) for p in pointers})
        if ptr == "biaffine":
            # Dozat & Manning (2017): bilinear term plus a prior on each candidate
            self.bilinear = nn.ParameterDict({"p_" + p: nn.Parameter(torch.eye(ptr_dim)) for p in pointers})
            self.prior = nn.ParameterDict({"p_" + p: nn.Parameter(torch.zeros(ptr_dim)) for p in pointers})
        # learned bias per word offset, clipped to +-dist (0 = off)
        self.dist = dist
        if dist:
            self.offset = nn.ParameterDict({"p_" + p: nn.Parameter(torch.zeros(2 * dist + 1)) for p in pointers})
        self.ptr = ptr
        self.ptr_dim = ptr_dim

    def forward(self, input_ids, attention_mask, first, word_mask, seg):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        words = torch.gather(h, 1, first.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        words = self.dropout(words)
        out = {name: head(words).float() for name, head in self.heads.items()}
        same_seg = (seg.unsqueeze(2) == seg.unsqueeze(1)) & word_mask.unsqueeze(1)
        valid = torch.cat([torch.ones_like(same_seg[:, :, :1]), same_seg], dim=2)
        if self.dist:
            pos = torch.arange(words.size(1), device=words.device)
            off = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.dist, self.dist) + self.dist
        # autocast would run these matmuls in half, which cannot hold the -1e9 mask
        with torch.autocast("cuda", enabled=False):
            words = words.float()
            for p in pointers:
                q = self.query["p_" + p](words)
                k = self.key["p_" + p](words)
                keys = torch.cat([self.null["p_" + p].expand(k.size(0), 1, -1), k], dim=1)
                if self.ptr == "biaffine":
                    scores = (q @ self.bilinear["p_" + p]) @ keys.transpose(1, 2)
                    scores = scores + (keys @ self.prior["p_" + p]).unsqueeze(1)
                else:
                    scores = q @ keys.transpose(1, 2)
                scores = scores / math.sqrt(self.ptr_dim)
                if self.dist:
                    scores = torch.cat([scores[:, :, :1], scores[:, :, 1:] + self.offset["p_" + p][off]], dim=2)
                out[p] = scores.masked_fill(~valid, -1e9)
        return out


def loss_fn(out, batch, smoothing=0.0):
    """loss_fn sums cross-entropy over every head; labels of -100 are skipped."""
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=smoothing)
    # smoothing would put mass on the masked pointer slots
    plain = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    for name, logits in out.items():
        labels = batch[name]
        if (labels != -100).any():
            f = plain if name in pointers else ce
            total = total + f(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return total


def save(model, tokenizer, vocab, maps, path, args):
    os.makedirs(path, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "maps": maps, "args": args}, f, ensure_ascii=False)


def fetch(path):
    """fetch returns a local folder as is, else pulls the Hugging Face repo it names (cached copy offline)."""
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path)


def load(path, device="cpu"):
    """load returns (model, tokenizer, vocab, maps, args) saved by save, from a folder or a Hugging Face repo id."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    path = fetch(path)
    with open(os.path.join(path, "labels.json"), encoding="utf-8") as f:
        meta = json.load(f)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(path), add_pooling_layer=False)
    a = meta["args"]
    model = IntensifierTagger(encoder, meta["vocab"], a["ptr_dim"], ptr=a.get("ptr", "dot"), dist=a.get("dist", 0))
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
    return model, tokenizer, meta["vocab"], meta["maps"], meta["args"]


@torch.no_grad()
def predict_records(model, tokenizer, vocab, records, max_len, context=64, batch_size=32, device="cpu"):
    """predict_records labels every word of each record (words and seg needed) and returns records in the data/finetune format."""
    jobs = []
    preds = []
    for r, rec in enumerate(records):
        n = len(rec["words"])
        pred = {"speaker": rec["speaker"], "words": rec["words"], "seg": rec["seg"]}
        pred["upos"] = ["X"] * n
        pred["lemma"] = [identity] * n
        pred["host"] = ["O"] * n
        for h in host_heads + pointers:
            pred[h] = [None] * n
        preds.append(pred)
        subword_counts(rec, tokenizer)
        jobs += [(r, w) for w in chunks(rec, max_len, context)]
    # longest first keeps padding low
    jobs.sort(key=lambda j: sum(records[j[0]]["nsub"][j[1][0] : j[1][3]]), reverse=True)
    # models saved before the lexicon have none
    lexicon = vocab.get("lexicon", {"lemma": {}})
    use_amp = device != "cpu" and torch.cuda.is_available()
    for b in range(0, len(jobs), batch_size):
        part = jobs[b : b + batch_size]
        items = [encode(records[r], w, vocab, tokenizer, max_len, labels=False) for r, w in part]
        batch = {k: v.to(device) for k, v in collate(items, tokenizer.pad_token_id).items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(**batch)
        best = {k: v.argmax(-1).cpu().tolist() for k, v in out.items()}
        for row, ((r, (cs, s, e, _)), it) in enumerate(zip(part, items)):
            pred = preds[r]
            for w in range(s - cs, e - cs):
                if not it["word_mask"][w]:
                    continue
                k = cs + w
                pred["upos"][k] = vocab["upos"][best["upos"][row][w]]
                rule = vocab["lemma"][best["lemma"][row][w]]
                pred["lemma"][k] = identity if rule == unk else rule
                pred["host"][k] = vocab["host"][best["host"][row][w]]
                word = pred["words"][k]
                low = word.lower()
                lemma = lexicon["lemma"].get(f"{low}\t{pred['upos'][k]}")
                if lemma is not None:
                    pred["lemma"][k] = edit_rule(word, lemma)
                if pred["host"][k] == "O":
                    continue
                for h in host_heads:
                    v = vocab[h][best[h][row][w]]
                    pred[h][k] = identity if v == unk else v
                    if h in lexicon:
                        pred[h][k] = lexicon[h].get(f"{low}\t{pred['host'][k]}", pred[h][k])
                for p in pointers:
                    j = best[p][row][w]
                    pred[p][k] = -1 if j == 0 else cs + j - 1
    return preds
