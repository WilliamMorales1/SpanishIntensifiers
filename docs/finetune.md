# Fine-tuned intensifier tagger

One transformer model that reads plain words of a cleaned transcript and outputs one row per host adjective or adverb: its intensifier, `Adj_type`, `Noun`, `Verb`, `Negation` and lemmas (columns in [output.md](output.md), conventions in [annotation.md](annotation.md)). No tagger, rules or parse at inference. `Int_type` is not a model output: `predict.py` adds it from `Int` afterwards ([annotation.md](annotation.md#int_type)).

The first training set was the output of an earlier rule pipeline, mapped onto token positions, with 542 hand-fixed rows. From here on the training data grows by hand: run the latest model, correct its rows, merge them in, retrain ([the correction loop](#the-correction-loop)).

## Files

| file | what it does |
|---|---|
| `pipeline/clean_transcripts.py` | docx -> `data/text` ([cleaning.md](cleaning.md)) |
| `pipeline/dataset.py` | the training data: per-token label records, `decode_rows` (labels -> rows), `apply_rows` (corrected rows -> labels), `merge` (into `data/finetune/`), `compare_rows` (scores rows against rows) |
| `pipeline/output.py` | speaker metadata merge, `Number`/`Syllable`, `--drop-invariant`, sqlite/csv/xlsx writers |
| `pipeline/model.py` | the model: encoder + heads, chunking, save/load, `predict_records` |
| `train/train.py` | training loop, dev score per epoch, keeps the best epoch, test score |
| `pipeline/predict.py` | runs a trained model over a folder; writes the table and the label records |
| `train/run.sh` | trains the released config in the current env on any GPU machine; extra args go to `train.py` |
| `train/colab.sh` | trains and predicts on a Colab GPU and downloads the results |
| `train/kaggle.sh` | the same on a Kaggle T4: private dataset + private script kernel |
| `train/sweep.sh`, `kaggle_sweep.sh` | train a list of configs (`name args...` per line) and keep each `metrics.json` in `models/sweep/<name>/`; finished configs are skipped on rerun, `WATCH=1` only follows a running job. Kaggle runs one config per T4 of the T4 x2 |
| `train/job.sh`, `sweep.py` | what runs on the VM: train + pack + predict, and the config-list runner (one config per GPU) |
| `train/lib.sh`, `colab_setup.py`, `kaggle_kernel.py` | shared by the runners: packing, model install, Colab start and polling, Kaggle push and polling; the VM-side entry points that unpack the job and run its `cmd.sh` |

## What the model predicts

Per word (first subword of a RoBERTa encoder, default `bertin-project/bertin-roberta-base-spanish`, the model Stanza's tagger uses; the current model is `FacebookAI/xlm-roberta-large`, see Results):

- `upos`: the corrected UPOS (goes into `Phr_tagged`).
- `lemma`: an edit rule (cut/add prefix and suffix, as in UDPipe), so unseen words still get lemmas. Words seen in training with the predicted UPOS take their most common training lemma instead (`build_lexicon`, saved in `labels.json`): edit rules invent lemmas for irregular verbs (*sean* -> *sear*, *fuimos* -> *fuer*), and on dev the lexicon cut wrong `Inf_Verb` from 835 to 679 of 7,853 rows.
- `host`: `O`, `ADJ` or `ADV`.

Per host word:

- `bound`: a bound intensifier (`-ísimo`, `súper`, `re`...) or `/`.
- `adj_type`: attributive, predicative or ambiguous.
- `form`, `lex`: edit rules from the word to `Adj`/`Adv` and `Lex_Adj` (this is how *-ísimo* bases, -dora lemmas and PR lateralized lemmas are learned); a host word seen in training as the same host class takes its most common training rule (dev wrong `Lex_Adj` 99 -> 86).
- pointers to the word that is its intensifier, double intensifier, negator, noun and verb, or to nothing. A pointer can only land in the host's own Phr segment (the text between `,` `.` `!` `?`), as in the rules.

Training text is `data/text` after `normalize_text` ([cleaning.md](cleaning.md#text-normalization-normalize_text-2026-11-15)): no headers, annotator notes, `[T]` marks, brackets, laughter or `xxx` (now pauses), emphasis capitals, split words or missing accents on unambiguous words. Input words get one more cleaning step: `clean_words` rejoins transcript splits from `split_words` (*distin to*, *lejo s*, *muy bue no*). Everything else (tags, lemmas, segments) comes from the words.

`Int` and `Double` are the intensifier words mapped through `maps.json` (*mas* -> *más*; an `Int` *menos* is *más o menos*, a `Double` *menos* stays *menos*); both maps live in `maps.json` of the dataset, and new intensifiers in corrected tables are added to them. `Inf_Verb` is the predicted lemma of the verb word.

## 1. The training data

`data/finetune/` (gitignored, corpus text): `{train,dev,test}.jsonl.gz` (one label record per speaker file), `maps.json` and `splits.json`. Splits are by speaker file, stratified by corpus (MEDE, TCali, TUNJ, CTG, TrCTG, LAPR, and the numbered Cartagena files as `CTG-num`): 245 train, 32 dev, 32 test (`split_speakers`, seed 13). Keep dev and test fixed so scores stay comparable across rounds; add new files to train.

A record holds `words`, `upos`, `lemma` (edit rules), `seg` and per host word `bound`, `adj_type`, `form`, `lex`, `int`, `double`, `neg`, `noun`, `verb` (pointers are word indices, -1 for none).

## The correction loop

1. Clean new transcripts (`python pipeline/clean_transcripts.py`), or pick files of `data/text` whose rows need fixing.
2. Run the latest model with `--xlsx` (or `--csv`):

   ```fish
   python pipeline/predict.py --text data/text --only <speakers> --output data/output/round2 --xlsx
   ```

   Writes `data/output/round2.{sqlite,xlsx}` and `data/output/round2.records.jsonl.gz` (the words and tokenization the rows point into).
3. Correct the table by hand ([annotation.md](annotation.md)):
   - fix any cell of a row; keep its `Token` (word index in the file) so it is matched to its word;
   - delete a row that is not a host;
   - add a missed host as a new row with `Token` blank and `Speaker`, `Phr`, and `Adj` or `Adv` filled: it is matched by `Phr` (punctuation ignored) and the word or its lemma (*rico* finds *riquísimo*);
   - `Int`, `Double`, `Noun`, `Negation`, `Verb` must be words of the host's `Phr`, before the host for `Int`, `Double`, `Negation`; an `Int` not in the segment is a bound intensifier (`-ísimo`, `súper`, `archi`);
   - fixing a tag in `Phr_tagged` (`word_UPOS_lemma`, same token count) fixes the record's UPOS and lemma.
   - speaker metadata, `Number` and `Syllable` are ignored (rebuilt on output), and so is `Int_type` (from `pipeline/int_type.json`; add a new intensifier there).
4. Merge into the dataset:

   ```fish
   python pipeline/dataset.py data/output/round2.records.jsonl.gz data/output/round2.xlsx
   ```

   Every table speaker found in the records replaces its file in the dataset, keeping that file's split; new files go to `--split` (default `train`). Options: `--dataset data/finetune` (made if missing). It prints the problems it found (`host not found`, `two rows on one host`, `<column> not found`) with speaker and row; fix those rows and rerun, since the merge rewrites the files each time.
5. Retrain (next section), compare `metrics.json` with the previous model, repeat.

A whole file counts as checked once merged: rows left uncorrected are taken as right, so correct every row of a file, not a sample.

## 2. Train on Colab (~30-60 min on a T4)

Needs the `colab` CLI authenticated (see the `colab` skill) and no other GPU session holding your quota.

```fish
mkdir -p data/colab_finetune
tmux new -d -s intensifier-ft -c (pwd) 'conda run --no-capture-output -n intensifier-research train/colab.sh > data/colab_finetune/run.log 2>&1'
tail -f data/colab_finetune/run.log
```

`colab.sh` needs `data/finetune/train.jsonl.gz`, packs `pipeline/`, `train/`, `data/finetune/` and `data/text/` into `data/colab_finetune/job.tgz`, opens a fresh session (`COLAB_SESSION`, default `intensifier-ft`; `COLAB_GPU`, default `T4`), trains, runs `predict.py` over `data/text`, then downloads and unpacks:

- `models/intensifier/`: `model.pt`, encoder config, tokenizer, `labels.json` (label vocabularies, maps, training args), `metrics.json` (dev scores per epoch, test scores).
- `data/output/intensifier_data_model_<date>.sqlite`.

It polls every 5 min. As soon as training ends it downloads the model into `models/intensifier/` on this machine (a previous model is moved to `models/intensifier.old/`), before prediction starts, so a failed predict or a lost VM does not lose it. It replaces a VM preempted during training up to 3 times (training restarts from scratch); a VM lost after training ends the script, and `predict.py` can then run locally. It stops the VM when done, and on a failed job leaves it running and saves `data/colab_finetune/job.log`. Arguments are passed to `train.py`:

```fish
train/colab.sh --epochs 6 --lr 2e-5
```

On Kaggle instead (needs the `kaggle` CLI with a phone-verified account; dataset and kernel stay private):

```fish
tmux new -d -s intensifier-ft -c (pwd) 'conda run --no-capture-output -n intensifier-research train/kaggle.sh --model FacebookAI/xlm-roberta-large --batch 8 --lr 2e-5 --epochs 8 > data/kaggle_finetune.log 2>&1'
```

It uploads the same tarball as dataset `<user>/intensifier-ft-data`, runs `kaggle_kernel.py` as kernel `<user>/intensifier-ft` (`job.sh`: train, then predict), polls every 5 min and, once the kernel is COMPLETE, unpacks `models/intensifier/` (old one to `models/intensifier.old/`) and the sqlite. Kaggle returns output only after the run ends, so a failed predict means rerunning `predict.py` locally from the downloaded model. `WATCH=1` follows a kernel already pushed. `kaggle_sweep.sh data/finetune/exp/configs.txt` runs a config list the same way (`models/sweep/<name>/` skipped on rerun).

`train.py` options: `--model` (any HF encoder with a fast tokenizer, e.g. `PlanTL-GOB-ES/roberta-base-bne`), `--epochs 4`, `--batch 16`, `--lr 3e-5` (encoder), `--head-lr 5e-4`, `--warmup 0.1`, `--max-len 512` (subwords per window, capped by the encoder's positions), `--context 64` (unlabelled subwords seen on each side of a window), `--ptr-dim 256`, `--min-rule 2` (rarer lemma edit rules become "keep the word"), `--ptr dot|biaffine` (pointer scorer; biaffine adds an MLP, bilinear term and candidate prior after Dozat & Manning 2017), `--dist N` (learned pointer bias per word offset up to +-N, 0 = off), `--smoothing 0.1` (label smoothing of the classification heads, 0 = off; pointers stay plain), `--llrd 0.9` (encoder layer-wise lr decay, 1 = off), `--seed`, `--limit N` (N files per split, for smoke tests), `--device`.

The best epoch is the one with the highest dev **select** score: the mean of row F1 (a predicted row counts only when the host is the same token and every column but `Phr_tagged` matches the rule row) and intensifier F1 (right intensifier word on an intensified row). Row F1 alone is dominated by bare rows, which are ~90% of rows.

### Class imbalance

The labels are very unbalanced (train split, per word or per host row):

| head | majority class | share | dev accuracy of always-majority | of guessing by train frequencies |
|---|---|---|---|---|
| host | `O` | 94.1% | 94.4% | 89.0% |
| intensifier pointer | none | 90.5% | 90.3% | 82.7% |
| double | none | 99.8% | | |
| negation | none | 95.5% | | |
| bound | `/` | 99.8% (157 *-ísimo*, 5 *súper*, 1 *re*) | | |
| adj_type | attributive | 45.8% (predicative 35.5, ambiguous 18.7) | 45.2% | 37.1% |
| noun pointer | none | 77% | | |

So a per-column accuracy of 95% on `Double`, `Negation` or `Int` says almost nothing. `compare_rows` reports, next to each column's accuracy, the always-majority accuracy (`majority`) and Cohen's kappa (`kappa`: accuracy above what guessing by each side's label frequencies gets, 0 = chance, 1 = perfect); P/R/F1 of the rows where `Double`, `Negation`, `Noun` and `Verb` have a value (`filled`); `Adj_type` macro F1 over the three classes; the share of adjective rows intensified, gold vs model (`adj_intensified_rate`, the study's main number); and row F1 and that share per corpus (`by_corpus`), since corpora differ in size (MEDE 37% of files, CTG 12%) and rate (MEDE ~24% vs CTG ~9%).

The loss stays plain, unweighted cross-entropy. Class weights or focal loss raise recall of rare classes by making the model predict them more often than they occur, which would push `adj_intensified_rate` up: a biased rate is worse here than a few missed rare cases. A pretrained encoder learns 6% hosts and 10% intensified hosts without help, and the rare bound intensifiers are also covered by the lemma/form rules. No input normalization is needed: the encoder is cased and accent-aware, and spelling variants (*mas*/*más*) are mapped at decoding. If a run shows a calibrated rate but low recall on `Double` or `bound`, try weights only on those heads.

Training on this laptop's CPU is not practical for the base model (days). A smoke test with a tiny random model checks the code in a few minutes:

```fish
python train/train.py --model hf-internal-testing/tiny-random-RobertaModel --limit 2 --epochs 1 --batch 8 --out /tmp/tiny --device cpu
```

## 3. Predict

On Colab `colab.sh` already did it. Locally (CPU works for inference, slowly; ~1.5 M words take hours, a few files minutes):

```fish
# cleaned transcripts: Stanza's tokenizer only (no tagger, no parse)
python pipeline/predict.py --text data/text --speakers data/speaker_data_ALL_2026-3-12.xlsx
python pipeline/predict.py --text data/text --only MEDE_H01_2_1991m_2007 TCali_H11_01_1996b_2019 --xlsx
```

Options: `--model` (Hugging Face repo id or local folder; default `Hydragon123/spanish-intensifier-tagger`, private, pulled into the HF cache before each run, cached copy used offline; `--model models/intensifier` for a freshly trained local model), `--output data/output/intensifier_data_model_<date>`, `--speakers`, `--drop-invariant`, `--xlsx`, `--csv`, `--batch 32`, `--device`, `--only` (speaker names), `--words` (folder of `word_UPOS_lemma` files instead of `--text`; only the words are used). Columns in [output.md](output.md), plus `Token`. Next to the table it writes `<output>.records.jsonl.gz`, the input of `pipeline/dataset.py`.

New transcripts: clean them with `pipeline/clean_transcripts.py` as usual (labels, interviewer turns, minute-10 cut, `normalize_text`), then `--text`. Text not run through it is noisier than the training text.

## 4. Evaluate

`models/intensifier/metrics.json`: dev and test scores against the dataset rows (host P/R/F1, intensified-row P/R/F1, intensifier word, per-column accuracy with majority baseline and kappa, filled-column F1, `Adj_type` macro F1, intensified rate, per corpus, row F1). Test files are never trained on or used to pick the epoch. Hand-corrected dev and test files make these scores gold scores; until then they are scores against the old rule output.

`data/eval/adj_gold.tsv` (500 hand-labelled ADJ rows) is a test set only: never merge its files' corrections into train or pick settings by it.

## Results

Current model (2026-11-15): `--model FacebookAI/xlm-roberta-large --batch 8 --lr 2e-5 --epochs 8`, trained on Kaggle (~70 min on a T4, prediction over the corpus ~10 min; 2.2 GB). Chosen on dev from a sweep of epochs, pointer scorer, distance bias, label smoothing, layer-wise lr decay, seeds and encoders (`data/finetune/IMPROVE_PLAN.md`); only more epochs and the large encoder beat the seed gap. Test split, against the rule rows:

| | v1 (bertin base, 4 epochs, no lexicon) | current |
|---|---|---|
| select | .880 | .912 |
| row F1 | .798 | .848 |
| intensifier F1 | .963 | .977 |
| host F1 | .961 | .965 |
| `Adj_type` macro F1 | .904 | .901 |
| adjective rows intensified (rules .199) | .197 | .201 |
| gold test split, `Adj_type` right | 94.1% (32 of 34) | 94.1% (32 of 34) |

Rules on the same gold rows: 94.3% (33 of 35). Corpus output: 87,884 rows, 8,582 intensified (rules: 87,495 and 8,415).

## Limits

- The first training set is mostly rule output: outside the 542 hand-fixed rows, rule errors are learned as targets (secondary predicates as `amod`, the ~6% confidently wrong `Adj_type`). Corrected files replace them round by round.
- Segments come from punctuation tokens only. Windows hold whole segments up to 382 subwords plus 64 subwords of context per side; a longer unpunctuated stretch is cut, and a pointer can reach into the context. `train.py` prints how many pointer targets still fall outside their window (0 on the corpus).
- The test-split gold check has 51 rows: a wide interval.
