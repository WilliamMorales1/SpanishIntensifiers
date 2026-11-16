# IntensifierResearch

![Tests](https://github.com/WilliamMorales1/IntensifierResearch/actions/workflows/tests.yml/badge.svg)

Finds intensifier + adjective/adverb tokens (*muy bueno*, *riquísimo*, *archiconocido*, *muy muy lejos*) in Spanish sociolinguistic interviews with one fine-tuned transformer. Each adjective is classified as predicative or attributive, and speaker metadata is merged in. Output is one row per adjective or adverb token, a variable envelope ready for variationist analysis.

Results from an earlier, rule-based version are published in [*Proceedings of the LSA* 11(1)](https://doi.org/10.3765/plsa.v11i1.6091); that code is on the [`LSA-2026-paper`](../../tree/LSA-2026-paper) branch, with its citation.

## Dependencies

Python 3.10+ with PyTorch 2.6+, transformers, [Stanza](https://stanfordnlp.github.io/stanza/) 1.15.0 (tokenizer only), pandas, openpyxl (speaker sheets) and python-docx (transcripts). Dev: pytest, ruff. All in `requirements.txt` (conda, uv and pip alike); see [Setup](docs/setup.md).

## Workflow

```fish
conda create -n intensifier-research python=3.12 && conda activate intensifier-research
pip install -r requirements.txt
python pipeline/clean_transcripts.py                       # data/original/*.docx -> data/text/*.txt, from minute 10
python pipeline/predict.py --text data/text --xlsx  # latest model -> data/output/intensifier_data_model_<date>.{sqlite,xlsx}
# correct the xlsx by hand (docs/annotation.md), then add it to the training data:
python pipeline/dataset.py data/output/intensifier_data_model_<date>.records.jsonl.gz <corrected>.xlsx
train/kaggle.sh --model FacebookAI/xlm-roberta-large --batch 8 --lr 2e-5 --epochs 8   # retrain on a Kaggle T4 (colab.sh for Colab)
```

Training needs a GPU (~70 min on a T4); prediction runs on a laptop CPU (a few files per minute, 2.2 GB model). The model and the training data are not in the repo (corpus text); see [Fine-tuned tagger](docs/finetune.md).

To try it without the corpus, `data/examples/` holds one synthetic interview:

```fish
python pipeline/clean_transcripts.py --input data/examples/original --output data/examples/text --skip-words 0
python pipeline/predict.py --text data/examples/text --speakers data/examples/speakers.xlsx --output data/examples/intensifier_data --csv
```

## Layout

| path | contents |
| --- | --- |
| `pipeline/` | `clean_transcripts.py` (docx -> `data/text`), `model.py` and `predict.py` (inference), `dataset.py` (label records, corrected-table merge, scoring), `output.py` (table writers) |
| `train/` | `train.py` and the Kaggle/Colab runners and sweeps |
| `tests/` | pytest suite |
| `data/` (gitignored, corpus text) | `original/` docx, `text/` cleaned text, `finetune/` training data, `eval/`, `output/` model tables |
| `data/examples/` (tracked) | one synthetic interview with its model output |
| `models/` (gitignored) | trained models |

## Intensifier classification

`Int_type` sorts intensifiers into maximizers, boosters, approximators, compromisers and emphasizers. This follows Quirk et al.'s (1985:589-590) scale as operationalized by Stratton & Sundquist (2022) and Stratton (2020). `Negation` records the negator (*no*, *nunca*, *ni*, *tampoco*, *jamás*) that scopes over a host. Negated rows are kept and marked, not dropped as in Stratton & Sundquist (2022:395).

The classification is partly arbitrary. The scale was written for English, and the class boundaries are judgment calls, not facts about Spanish:

- Stratton & Sundquist leave Norwegian *ganske* 'quite' out of their amplifier/downtoner counts because it does both jobs (2022:400, fn. 10), and *utrolig* 'unbelievably' because it is unclear whether it is an emphasizer, booster or maximizer (2022:396). Spanish has the same kind of items, but this code gives each one a single class.
- *bastante* and *demasiado* are boosters because in this Colombian corpus they mostly mean 'very'. Elsewhere they can read as 'fairly' (compromiser) or 'too' (excess).
- *extremadamente* and *sumamente* are maximizers because Quirk lists *extremely* among them. Many studies call them boosters.
- *realmente*, *verdaderamente* and *increíblemente* are boosters, as *really* is in Tagliamonte's work. Quirk calls *really* an emphasizer.
- *-ísimo*, *-érrimo*, *archi-*, *requete-*, *recontra-* and attached *súper-* are boosters, though grammars call *-ísimo* the *superlativo absoluto*.
- *más o menos* is a compromiser, after Quirk's *more or less*.
- Which uses of *tan* and *más* count as intensifiers is a judgment call: comparatives, *tan X como*, *tan X que* result clauses and 'how' readings are dropped. Earlier Spanish work dropped *tan* and *más* entirely (Brown & Cortés-Torres 2013; Orozco, Morales & Valenzuela 2026).

Check every class against your own reading before using `Int_type` in a model. The mapping is `int_type` in the dataset's `maps.json`, read from corrected tables; the classes and references are in [docs/annotation.md](docs/annotation.md#int_type).

## Docs

- [Setup](docs/setup.md): conda, uv, pip, CPU-only PyTorch, tests.
- [Cleaning](docs/cleaning.md): every transcript cleaning rule and the minute-10 cut.
- [Fine-tuned tagger](docs/finetune.md): the model, the correction loop, training on Colab/Kaggle, results.
- [Annotation](docs/annotation.md): which tokens are rows and how each column is filled.
- [Output](docs/output.md): columns and reading the SQLite file.

## License

MIT
