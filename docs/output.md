# Output

`pipeline/predict.py` writes `data/output/intensifier_data_model_<date>.sqlite` with the full table (`--output` for another name). `--xlsx` and `--csv` write the same table as `.xlsx` and `.csv`. One row per adjective or adverb token, intensified or not, except comparatives, superlatives, *tan* X *como/que* and other contexts outside the envelope of variation ([annotation.md](annotation.md)); `--drop-invariant` keeps only hosts intensified at least once in the output. `<output>.records.jsonl.gz` next to it holds the words and labels the rows came from, for [the correction loop](finetune.md#the-correction-loop).

| Column | Meaning |
| --- | --- |
| `Speaker` | transcript file name |
| `Phr_tagged`, `Phr` | the segment the token came from, tagged (`word_UPOS_lemma` per token; `_` inside a word becomes `-`) and plain |
| `Int` | intensifier (*muy*, *tan*, *súper*, *-ísimo*, *re-*, *archi-*, *más o menos*, ...) or `/` |
| `Int_type` | `maximizer`, `booster`, `approximator`, `compromiser` or `emphasizer` (Quirk et al. 1985 via Stratton & Sundquist 2022; see [annotation.md](annotation.md#int_type)), or `/`. Emphasizers count as unintensified in Stratton's method |
| `Double` | second intensifier (*muy muy*) or `/` |
| `Adj`, `Lex_Adj` | adjective host as written, and its masculine singular lemma |
| `Adv` | adverb host |
| `Adj_type` | `predicative`, `attributive` or `ambiguous` |
| `Noun` | noun the adjective modifies or describes |
| `Negation` | negator scoping over the host (*no*, *nunca*, *ni*, *tampoco*, *jamás*) or `/`; Stratton & Sundquist (2022) drop these rows, so filter `Negation = '/'` to match them |
| `Verb`, `Inf_Verb` | verb or copula tied to the host, as written and as its infinitive lemma |
| `Token` | word index of the host in the speaker's file; keeps a corrected row on its word |
| `Number` | token id within the speaker (`<Speaker>_01`, ...) |
| `Syllable` | syllables in the host |

`/` means not applicable or not found. The SQLite file is about half the size of the csv: speakers, phrases and short words are stored once, and the `intensifier_data` view rebuilds the table.

```python
import sqlite3
import pandas as pd

df = pd.read_sql(
    "SELECT * FROM intensifier_data", sqlite3.connect("data/output/intensifier_data_model_YYYY-MM-DD.sqlite")
)
```

Speaker metadata columns (`Gender`, `Age`, `Corpus`, ...) come from `--speakers`, a sheet with a `Speaker` column.
